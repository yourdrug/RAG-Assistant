"""
Locust load test for RAG /chat (SSE streaming) endpoint.

Measures:
- time_to_first_byte (TTFB)
- time_to_last_byte (full stream time)
- Stream errors: broken connections, timeouts

Run:
  locust -f locustfile.py --host=http://host.docker.internal:8001
  # or headless:
  locust -f locustfile.py --host=http://host.docker.internal:8001 \
    --users 200 --spawn-rate 10 --run-time 10m \
    --headless --csv=results/sse
"""

import json
import random
import time
from pathlib import Path

import requests
from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8001"

DATA_DIR = Path(__file__).parent.parent / "data"
USERS_FILE = DATA_DIR / "test_users.json"
QUESTIONS_FILE = DATA_DIR / "test_questions.json"

# ---------------------------------------------------------------------------
# Shared state — loaded once per worker process
# ---------------------------------------------------------------------------

_test_users: list[dict] = []
_questions: list[str] = []
_token_cache: dict[str, str] = {}


def _load_json(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text())
    return []


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global _test_users, _questions
    _test_users = _load_json(USERS_FILE) or [
        {"email": "admin", "password": "admin"},
    ]
    _questions = _load_json(QUESTIONS_FILE) or [
        "Что такое RAG?",
        "Объясни архитектуру системы",
        "Какие документы загружены в базу?",
    ]
    print(f"Loaded {len(_test_users)} users, {len(_questions)} questions")
    print(f"Host: {environment.host}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    if exception:
        print(f"ERROR {request_type} {name}: {response_time:.0f}ms - {exception}")


# ---------------------------------------------------------------------------
# User class
# ---------------------------------------------------------------------------


class ChatStreamUser(HttpUser):
    """
    Simulates a real user:
    1. Login (once, cached)
    2. Send question via /chat (SSE stream) or /chat/sync
    3. Measure TTFB and total stream time
    4. Think time between requests
    """

    wait_time = between(2, 8)
    weight = 3  # 3x more users doing streaming vs sync

    def on_start(self):
        self.user_data = _test_users[self.environment.runner.user_count % len(_test_users)]
        self.token = self._login()
        self.conversation_id = None

    def _login(self) -> str:
        cache_key = self.user_data["email"]
        if cache_key in _token_cache:
            return _token_cache[cache_key]

        host = self.environment.host or BASE_URL
        try:
            resp = requests.post(
                f"{host}/auth/login",
                json={
                    "email": self.user_data["email"],
                    "password": self.user_data["password"],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                token = resp.json()["access_token"]
                _token_cache[cache_key] = token
                return token
        except Exception as e:
            print(f"Login failed for {self.user_data['email']}: {e}")

        return ""

    @task(3)
    def chat_stream(self):
        """POST /chat — SSE streaming response."""
        if not self.token:
            return

        question = random.choice(_questions)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {"question": question}
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id

        start_time = time.monotonic()
        ttfb = None
        full_text = ""
        error = None

        try:
            with self.client.post(
                "/chat",
                json=payload,
                headers=headers,
                stream=True,
                timeout=120,
                catch_response=True,
                name="/chat [SSE]",
            ) as response:
                if response.status_code == 429:
                    response.failure("Rate limited (429)")
                    return
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                    return

                for line in response.iter_lines():
                    if not line:
                        continue

                    if ttfb is None:
                        ttfb = (time.monotonic() - start_time) * 1000

                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "text" in data:
                                full_text += data["text"]
                            if "conversation_id" in data:
                                self.conversation_id = data["conversation_id"]
                        except json.JSONDecodeError:
                            pass
                    elif line.startswith("event: done"):
                        pass
                    elif line.startswith("event: error"):
                        error = line

                total_time = (time.monotonic() - start_time) * 1000

                if error:
                    response.failure(f"Stream error: {error}")
                elif not full_text:
                    response.failure("Empty response")
                else:
                    response.success()

        except requests.exceptions.Timeout:
            total_time = (time.monotonic() - start_time) * 1000
            self.environment.runner.quit()
        except Exception as e:
            total_time = (time.monotonic() - start_time) * 1000
            print(f"Stream error: {e}")

    @task(1)
    def chat_sync(self):
        """POST /chat/sync — blocking response (baseline comparison)."""
        if not self.token:
            return

        question = random.choice(_questions)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        with self.client.post(
            "/chat/sync",
            json={"question": question},
            headers=headers,
            timeout=120,
            name="/chat/sync",
        ) as response:
            if response.status_code == 429:
                response.failure("Rate limited (429)")
            elif response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def list_documents(self):
        """GET /documents — list indexed documents."""
        if not self.token:
            return

        headers = {"Authorization": f"Bearer {self.token}"}
        with self.client.get(
            "/documents",
            headers=headers,
            timeout=10,
            name="/documents",
        ) as response:
            if response.status_code == 429:
                response.failure("Rate limited (429)")
            elif response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
