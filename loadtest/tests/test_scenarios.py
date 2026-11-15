"""Exercise the real k6 scripts against a disposable HTTP API (Docker required)."""

# These tests also run with stdlib unittest, without installing server dependencies.
# ruff: noqa: PT009

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
from threading import Thread
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CapacityScenariosTest(unittest.TestCase):
    def run_scenario(
        self,
        script: str,
        *,
        upload_statuses: tuple[int, ...] = (200, 429, 503),
        bad_login: bool = False,
        capacity_errors: bool = True,
        **settings: str,
    ) -> tuple[subprocess.CompletedProcess[str], list[tuple[str, str | None, bytes]], dict[str, Any]]:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                status = 200
                headers = {}
                if self.path == "/auth/login":
                    email = json.loads(body)["email"]
                    status = 401 if bad_login else 200
                    payload = {"access_token": email}
                elif self.path == "/documents":
                    index = sum(path == "/documents" for path, _, _ in requests)
                    status = upload_statuses[index % len(upload_statuses)]
                    payload = {"status": "processing", "document_id": index + 1, "filename": "test.txt"}
                    if status in (429, 503) and capacity_errors:
                        headers["Retry-After"] = "30"
                        payload = {"message": "Background job capacity reached; retry later"}
                else:
                    payload = {"answer": "Test answer"}
                requests.append((self.path, self.headers.get("Authorization"), body))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        with (
            tempfile.TemporaryDirectory() as directory,
            ThreadingHTTPServer(("127.0.0.1", 0), Handler) as api,
        ):
            fixtures = Path(directory)
            (fixtures / "users.json").write_text(
                json.dumps(
                    [
                        {"email": "uploader", "password": "test", "role": "curator"},
                        {"email": "chat-1", "password": "test", "role": "user"},
                        {"email": "chat-2", "password": "test", "role": "user"},
                    ]
                )
            )
            worker = Thread(target=api.serve_forever, daemon=True)
            worker.start()
            env = {
                "BASE_URL": f"http://127.0.0.1:{api.server_port}",
                "USERS_FILE": "/fixtures/users.json",
                "UPLOAD_COUNT": "12",
                "UPLOAD_VUS": "2",
                "CHAT_VUS": "2",
                "DURATION": "2s",
                "CHAT_PAUSE_SECONDS": "0.05",
                "API_REPLICAS": "2",
                "CHAT_VUS_PER_REPLICA": "1",
                **settings,
            }
            command = [
                "docker",
                "run",
                "--rm",
                "--network=host",
                "--user",
                "0",
                "-v",
                f"{ROOT}:/scripts:ro",
                "-v",
                f"{fixtures}:/fixtures",
            ]
            for key, value in env.items():
                command += ["-e", f"{key}={value}"]
            try:
                result = subprocess.run(  # noqa: S603 -- fixed image, repository scripts and local fixtures
                    [
                        *command,
                        "grafana/k6:0.54.0",
                        "run",
                        "--quiet",
                        "--summary-export=/fixtures/summary.json",
                        f"/scripts/k6/{script}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=40,
                    check=False,
                )
            finally:
                api.shutdown()
                worker.join()
            summary_path = fixtures / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        return result, requests, summary

    def test_upload_burst_uses_one_principal_while_other_users_chat(self) -> None:
        result, requests, summary = self.run_scenario(
            "upload-chat.js",
            UPLOAD_COUNT="1000",
            UPLOAD_VUS="4",
            DURATION="5s",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        uploads = [(auth, body) for path, auth, body in requests if path == "/documents"]
        chats = [auth for path, auth, _ in requests if path == "/chat/sync"]
        self.assertEqual(len(uploads), 1000)
        self.assertEqual({auth for auth, _ in uploads}, {"Bearer uploader"})
        self.assertEqual(set(chats), {"Bearer chat-1", "Bearer chat-2"})
        questions = [json.loads(body)["question"] for path, _, body in requests if path == "/chat/sync"]
        self.assertEqual(len(questions), len(set(questions)))
        self.assertEqual(len({body for _, body in uploads}), 1000)
        metrics = summary["metrics"]
        self.assertEqual(metrics["uploads_accepted"]["count"], 334)
        self.assertEqual(metrics["uploads_principal_backlog"]["count"], 333)
        self.assertEqual(metrics["uploads_global_backlog"]["count"], 333)
        self.assertEqual(metrics["upload_unexpected_errors"]["value"], 0)

    def test_unrelated_server_errors_fail_the_burst(self) -> None:
        result, requests, summary = self.run_scenario("upload-chat.js", upload_statuses=(500,))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any(path == "/documents" for path, _, _ in requests))
        self.assertEqual(summary["metrics"]["upload_unexpected_errors"]["value"], 1)

    def test_unrelated_429_and_503_fail_the_burst(self) -> None:
        for status in (429, 503):
            with self.subTest(status=status):
                result, _, summary = self.run_scenario(
                    "upload-chat.js",
                    upload_statuses=(status,),
                    capacity_errors=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(summary["metrics"]["upload_unexpected_errors"]["value"], 1)

    def test_failed_login_stops_before_producing_load(self) -> None:
        result, requests, _ = self.run_scenario("upload-chat.js", bad_login=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Login failed", result.stderr)
        self.assertFalse(any(path != "/auth/login" for path, _, _ in requests))

    def test_replica_scenario_produces_chat_from_distinct_principals(self) -> None:
        result, requests, _ = self.run_scenario("replicas.js")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        chats = [auth for path, auth, _ in requests if path == "/chat/sync"]
        self.assertEqual(set(chats), {"Bearer uploader", "Bearer chat-1"})
        self.assertFalse(any(path == "/documents" for path, _, _ in requests))

    def test_insufficient_distinct_users_fails_before_load(self) -> None:
        result, requests, _ = self.run_scenario("replicas.js", API_REPLICAS="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("distinct users", result.stderr)
        self.assertFalse(any(path != "/auth/login" for path, _, _ in requests))


if __name__ == "__main__":
    unittest.main()
