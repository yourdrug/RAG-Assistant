#!/usr/bin/env python3
"""Generate test users and questions for load testing.

Usage:
  python generate_users.py --count 50 --base-url http://localhost:8001

Creates test_users.json and test_questions.json in ../data/.
If users don't exist on the server, creates them via /auth/users (admin required).
"""

import argparse
import json
import logging
from pathlib import Path
import secrets
import string
import sys
import urllib.error
from urllib.parse import urlsplit
import urllib.request


DATA_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


def generate_email(index: int, prefix: str = "loadtest") -> str:
    return f"{prefix}_user_{index:04d}@test.local"


def generate_password(length: int = 12) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def request_json(method: str, url: str, data: dict | None = None, headers: dict | None = None) -> dict | list:
    if urlsplit(url).scheme not in {"http", "https"}:
        message = "API URL must use HTTP or HTTPS"
        raise ValueError(message)
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)  # noqa: S310 -- scheme checked above
    if body:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 -- operator-provided HTTP(S) API
        return json.loads(resp.read())


def login_admin(base_url: str, email: str, password: str) -> str:
    result = request_json("POST", f"{base_url}/auth/login", {"email": email, "password": password})
    return result["access_token"]


def list_users(base_url: str, token: str) -> list[dict]:
    return request_json("GET", f"{base_url}/auth/users", headers={"Authorization": f"Bearer {token}"})


def create_user(base_url: str, token: str, email: str, password: str, role: str) -> dict:
    return request_json(
        "POST",
        f"{base_url}/auth/users",
        {"email": email, "password": password, "role": role, "kind": "internal"},
        headers={"Authorization": f"Bearer {token}"},
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate load test data")
    parser.add_argument("--count", type=int, default=50, help="Number of test users")
    parser.add_argument(
        "--prefix", default="loadtest", help="Use a fresh prefix when old credentials are unavailable"
    )
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--admin-email", default="admin", help="Admin email for user creation")
    parser.add_argument("--admin-password", default="admin", help="Admin password")
    parser.add_argument(
        "--skip-server", action="store_true", help="Skip server interaction, just generate files"
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    return args


def provision_users(args: argparse.Namespace, token: str, saved: dict[str, dict]) -> list[dict]:
    existing = list_users(args.base_url, token)
    existing_by_email = {u["email"]: u for u in existing}
    logger.info("Found %d existing users on server", len(existing))
    users = []
    for i in range(args.count):
        email = generate_email(i, args.prefix)
        role = "curator" if i == 0 else "user"
        password = saved[email]["password"] if email in saved else generate_password()
        if email in existing_by_email:
            if email not in saved:
                sys.exit("No saved password for an existing user; use --prefix with a new value")
            role = existing_by_email[email]["role"]
            if i == 0 and role not in ("curator", "admin"):
                sys.exit("First user cannot upload; use --prefix to create a new curator")
        else:
            create_user(args.base_url, token, email, password, role)
        users.append({"email": email, "password": password, "role": role})
        logger.info("Prepared %d/%d test users", i + 1, args.count)
    return users


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_arguments()
    users_file = DATA_DIR / "test_users.json"
    saved = {u["email"]: u for u in json.loads(users_file.read_text())} if users_file.exists() else {}

    questions = [
        "Что такое RAG и как он работает?",
        "Объясни архитектуру системы поиска",
        "Какие документы загружены в базу знаний?",
        "Сравни гибридный поиск и простой векторный",
        "Как работает реранжирование результатов?",
        "Опиши процесс индексации документов",
        "Какие форматы файлов поддерживаются?",
        "Как устроена система авторизации?",
        "Что такое BM25 и зачем он нужен в RAG?",
        "Объясни разницу между chunking стратегиями",
        "Как система обрабатывает PDF со сканами?",
        "Опиши процесс embedдинга текста",
        "Как работает GPU-ускорение для инференса?",
        "Что такое collection в Qdrant?",
        "Как настроить crop_size для OCR?",
        "Объясни workflow обработки документов",
        "Какие метрики используются для оценки качества RAG?",
        "Что такое conversation history и зачем она нужна?",
        "Как работает access control для документов?",
        "Опиши процесс deploying RAG-системы",
        "Короткий вопрос",
        "Длинный подробный вопрос с множеством деталей и уточнений, требующий развёрнутого ответа",
        "Как интегрировать внешний LLM вместо Ollama?",
        "Что будет если упадёт Qdrant?",
        "Как масштабировать систему на 10000 пользователей?",
    ]

    # Save questions
    questions_file = DATA_DIR / "test_questions.json"
    questions_file.write_text(json.dumps(questions, ensure_ascii=False, indent=2))
    logger.info("Saved %d questions to %s", len(questions), questions_file)

    if args.skip_server:
        users = [
            {
                "email": generate_email(i, args.prefix),
                "password": generate_password(),
                "role": "curator" if i == 0 else "user",
            }
            for i in range(args.count)
        ]
        users_file = DATA_DIR / "test_users.json"
        users_file.write_text(json.dumps(users, indent=2))
        logger.info("Saved %d fake users to %s", len(users), users_file)
        return

    # Login as admin
    logger.info("Logging in as admin")
    try:
        token = login_admin(args.base_url, args.admin_email, args.admin_password)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        logger.exception(
            "Cannot log in as admin; check that the server is running and credentials are correct"
        )
        sys.exit(1)

    users = provision_users(args, token, saved)

    # Save users file
    users_file = DATA_DIR / "test_users.json"
    users_file.write_text(json.dumps(users, indent=2))
    logger.info("Saved %d test users to %s", len(users), users_file)


if __name__ == "__main__":
    main()
