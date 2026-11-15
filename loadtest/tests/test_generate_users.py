"""Generated credentials must work on repeated setup, including the uploader."""

# Runnable with stdlib unittest as well as pytest.
# ruff: noqa: PT009

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Thread
import unittest


class GenerateUsersTest(unittest.TestCase):
    def test_repeated_setup_preserves_passwords_and_creates_a_curator(self) -> None:
        accounts = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def respond(self, payload: object) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_GET(self) -> None:
                self.respond(list(accounts.values()))

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/auth/users":
                    accounts[body["email"]] = body
                    self.respond(body)
                else:
                    self.respond({"access_token": "admin-token"})

        with (
            tempfile.TemporaryDirectory() as directory,
            ThreadingHTTPServer(("127.0.0.1", 0), Handler) as api,
        ):
            script = Path(directory) / "generate_users.py"
            shutil.copy(Path(__file__).resolve().parents[1] / "data/generate_users.py", script)
            worker = Thread(target=api.serve_forever, daemon=True)
            worker.start()
            try:
                command = [
                    sys.executable,
                    str(script),
                    "--count",
                    "3",
                    "--base-url",
                    f"http://127.0.0.1:{api.server_port}",
                ]
                for _ in range(2):
                    result = subprocess.run(  # noqa: S603 -- copied repository script and local HTTP server
                        command,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                saved = json.loads((Path(directory) / "test_users.json").read_text())
                self.assertEqual(accounts[saved[0]["email"]]["role"], "curator")
                self.assertEqual(saved[0]["role"], "curator")
                for user in saved:
                    self.assertEqual(user["password"], accounts[user["email"]]["password"])
            finally:
                api.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
