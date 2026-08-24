#!/usr/bin/env bash
# Create test users for load testing.
# Usage: ./setup-users.sh [count] [base-url]
set -euo pipefail

COUNT="${1:-50}"
BASE_URL="${2:-http://localhost:8001}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Creating $COUNT test users at $BASE_URL ==="
python3 "$SCRIPT_DIR/data/generate_users.py" \
  --count "$COUNT" \
  --base-url "$BASE_URL"

echo ""
echo "=== Done ==="
echo "Users saved to: $SCRIPT_DIR/data/test_users.json"
echo "Questions saved to: $SCRIPT_DIR/data/test_questions.json"
