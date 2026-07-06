"""
Validate that OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY (from .env)
are present and accepted by their respective APIs.

Each provider is checked with a cheap, read-only endpoint (list models) so
running this script does not consume completion tokens.

Usage:
    python scripts/test_api_keys.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

TIMEOUT = 15


def check_openai() -> tuple[bool, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False, "OPENAI_API_KEY not set"
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=TIMEOUT,
    )
    if resp.status_code == 200:
        return True, "OK"
    return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


def check_anthropic() -> tuple[bool, str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return False, "ANTHROPIC_API_KEY not set"
    resp = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        timeout=TIMEOUT,
    )
    if resp.status_code == 200:
        return True, "OK"
    return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


def check_google() -> tuple[bool, str]:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        return False, "GOOGLE_API_KEY not set"
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": key},
        timeout=TIMEOUT,
    )
    if resp.status_code == 200:
        return True, "OK"
    return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


def main() -> int:
    checks = {
        "OpenAI": check_openai,
        "Anthropic": check_anthropic,
        "Google": check_google,
    }

    all_ok = True
    for name, check in checks.items():
        try:
            ok, detail = check()
        except requests.RequestException as exc:
            ok, detail = False, f"request failed: {exc}"

        all_ok &= ok
        status = "VALID" if ok else "INVALID"
        print(f"[{status}] {name}: {detail}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
