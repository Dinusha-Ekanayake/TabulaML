"""Deployment smoke test: boot the app headless and check that it serves.

Usage: python scripts/smoke_deploy.py [--port 8599] [--cloud]
Exit code 0 when /_stcore/health answers "ok" and the page HTML loads within the timeout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 60


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - localhost only
        return resp.status, resp.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--cloud", action="store_true", help="simulate TABULAML_CLOUD=1")
    args = parser.parse_args()
    env = {**os.environ, "TABULAML_CLOUD": "1" if args.cloud else "0"}
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.headless",
        "true",
        "--server.port",
        str(args.port),
    ]
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.time() + TIMEOUT_S
        while time.time() < deadline:
            try:
                status, body = _get(f"http://localhost:{args.port}/_stcore/health")
                if status == 200 and body.strip() == "ok":
                    page_status, _ = _get(f"http://localhost:{args.port}/")
                    print(f"OK: health=ok, page={page_status}")
                    return 0 if page_status == 200 else 1
            except OSError:
                time.sleep(1)
        print("FAILED: server did not become healthy in time")
        return 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
