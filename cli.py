#!/usr/bin/env python3
"""CLI tool to run the deep-fake detector on a local file."""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def wait_for_server(base_url: str, timeout: float = 120) -> bool:
    """Wait until the server is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Deep-fake detector CLI")
    parser.add_argument("file", help="Path to image or video file")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--no-server", action="store_true", help="Don't start server (assume it's already running)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    base_url = f"http://{args.host}:{args.port}"
    server_proc = None

    if not args.no_server:
        print("Starting server...")
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "app.main:app",
                "--host", args.host,
                "--port", str(args.port),
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if not wait_for_server(base_url):
            print("Error: server failed to start within timeout", file=sys.stderr)
            server_proc.terminate()
            sys.exit(1)

        print("Server ready.")

    try:
        filename = os.path.basename(args.file)
        with open(args.file, "rb") as f:
            response = requests.post(
                f"{base_url}/detect",
                files={"file": (filename, f)},
                timeout=60,
            )

        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
            sys.exit(1)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
