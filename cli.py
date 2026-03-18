#!/usr/bin/env python3
"""CLI tool to run the deep-fake detector on a local file."""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

DEFAULT_HOST = os.environ.get("HOST", "")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))


def wait_for_server(base_url: str, timeout: float = 300) -> bool:
    """Wait until the server is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
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

    # URL or HOST env var overrides and skips local server startup
    env_url = os.environ.get("URL")
    host_env = os.environ.get("HOST", "")
    if env_url:
        base_url = env_url.rstrip("/")
        args.no_server = True
    elif host_env.startswith("http://") or host_env.startswith("https://"):
        base_url = host_env.rstrip("/")
        args.no_server = True
    elif host_env:
        base_url = f"http://{host_env}:{args.port}"
        args.no_server = True
    else:
        base_url = f"http://127.0.0.1:{args.port}"

    server_proc = None

    if not args.no_server:
        # Check if server is already running
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("Server already running.")
                args.no_server = True
        except requests.ConnectionError:
            pass

    if not args.no_server:
        print("Starting server...")
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "app.main:app",
                "--host", "127.0.0.1",
                "--port", str(args.port),
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
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
