#!/usr/bin/env python3
"""Benchmark script: starts server, runs detection on all files in /media concurrently, saves results to CSV."""

import asyncio
import csv
import os
import subprocess
import sys
import time

import aiohttp

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
_HOST_ENV = os.environ.get("HOST", "")
PORT = int(os.environ.get("PORT", "8000"))
if os.environ.get("URL"):
    BASE_URL = os.environ["URL"].rstrip("/")
elif _HOST_ENV.startswith("http://") or _HOST_ENV.startswith("https://"):
    BASE_URL = _HOST_ENV.rstrip("/")
elif _HOST_ENV:
    BASE_URL = f"http://{_HOST_ENV}:{PORT}"
else:
    BASE_URL = f"http://127.0.0.1:{PORT}"
MAX_CONCURRENT = 4


async def wait_for_server(timeout: float = 300) -> bool:
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("status") == "ok":
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1)
    return False


def get_ground_truth(filename: str) -> str:
    return "fake" if filename.lower().startswith("fake") else "real"


async def detect_file(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    filepath: str,
    filename: str,
    index: int,
    total: int,
) -> list[dict]:
    """Send a single file to /detect and return parsed result rows."""
    ground_truth = get_ground_truth(filename)

    async with semaphore:
        print(f"[{index}/{total}] {filename} (actual: {ground_truth})...")

        data = aiohttp.FormData()
        data.add_field("file", open(filepath, "rb"), filename=filename)

        try:
            async with session.post(
                f"{BASE_URL}/detect",
                data=data,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"  ERROR {response.status}: {text}")
                    return []

                body = await response.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            return []

    results = body["results"]

    # Group results by model
    models: dict[str, dict[str, float]] = {}
    for r in results:
        model = r["model_used"]
        if model not in models:
            models[model] = {"fake": 0.0, "real": 0.0}
        models[model][r["label"]] = r["score"]

    rows = []
    for model, scores in models.items():
        fake_score = scores.get("fake", 0.0)
        real_score = scores.get("real", 0.0)
        prediction = "fake" if fake_score >= real_score else "real"
        correct = prediction == ground_truth

        rows.append({
            "model": model,
            "file": filename,
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "prediction": prediction,
            "actual": ground_truth,
            "correct": correct,
        })

        status = "OK" if correct else "WRONG"
        print(f"  {model}: fake={fake_score:.4f} real={real_score:.4f} -> {prediction} [{status}]")

    print()
    return rows


async def run_benchmark():
    files = sorted(f for f in os.listdir(MEDIA_DIR) if not f.startswith("."))
    if not files:
        print("No files found in media/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files in media/")

    server_proc = None
    use_remote = os.environ.get("URL") is not None or os.environ.get("HOST") is not None

    if use_remote:
        print(f"Using remote server: {BASE_URL}\n")
    else:
        # Check if server is already running
        try:
            import requests
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("Server already running.\n")
                use_remote = True
        except Exception:
            pass

        if not use_remote:
            print("Starting server...")
            server_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "app.main:app",
                    "--host", "127.0.0.1",
                    "--port", str(PORT),
                ],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
            )

            if not await wait_for_server():
                print("Error: server failed to start", file=sys.stderr)
                server_proc.terminate()
                sys.exit(1)

            print("Server ready.\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_rows = []

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                detect_file(session, semaphore, os.path.join(MEDIA_DIR, f), f, i, len(files))
                for i, f in enumerate(files, 1)
            ]
            results = await asyncio.gather(*tasks)
            for rows in results:
                all_rows.extend(rows)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "file", "fake_score", "real_score", "prediction", "actual", "correct"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Results saved to {OUTPUT_CSV}")

    # Print summary per model
    print("\n=== Summary ===")
    model_names = sorted(set(r["model"] for r in all_rows))
    for model in model_names:
        model_rows = [r for r in all_rows if r["model"] == model]
        correct = sum(1 for r in model_rows if r["correct"])
        total = len(model_rows)
        print(f"{model}: {correct}/{total} correct ({100 * correct / total:.1f}%)")


def main():
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
