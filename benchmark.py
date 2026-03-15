#!/usr/bin/env python3
"""Benchmark script: starts server, runs detection on all files in /media, saves results to CSV."""

import csv
import os
import subprocess
import sys
import time

import requests

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"


def wait_for_server(timeout: float = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def get_ground_truth(filename: str) -> str:
    return "fake" if filename.lower().startswith("fake") else "real"


def main():
    files = sorted(f for f in os.listdir(MEDIA_DIR) if not f.startswith("."))
    if not files:
        print("No files found in media/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files in media/")
    print("Starting server...")

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", HOST,
            "--port", str(PORT),
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )

    if not wait_for_server():
        print("Error: server failed to start", file=sys.stderr)
        server_proc.terminate()
        sys.exit(1)

    print("Server ready.\n")

    rows = []

    try:
        for i, filename in enumerate(files, 1):
            filepath = os.path.join(MEDIA_DIR, filename)
            ground_truth = get_ground_truth(filename)
            print(f"[{i}/{len(files)}] {filename} (actual: {ground_truth})...")

            with open(filepath, "rb") as f:
                response = requests.post(
                    f"{BASE_URL}/detect",
                    files={"file": (filename, f)},
                    timeout=300,
                )

            if response.status_code != 200:
                print(f"  ERROR {response.status_code}: {response.text}")
                continue

            results = response.json()["results"]

            # Group results by model
            models: dict[str, dict[str, float]] = {}
            for r in results:
                model = r["model_used"]
                if model not in models:
                    models[model] = {"fake": 0.0, "real": 0.0}
                models[model][r["label"]] = r["score"]

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

    finally:
        server_proc.terminate()
        server_proc.wait()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "file", "fake_score", "real_score", "prediction", "actual", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {OUTPUT_CSV}")

    # Print summary per model
    print("\n=== Summary ===")
    model_names = sorted(set(r["model"] for r in rows))
    for model in model_names:
        model_rows = [r for r in rows if r["model"] == model]
        correct = sum(1 for r in model_rows if r["correct"])
        total = len(model_rows)
        print(f"{model}: {correct}/{total} correct ({100 * correct / total:.1f}%)")


if __name__ == "__main__":
    main()
