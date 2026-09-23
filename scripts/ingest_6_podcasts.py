"""Script to ingest all 6 8-minute English podcast episodes into the Audio Search stack via POST /ingest API.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    ("podcast_ep1_python.wav", "Lex Fridman & Guido van Rossum", "Python Language Design & History"),
    ("podcast_ep2_ai.wav", "Bill Gates & Sam Altman", "AI Development & Future of OpenAI"),
    ("podcast_ep3_product.wav", "Lenny Rachitsky & Shreyas Doshi", "Product Management & Career Growth"),
    ("podcast_ep4_neuroscience.wav", "Andrew Huberman & Matthew Walker", "Neuroscience, Brain Performance & Sleep"),
    ("podcast_ep5_technews.wav", "Kevin Roose & Casey Newton", "Tech News, AI Regulation & Industry Shifts"),
    ("podcast_ep6_meta.wav", "Dwarkesh Patel & Mark Zuckerberg", "Llama 3 Open Source AI & Meta Infrastructure"),
]


def ingest_file(filename: str, speakers: str, topic: str):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"[WAIT] {filename} does not exist yet...")
        return None

    size_mb = file_path.stat().st_size / 1e6
    print(f"\n========================================================")
    print(f"Ingesting: {filename} ({size_mb:.2f} MB)")
    print(f"Speakers: {speakers}")
    print(f"Topic: {topic}")
    print(f"========================================================")

    url = "http://127.0.0.1:8000/ingest"
    payload = json.dumps({"audio_file_path": str(file_path)}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            t1 = time.perf_counter()
            print(f"[SUCCESS] Ingested in {t1 - t0:.2f}s")
            print(f"  audio_file_id: {data['audio_file_id']}")
            print(f"  chunk_count:   {data['chunk_count']}")
            print(f"  duration_ms:   {data['duration_ms']:.2f}")
            print(f"  skipped:       {data['skipped_existing']}")
            return data
    except Exception as exc:
        print(f"[ERROR] Failed to ingest {filename}: {exc}")
        return None


def main():
    print("Ingesting 6 8-minute English podcast episodes...")
    results = []
    for filename, speakers, topic in FILES:
        res = ingest_file(filename, speakers, topic)
        if res:
            results.append((filename, speakers, topic, res))

    print("\n========================================================")
    print(f"INGESTION SUMMARY ({len(results)}/{len(FILES)} processed)")
    print("========================================================")
    for fname, spk, top, res in results:
        print(f"• {fname} | {spk} | Chunks: {res['chunk_count']} | Skipped: {res['skipped_existing']}")


if __name__ == "__main__":
    main()
