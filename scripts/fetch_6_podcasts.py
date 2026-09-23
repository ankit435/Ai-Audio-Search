"""Script to download 6 distinct 8-10 minute English podcast audio files, each featuring a unique pair of two speakers.
Trims each audio file locally using ffmpeg to guarantee exact 8-minute length (480 seconds).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = [
    {
        "id": "podcast_ep1_python.wav",
        "speakers": "Lex Fridman & Guido van Rossum",
        "topic": "Python Language Design & History",
        "query": "ytsearch1:Lex Fridman Podcast Guido van Rossum",
    },
    {
        "id": "podcast_ep2_ai.wav",
        "speakers": "Bill Gates & Sam Altman",
        "topic": "AI Development & Future of OpenAI",
        "query": "ytsearch1:Unconfuse Me Bill Gates Sam Altman podcast",
    },
    {
        "id": "podcast_ep3_product.wav",
        "speakers": "Lenny Rachitsky & Shreyas Doshi",
        "topic": "Product Management & Career Growth",
        "query": "ytsearch1:Lenny Podcast Shreyas Doshi product management",
    },
    {
        "id": "podcast_ep4_neuroscience.wav",
        "speakers": "Andrew Huberman & Matthew Walker",
        "topic": "Neuroscience, Brain Performance & Sleep",
        "query": "ytsearch1:Huberman Lab Matthew Walker sleep",
    },
    {
        "id": "podcast_ep5_technews.wav",
        "speakers": "Kevin Roose & Casey Newton",
        "topic": "Tech News, AI Regulation & Industry Shifts",
        "query": "ytsearch1:Hard Fork New York Times Kevin Roose Casey Newton",
    },
    {
        "id": "podcast_ep6_meta.wav",
        "speakers": "Dwarkesh Patel & Mark Zuckerberg",
        "topic": "Llama 3 Open Source AI & Meta Infrastructure",
        "query": "ytsearch1:Dwarkesh Patel Mark Zuckerberg podcast",
    },
]


def download_and_trim_episode(ep: dict) -> Path:
    out_path = DATA_DIR / ep["id"]
    if out_path.exists() and out_path.stat().st_size > 5_000_000:
        print(f"[SKIP] {ep['id']} already exists ({out_path.stat().st_size / 1e6:.2f} MB)")
        return out_path

    print(f"\n========================================================")
    print(f"Downloading & Trimming: {ep['id']}")
    print(f"Speakers: {ep['speakers']}")
    print(f"Topic: {ep['topic']}")
    print(f"========================================================")

    yt_dlp_bin = sys.executable.replace("python", "yt-dlp")
    temp_raw = DATA_DIR / f"temp_raw_{ep['id']}"

    # 1. Download raw audio
    dl_cmd = [
        yt_dlp_bin,
        "-x",
        "--audio-format", "wav",
        ep["query"],
        "-o", str(temp_raw),
    ]
    subprocess.run(dl_cmd, check=True)

    # Find downloaded file
    actual_temp = list(DATA_DIR.glob(f"temp_raw_{ep['id']}*"))[0]

    # 2. Trim to exactly 8 minutes (00:01:00 to 00:09:00) @ 16kHz mono
    ff_cmd = [
        "ffmpeg",
        "-y",
        "-i", str(actual_temp),
        "-ss", "00:01:00",
        "-to", "00:09:00",
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]
    subprocess.run(ff_cmd, check=True)

    # Clean up temp file
    if actual_temp.exists():
        actual_temp.unlink()

    print(f"[DONE] {ep['id']} created ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


def main():
    print(f"Starting fetch & trim for {len(EPISODES)} 8-minute English podcast episodes...")
    for ep in EPISODES:
        try:
            download_and_trim_episode(ep)
        except Exception as exc:
            print(f"[ERROR] Failed {ep['id']}: {exc}")

    print("\nAll downloads and trimming completed!")


if __name__ == "__main__":
    main()
