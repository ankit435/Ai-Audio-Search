"""Generates the synthetic golden dataset: real .wav audio files plus
ground-truth `*.turns.json` sidecar files recording the exact
speaker/text/start_time/end_time for every turn.

Uses macOS `say` (two distinct voices standing in for two speakers) and
`afconvert`/`ffmpeg` to produce real audio — not silence or noise, real
synthesized speech with real, original (non-copyrighted) dialogue
authored in `golden_conversations.py`.

Why synthetic instead of downloaded interviews/podcasts: real copyrighted
audio can't be redistributed with this project, and the golden dataset
must be committed alongside the code for reproducible evaluation. Every
turn's ground-truth timing is recorded exactly as generated (not
estimated), which is what makes the recall@k and speaker-accuracy
evaluation in `tests/test_recall_at_k_golden.py` trustworthy.

Usage:
    python3 scripts/generate_golden_dataset.py
"""

from __future__ import annotations

import json
import os
import subprocess
import wave

from golden_conversations import CONVERSATIONS, VOICE_BY_SPEAKER

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden")
SILENCE_GAP_SECONDS = 0.6


def _wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _synthesize_turn(text: str, voice: str, out_wav: str) -> float:
    aiff = out_wav.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", voice, text, "-o", aiff], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", "-c", "1", "-r", "16000", aiff, out_wav], check=True)
    os.remove(aiff)
    return _wav_duration_seconds(out_wav)


def _make_silence(out_wav: str, seconds: float, rate: int = 16000) -> None:
    n_frames = int(seconds * rate)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)


def _concat_wavs(wav_paths: list[str], out_wav: str) -> None:
    with wave.open(wav_paths[0], "rb") as first:
        params = first.getparams()
    with wave.open(out_wav, "wb") as out:
        out.setparams(params)
        for p in wav_paths:
            with wave.open(p, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))


def generate_conversation(name: str, turns: list[tuple[str, str]]) -> None:
    work_dir = os.path.join(OUT_DIR, f"_tmp_{name}")
    os.makedirs(work_dir, exist_ok=True)

    segment_paths: list[str] = []
    turn_records: list[dict] = []
    cursor = 0.0
    silence_path = os.path.join(work_dir, "silence.wav")
    _make_silence(silence_path, SILENCE_GAP_SECONDS)

    for i, (speaker, text) in enumerate(turns):
        voice = VOICE_BY_SPEAKER[speaker]
        turn_wav = os.path.join(work_dir, f"turn_{i:03d}.wav")
        duration = _synthesize_turn(text, voice, turn_wav)

        start_time = cursor
        end_time = cursor + duration
        turn_records.append(
            {
                "turn_index": i,
                "speaker_id": speaker,
                "text": text,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
            }
        )
        segment_paths.append(turn_wav)
        cursor = end_time
        if i != len(turns) - 1:
            segment_paths.append(silence_path)
            cursor += SILENCE_GAP_SECONDS

    final_wav = os.path.join(OUT_DIR, f"{name}.wav")
    _concat_wavs(segment_paths, final_wav)

    sidecar = {
        "file_name": f"{name}.wav",
        "duration_seconds": round(cursor, 3),
        "speakers": sorted({t["speaker_id"] for t in turn_records}),
        "turns": turn_records,
    }
    with open(os.path.join(OUT_DIR, f"{name}.turns.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    # cleanup per-turn temp files
    for p in segment_paths:
        if os.path.exists(p) and p != silence_path:
            os.remove(p)
    os.remove(silence_path)
    os.rmdir(work_dir)

    print(f"{name}: {len(turns)} turns, {cursor:.1f}s -> {final_wav}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, turns in CONVERSATIONS.items():
        generate_conversation(name, turns)


if __name__ == "__main__":
    main()
