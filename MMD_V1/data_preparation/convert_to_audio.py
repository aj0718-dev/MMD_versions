#!/usr/bin/env python3
"""
motif_pe2audio.py

Convert MOTIF PE samples to WAV audio while preserving
family directory structure.

Input:
    /home/aakanksha/MOTIF/family_samples/<family>/<MOTIF_md5>

Output:
    /home/aakanksha/MOTIF/family_audio/<family>/<MOTIF_md5>.wav

Byte-to-audio mapping:
    Direct 8-bit unsigned PCM (lossless, reversible)
"""

import sys
import wave
from pathlib import Path
from datetime import datetime

# ================= CONFIG =================
INPUT_ROOT = Path("/home/aakanksha/MOTIF/family_samples")
OUTPUT_ROOT = Path("/home/aakanksha/MOTIF/family_audio")
SAMPLE_RATE = 8000  #can change, does not affect reversibility
# ==========================================


def write_wav_from_bytes(raw: bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)      # mono
        w.setsampwidth(1)      # 8-bit unsigned PCM
        w.setframerate(SAMPLE_RATE)
        w.writeframes(raw)


def main():
    if not INPUT_ROOT.exists():
        print(f"ERROR: Input path does not exist: {INPUT_ROOT}")
        sys.exit(1)

    total = 0
    converted = 0

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting conversion...")
    print(f"Input : {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}\n")

    for family_dir in sorted(INPUT_ROOT.iterdir()):
        if not family_dir.is_dir():
            continue

        family_name = family_dir.name

        for sample_file in family_dir.iterdir():
            if not sample_file.is_file():
                continue

            total += 1

            # Preserve folder structure
            relative_path = sample_file.relative_to(INPUT_ROOT)
            out_path = OUTPUT_ROOT / relative_path.with_suffix(".wav")

            print(f"[{family_name}] {sample_file.name} -> {out_path.name}")

            try:
                raw_bytes = sample_file.read_bytes()
                write_wav_from_bytes(raw_bytes, out_path)
                converted += 1
            except Exception as e:
                print(f"  ERROR processing {sample_file}: {e}")

    print("\nDone.")
    print(f"Total samples scanned: {total}")
    print(f"Total WAV files created: {converted}")
    print(f"Saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
