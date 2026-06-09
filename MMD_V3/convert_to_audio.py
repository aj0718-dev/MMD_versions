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
import argparse
from pathlib import Path
from datetime import datetime

def write_wav_from_bytes(raw: bytes, out_path: Path, sample_rate: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)      # mono
        w.setsampwidth(1)      # 8-bit unsigned PCM
        w.setframerate(sample_rate)
        w.writeframes(raw)


def main():
    parser = argparse.ArgumentParser(description="Convert MOTIF PE samples to WAV audio.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing malware families with PE files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted WAV files.")
    parser.add_argument("--sample_rate", type=int, default=8000, help="Sample rate for the WAV file (default: 8000).")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    sample_rate = args.sample_rate

    if not input_root.exists():
        print(f"ERROR: Input path does not exist: {input_root}")
        sys.exit(1)

    total = 0
    converted = 0

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting conversion...")
    print(f"Input : {input_root}")
    print(f"Output: {output_root}\n")

    for family_dir in sorted(input_root.iterdir()):
        if not family_dir.is_dir():
            continue

        family_name = family_dir.name

        for sample_file in family_dir.iterdir():
            if not sample_file.is_file():
                continue

            total += 1

            # Preserve folder structure
            relative_path = sample_file.relative_to(input_root)
            out_path = output_root / relative_path.with_suffix(".wav")

            # Only print periodically to improve terminal performance
            if total % 1000 == 0:
                print(f"Processed {total} files...")

            try:
                raw_bytes = sample_file.read_bytes()
                write_wav_from_bytes(raw_bytes, out_path, sample_rate)
                converted += 1
            except Exception as e:
                print(f"  ERROR processing {sample_file}: {e}")

    print("\nDone.")
    print(f"Total samples scanned: {total}")
    print(f"Total WAV files created: {converted}")
    print(f"Saved under: {output_root}")


if __name__ == "__main__":
    main()
