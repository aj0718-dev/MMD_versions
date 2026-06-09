import json
import shutil
from pathlib import Path

# ================= PATHS =================

IMG_DIR = Path("/home/aakanksha/MOTIF/MOTIF_defanged")
JSONL_PATH = Path("/home/aakanksha/MOTIF/dataset/motif_dataset.jsonl")
OUT_ROOT = Path("/home/aakanksha/MOTIF/family_samples")

# =========================================


def sanitize_name(name):
    """
    Make folder-safe names.
    """
    return name.replace(" ", "_").replace("/", "_").lower()


def load_md5_to_family(jsonl_path):
    md5_to_family = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            md5_to_family[record["md5"]] = record["reported_family"]

    return md5_to_family


def create_family_folders():
    md5_to_family = load_md5_to_family(JSONL_PATH)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    for img_path in IMG_DIR.glob("MOTIF_*"):
        md5 = img_path.stem.replace("MOTIF_", "")

        if md5 not in md5_to_family:
            skipped += 1
            continue

        family = sanitize_name(md5_to_family[md5])
        family_dir = OUT_ROOT / family
        family_dir.mkdir(parents=True, exist_ok=True)

        dest_path = family_dir / img_path.name

        shutil.copy2(img_path, dest_path)
        count += 1

        if count % 200 == 0:
            print(f"[INFO] {count} images organized")

    print(f"\nDone.")
    print(f"Images moved: {count}")
    print(f"Skipped (no family match): {skipped}")


if __name__ == "__main__":
    create_family_folders()
