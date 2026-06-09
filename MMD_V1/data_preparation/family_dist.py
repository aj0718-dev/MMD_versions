import json
from pathlib import Path
from collections import Counter

# ================= PATHS =================

IMG_DIR = Path("/home/aakanksha/MOTIF/rgb_images")
JSONL_PATH = Path("/home/aakanksha/MOTIF/dataset/motif_dataset.jsonl")
OUT_FILE = Path("/home/aakanksha/MOTIF/family_distribution.txt")

# =========================================


def load_md5_to_family(jsonl_path):
    md5_to_family = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            md5_to_family[record["md5"]] = record["reported_family"]

    return md5_to_family


def compute_distribution():
    md5_to_family = load_md5_to_family(JSONL_PATH)

    family_counts = Counter()
    missing_md5 = []

    image_files = list(IMG_DIR.glob("MOTIF_*.png"))

    for img_path in image_files:
        md5 = img_path.stem.replace("MOTIF_", "")

        if md5 in md5_to_family:
            family_counts[md5_to_family[md5]] += 1
        else:
            missing_md5.append(md5)

    total_images = len(image_files)
    matched = sum(family_counts.values())

    # Write to file
    with open(OUT_FILE, "w") as f:
        f.write("===== FAMILY DISTRIBUTION =====\n\n")
        f.write(f"Total images found: {total_images}\n")
        f.write(f"Matched to dataset: {matched}\n")
        f.write(f"Missing MD5 matches: {len(missing_md5)}\n\n")

        f.write("Family Counts:\n\n")

        for family, count in sorted(family_counts.items()):
            percent = (count / matched) * 100 if matched > 0 else 0
            f.write(f"{family:30s} {count:5d} ({percent:6.2f}%)\n")

        if missing_md5:
            f.write("\n===== UNMATCHED MD5s =====\n")
            for md5 in missing_md5:
                f.write(md5 + "\n")

    print(f"\nDone. Distribution saved to:\n{OUT_FILE}")


if __name__ == "__main__":
    compute_distribution()
