import numpy as np
from PIL import Image
import pefile
from pathlib import Path

# ================= CONFIG =================

IN_DIR = Path("/home/aakanksha/MOTIF/MOTIF_defanged")
OUT_DIR = Path("/home/aakanksha/MOTIF/rgb_images")

IMG_SIZE = 256
N_BYTES = IMG_SIZE * IMG_SIZE

# =========================================


def pe_to_colored_image(pe_path, out_path):
    raw = pe_path.read_bytes()

    if len(raw) < N_BYTES:
        raw += b"\x00" * (N_BYTES - len(raw))
    raw = raw[:N_BYTES]

    base = np.frombuffer(raw, dtype=np.uint8)

    r = np.zeros(N_BYTES, dtype=np.uint8)
    g = np.zeros(N_BYTES, dtype=np.uint8)
    b = np.zeros(N_BYTES, dtype=np.uint8)

    try:
        pe = pefile.PE(str(pe_path), fast_load=True)
        pe.parse_data_directories()
    except Exception:
        # fallback: grayscale
        img = base.reshape(IMG_SIZE, IMG_SIZE)
        Image.fromarray(img, mode="L").convert("RGB").save(out_path)
        return

    assigned = np.zeros(N_BYTES, dtype=bool)

    # -------- HEADER → RED --------
    header_end = min(pe.OPTIONAL_HEADER.SizeOfHeaders, N_BYTES)
    r[:header_end] = base[:header_end]
    assigned[:header_end] = True

    # -------- TABLES → BLUE --------
    table_dirs = [
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT'],
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE'],
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DEBUG'],
    ]

    for idx in table_dirs:
        entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[idx]
        if entry.VirtualAddress == 0 or entry.Size == 0:
            continue

        start = pe.get_offset_from_rva(entry.VirtualAddress)
        end = min(start + entry.Size, N_BYTES)

        mask = ~assigned[start:end]
        b[start:end][mask] = base[start:end][mask]
        assigned[start:end][mask] = True

    # -------- SECTION DATA → GREEN --------
    for sec in pe.sections:
        start = sec.PointerToRawData
        end = min(start + sec.SizeOfRawData, N_BYTES)

        mask = ~assigned[start:end]
        g[start:end][mask] = base[start:end][mask]
        assigned[start:end][mask] = True

    # -------- ANY UNASSIGNED → GREEN --------
    g[~assigned] = base[~assigned]

    img = np.stack(
        (
            r.reshape(IMG_SIZE, IMG_SIZE),
            g.reshape(IMG_SIZE, IMG_SIZE),
            b.reshape(IMG_SIZE, IMG_SIZE),
        ),
        axis=-1
    )

    Image.fromarray(img).save(out_path)


# ================= MAIN LOOP =================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    for pe_path in sorted(IN_DIR.glob("MOTIF_*")):
        if not pe_path.is_file():
            continue

        out_path = OUT_DIR / f"{pe_path.name}.png"

        try:
            pe_to_colored_image(pe_path, out_path)
            count += 1
        except Exception as e:
            print(f"[SKIP] {pe_path.name}: {e}")
            skipped += 1

        if count % 100 == 0:
            print(f"[INFO] {count} images generated")

    print(f"\n[DONE] {count} images generated, {skipped} skipped")


if __name__ == "__main__":
    main()
