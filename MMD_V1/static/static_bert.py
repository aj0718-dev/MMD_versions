#!/usr/bin/env python3
import os
import re
import math
import pefile
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path("/home/aakanksha/MOTIF/family_samples")
OUT_DIR = Path("./motif_bert_data")
OUT_DIR.mkdir(exist_ok=True)

CORPUS_FILE = OUT_DIR / "corpus.txt"
LABELS_FILE = OUT_DIR / "labels.txt"
ERR_LOG = OUT_DIR / "errors.log"

# Suspicious APIs (from your script)
SUSPICIOUS_APIS = [
    "LoadLibrary", "GetProcAddress", "VirtualAlloc", "VirtualProtect",
    "WriteProcessMemory", "CreateRemoteThread", "OpenProcess", "CreateProcess",
    "InternetOpenUrl", "URLDownloadToFile", "WinExec", "ShellExecute",
    "RegOpenKey", "RegCreateKey", "RegSetValue"
]

# ================= HELPERS =================

def entropy(data):
    if not data:
        return 0.0
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    ent = 0.0
    L = len(data)
    for v in freq.values():
        p = v / L
        ent -= p * math.log2(p)
    return ent


def bin_entropy(e):
    if e < 3:
        return "entropy_low"
    elif e < 6:
        return "entropy_medium"
    else:
        return "entropy_high"


def bin_size(size):
    if size < 50_000:
        return "size_small"
    elif size < 500_000:
        return "size_medium"
    else:
        return "size_large"


def bin_count(x, name):
    if x == 0:
        return f"{name}_0"
    elif x < 5:
        return f"{name}_few"
    elif x < 20:
        return f"{name}_moderate"
    else:
        return f"{name}_many"


# ================= FEATURE EXTRACTION =================

def extract_tokens(exe_path, sha):
    tokens = []

    try:
        pe = pefile.PE(str(exe_path), fast_load=True)
        pe.parse_data_directories()
    except Exception as e:
        with open(ERR_LOG, "a") as f:
            f.write(f"{sha}: PE load error {e}\n")
        return None

    # file bytes
    try:
        with open(exe_path, "rb") as f:
            data = f.read()
    except Exception:
        return None

    size = len(data)
    tokens.append(bin_size(size))
    tokens.append(bin_entropy(entropy(data)))

    # sections
    sections = pe.sections or []
    tokens.append(f"sections_{len(sections)}")

    sec_ents = []
    for s in sections:
        try:
            sec_ents.append(entropy(s.get_data()))
            name = s.Name.rstrip(b"\x00").decode(errors="ignore").lower()
            if name:
                tokens.append(f"sec_{name}")
        except:
            pass

    if sec_ents:
        tokens.append(bin_entropy(sum(sec_ents)/len(sec_ents)))
        tokens.append(bin_entropy(max(sec_ents)))

    # overlay
    try:
        end = max((s.PointerToRawData + s.SizeOfRawData) for s in sections)
        overlay = max(0, size - end)
    except:
        overlay = 0
    tokens.append(bin_count(overlay, "overlay"))

    # imports
    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name:
                    imports.append(imp.name.decode(errors="ignore").lower())

    tokens.append(bin_count(len(imports), "imports"))

    # suspicious APIs
    for api in SUSPICIOUS_APIS:
        if any(api.lower() in imp for imp in imports):
            tokens.append(f"api_{api.lower()}")

    # imphash
    try:
        imph = pe.get_imphash()
        if imph:
            tokens.append(f"imphash_{imph[:8]}")
    except:
        pass

    # resources
    if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        tokens.append("has_resources")

    # strings
    printable = re.findall(rb"[ -~]{4,}", data)
    tokens.append(bin_count(len(printable), "strings"))

    url_re = re.compile(rb"https?://")
    ip_re = re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    urls = sum(1 for s in printable if url_re.search(s))
    ips = sum(1 for s in printable if ip_re.search(s))

    if urls > 0:
        tokens.append("has_url")
    if ips > 0:
        tokens.append("has_ip")

    # signature
    if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY'):
        tokens.append("signed")
    else:
        tokens.append("unsigned")

    # timestamp
    try:
        ts = pe.FILE_HEADER.TimeDateStamp
        if ts == 0:
            tokens.append("timestamp_zero")
        else:
            tokens.append("timestamp_present")
    except:
        pass

    return " ".join(tokens)


# ================= MAIN =================

def main():
    corpus_f = open(CORPUS_FILE, "w")
    label_f = open(LABELS_FILE, "w")

    families = [d for d in BASE_DIR.iterdir() if d.is_dir()]

    total = 0

    for family_dir in families:
        family = family_dir.name

        for file in family_dir.glob("MOTIF_*"):
            if not file.is_file():
                continue

            sha = file.name

            print(f"[{total}] Processing {family}/{sha}")

            tokens = extract_tokens(file, sha)

            if tokens:
                corpus_f.write(tokens + "\n")
                label_f.write(family + "\n")
                total += 1

    corpus_f.close()
    label_f.close()

    print(f"\nDone. Processed {total} samples.")
    print(f"Corpus: {CORPUS_FILE}")
    print(f"Labels: {LABELS_FILE}")


if __name__ == "__main__":
    main()
