"""Decompress the shipped *.jsonl.gz files next to themselves (data/ and results/).

The pipeline reads plain .jsonl files; they are stored compressed only to keep the repository small.
Run once after cloning:  python scripts/unpack.py
"""
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

n = 0
for gz in sorted(list((ROOT / "data").rglob("*.jsonl.gz")) + list((ROOT / "results").rglob("*.jsonl.gz"))):
    out = gz.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        continue
    with gzip.open(gz, "rb") as f, open(out, "wb") as g:
        shutil.copyfileobj(f, g)
    n += 1
print(f"unpacked {n} files")
