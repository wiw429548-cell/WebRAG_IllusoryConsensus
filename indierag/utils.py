import json
import logging
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


# ---------------------------------------------------------------- answer normalisation / QA metrics (SQuAD style)
def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1(pred, golds):
    best = 0.0
    pt = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        pr, rc = ns / len(pt), ns / len(gt)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def contains(pred, golds):
    """accuracy used for free-form LLM answers: a (normalised) gold alias occurs in the (normalised) prediction"""
    p = " " + normalize_answer(pred) + " "
    return float(any(normalize_answer(g) and (" " + normalize_answer(g) + " ") in p for g in golds))


def mentions(text, aliases):
    t = " " + normalize_answer(text) + " "
    return any(normalize_answer(a) and (" " + normalize_answer(a) + " ") in t for a in aliases)


def extract_short_answer(text):
    """takes the content of <answer>..</answer> if present, otherwise the first line."""
    m = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if m:
        return m[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:：]\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip().split("\n")[0]
    return text.strip().split("\n")[0]
