import re
from urllib.parse import urlparse

WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def words(text):
    return WORD.findall(text.lower())


def shingles(text, n=5):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    """fraction of the smaller set contained in the larger one (robust to boilerplate added around a copy)"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def domain(url):
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


def window_around(text, aliases, n_words=220):
    """~n_words window centred on the first mention of any alias (falls back to the head of the page)."""
    low = text.lower()
    best = None
    for a in aliases:
        a = a.lower().strip()
        if not a:
            continue
        m = re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", low)
        if m and (best is None or m.start() < best):
            best = m.start()
    toks = text.split()
    pos = len(low[:best].split()) if best is not None else 0
    s = max(0, pos - n_words // 2)
    return " ".join(toks[s:s + n_words])


def clean_page(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()
