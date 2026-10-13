"""Document-level signals + pairwise dependence features.

Doc-level (precomputed once by `featurize`, keyed by sha1(text)):
  emb   : bge embedding (L2-normalised)
  aigc  : Fast-DetectGPT analytic curvature (higher = more machine-like)
Pairwise features (cheap, computed on the fly):
  lexical  : 5-gram Jaccard / containment, 3-gram Jaccard
  semantic : cosine, paraphrase signal (cos - 3gram-Jaccard)
  rare     : IDF-weighted overlap of *specific details* (numbers, dates, capitalised names, rare words) shared by
             both documents, excluding terms of the question  -> Dong et al.'s copy-detection principle
  aigc     : min / max of the two AIGC scores
  meta     : same domain, |date gap|   (only used in the '+meta' variant)
"""
import argparse
import hashlib
import math
import re
from collections import Counter

import numpy as np

from .textutil import containment, jaccard, shingles, words
from .utils import DATA, get_logger, load_config, read_jsonl, save_json, load_json

log = get_logger("signals")
FEAT = DATA / "features"
NUM = re.compile(r"\b\d[\d,.:/]*\d\b|\b\d\b")
CAP = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+(?:of|de|la|von|van|the|and)?\s*[A-Z][a-zA-Z'’\-]+)*")

FEATURE_GROUPS = {
    "lexical": ["j5", "c5", "j3"],
    "semantic": ["cos", "para"],
    "rare": ["rare", "rare_max", "n_shared"],
    "aigc": ["aigc_min", "aigc_max"],
    "length": ["len_ratio"],
    "meta": ["same_domain", "date_gap"],
}


def h(text):
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def details(text, idf, q_terms, min_idf=4.0):
    out = set(m.replace(",", "") for m in NUM.findall(text))
    out |= {c.lower() for c in CAP.findall(text) if len(c) > 2}
    out |= {w for w in words(text) if idf.get(w, 0.0) >= min_idf and len(w) > 3}
    return {d for d in out if d not in q_terms}


class DocStore:
    """lookup of precomputed doc-level signals (by text hash)."""

    def __init__(self):
        meta = load_json(FEAT / "docstore.json")
        self.index = meta["index"]
        self.idf = meta["idf"]
        self.default_idf = meta["default_idf"]
        self.emb = np.load(FEAT / "emb.npy")
        self.aigc = np.load(FEAT / "aigc.npy")
        self._cache = {}

    def has(self, text):
        return h(text) in self.index

    def get(self, text):
        k = h(text)
        i = self.index[k]
        return self.emb[i], float(self.aigc[i])

    def doc(self, text, question=""):
        """cached per-(doc, question) structure for pair features"""
        key = (h(text), question)
        if key not in self._cache:
            e, a = self.get(text)
            qt = set(words(question)) | {c.lower() for c in CAP.findall(question)}
            # shingles are stored as (process-local) string hashes: same Jaccard/containment, a fraction of the RAM
            self._cache[key] = {"e": e, "a": a, "s5": {hash(x) for x in shingles(text, 5)},
                                "s3": {hash(x) for x in shingles(text, 3)},
                                "det": details(text, self.idf, qt), "n": len(text.split())}
        return self._cache[key]

    def query_emb(self, question):
        if not hasattr(self, "_qe"):
            qi = FEAT / "q_index.json"
            self._qi = load_json(qi) if qi.exists() else {}
            self._qe = np.load(FEAT / "q_emb.npy") if qi.exists() else None
        i = self._qi.get(question)
        return None if i is None else self._qe[i]

    def idf_of(self, x):
        return self.idf.get(x, self.default_idf)


def pair_features(store, da, db, meta_a=None, meta_b=None):
    shared = da["det"] & db["det"]
    wa = sum(store.idf_of(x) for x in da["det"]) or 1.0
    wb = sum(store.idf_of(x) for x in db["det"]) or 1.0
    ws = sum(store.idf_of(x) for x in shared)
    j3 = jaccard(da["s3"], db["s3"])
    cos = float(da["e"] @ db["e"])
    f = {"j5": jaccard(da["s5"], db["s5"]), "c5": containment(da["s5"], db["s5"]), "j3": j3, "cos": cos,
         "para": cos - j3, "rare": ws / min(wa, wb), "rare_max": max([store.idf_of(x) for x in shared], default=0.0),
         "n_shared": math.log1p(len(shared)), "aigc_min": min(da["a"], db["a"]), "aigc_max": max(da["a"], db["a"]),
         "len_ratio": min(da["n"], db["n"]) / max(da["n"], db["n"], 1)}
    if meta_a is not None and meta_b is not None:
        f["same_domain"] = float(meta_a.get("domain") == meta_b.get("domain"))
        f["date_gap"] = math.log1p(abs(meta_a.get("date", 0) - meta_b.get("date", 0)))
    else:
        f["same_domain"], f["date_gap"] = 0.0, 0.0
    return f


# ---------------------------------------------------------------------------------------------- featurize job
def fast_detectgpt_scores(texts, model_name, max_tokens=256, bs=4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dev == "cuda" else torch.float32)).to(dev).eval()
    out = np.zeros(len(texts), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1].float()
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        lp = torch.log_softmax(logits, -1)
        ll = lp.gather(-1, labels[..., None]).squeeze(-1)
        p = lp.exp()
        mu = (p * lp).sum(-1)
        var = (p * lp.pow(2)).sum(-1) - mu.pow(2)
        num = ((ll - mu) * mask).sum(-1)
        den = ((var * mask).sum(-1)).clamp(min=1e-6).sqrt()
        out[idx] = (num / den).cpu().numpy()
    return out


def collect_texts():
    texts = []
    for p in [DATA / "echobench" / "docs.jsonl", DATA / "web" / "pages.jsonl", DATA / "hotpot_echo" / "docs.jsonl"]:
        if p.exists():
            texts += [d["text"] for d in read_jsonl(p)]
    p = DATA / "raw" / "ramdocs" / "RAMDocs_test.jsonl"
    if p.exists():
        texts += [d["text"] for r in read_jsonl(p) for d in r["documents"]]
    p = DATA / "agent" / "pool.jsonl"
    if p.exists():
        texts += [d["text"] for d in read_jsonl(p)]
    return texts


def featurize(cfg):
    from sentence_transformers import SentenceTransformer
    FEAT.mkdir(parents=True, exist_ok=True)
    texts = collect_texts()
    uniq, index = [], {}
    for t in texts:
        k = h(t)
        if k not in index:
            index[k] = len(uniq)
            uniq.append(t)
    log.info(f"featurizing {len(uniq)} unique documents")
    df = Counter()
    for t in uniq:
        df.update(set(words(t)) | {m.replace(",", "") for m in NUM.findall(t)} | {c.lower() for c in CAP.findall(t)})
    N = len(uniq)
    idf = {w: math.log(N / c) for w, c in df.items() if c <= N * 0.5}
    st = SentenceTransformer(cfg["models"]["embedder"], device="cuda")
    emb = st.encode(uniq, batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    # question embeddings (bge query instruction) -> retriever order of documents ('ranked' group, source-bias analysis)
    qtexts = sorted({q["question"] for p in [DATA / "echobench" / "questions.jsonl", DATA / "hotpot_echo" / "questions.jsonl"]
                     if p.exists() for q in read_jsonl(p)})
    if qtexts:
        qe = st.encode(["Represent this sentence for searching relevant passages: " + q for q in qtexts],
                       batch_size=128, normalize_embeddings=True).astype(np.float32)
        np.save(FEAT / "q_emb.npy", qe)
        save_json({q: i for i, q in enumerate(qtexts)}, FEAT / "q_index.json")
    del st
    aigc = fast_detectgpt_scores(uniq, cfg["models"]["aigc_scorer"])
    np.save(FEAT / "emb.npy", emb)
    np.save(FEAT / "aigc.npy", aigc)
    save_json({"index": index, "idf": idf, "default_idf": math.log(N), "n_docs": N}, FEAT / "docstore.json")
    log.info("featurize done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    featurize(load_config(a.config))


if __name__ == "__main__":
    main()
