"""Human validation (the only step that needs people). Two annotation sheets are exported automatically after the
experiments; fill the `label` column and run `python -m indierag.annotate score`.

sheet 1  web_pairs.csv     : page pairs from REAL search results (TriviaQA rc.web): same origin? (1/0)
                             -> precision / recall of the dependence detector on real web data, Cohen's kappa
sheet 2  echo_copies.csv   : EchoBench copies vs. their origin: is the copy a faithful derivative that still states the
                             wrong answer? (1/0)  -> validity of the synthetic benchmark
"""
import argparse
import csv
import random

import numpy as np

from .conditions import load_bench
from .dependence import DependenceModel
from .signals import DocStore
from .utils import DATA, ROOT, load_config, read_jsonl, save_json

OUT = ROOT / "results" / "annotation"


def export(cfg, n_pairs=300, n_copies=200):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    store = DocStore()
    dep = DependenceModel(store, "L", cfg=cfg)
    pages = read_jsonl(DATA / "web" / "pages.jsonl")
    by_q = {}
    for p in pages:
        by_q.setdefault(p["qid"], []).append(p)
    rows = []
    qids = list(by_q)
    rng.shuffle(qids)
    for qid in qids:
        docs = by_q[qid]
        if len(docs) < 2:
            continue
        P = dep.matrix(docs, docs[0]["question"])
        pairs = [(i, j) for i in range(len(docs)) for j in range(i + 1, len(docs))]
        # oversample predicted-dependent pairs so that positives are well represented
        pos = [x for x in pairs if P[x] >= 0.5]
        pick = list(dict.fromkeys(pos[:2] + rng.sample(pairs, min(2, len(pairs)))))[:3]  # no duplicate pairs
        for i, j in pick:
            rows.append({"id": f"{qid}|{i}|{j}", "question": docs[0]["question"], "url_a": docs[i]["url"],
                         "url_b": docs[j]["url"], "text_a": docs[i]["text"][:1500], "text_b": docs[j]["text"][:1500],
                         "pred_p": round(float(P[i, j]), 3), "label": ""})
        if len(rows) >= n_pairs:
            break
    _write(OUT / "web_pairs.csv", rows)
    qs, docs = load_bench()
    rows = []
    for q in qs:
        D = docs[q["qid"]]
        org = [d for d in D if d["role"] == "origin"]
        for c in [d for d in D if d["role"] == "copy"]:
            rows.append({"id": c["doc_id"], "question": q["question"], "wrong_answer": q["wrong"],
                         "copy_type": c["copy_type"], "generator": c["gen"], "origin": org[0]["text"][:1500] if org else "",
                         "copy": c["text"][:1500], "label": ""})
    rng.shuffle(rows)
    _write(OUT / "echo_copies.csv", rows[:n_copies])
    print("wrote", OUT)


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("label", "").strip() in ("0", "1")]


def score():
    from sklearn.metrics import cohen_kappa_score
    res = {}
    p = OUT / "web_pairs.csv"
    if p.exists():
        r = _read(p)
        if r:
            y = np.array([int(x["label"]) for x in r])
            pr = np.array([float(x["pred_p"]) >= 0.5 for x in r]).astype(int)
            tp = int(((pr == 1) & (y == 1)).sum())
            res["web_pairs"] = {"n": len(r), "human_positive_rate": float(y.mean()),
                                "precision": tp / max(1, pr.sum()), "recall": tp / max(1, y.sum()),
                                "kappa_detector_vs_human": float(cohen_kappa_score(y, pr))}
    p = OUT / "echo_copies.csv"
    if p.exists():
        r = _read(p)
        if r:
            res["echo_copies"] = {"n": len(r), "valid_rate": float(np.mean([int(x["label"]) for x in r])),
                                  "by_type": {t: float(np.mean([int(x["label"]) for x in r if x["copy_type"] == t]))
                                              for t in sorted({x["copy_type"] for x in r})}}
    save_json(res, OUT / "human_validation.json")
    print(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "score"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    export(load_config(a.config)) if a.action == "export" else score()


if __name__ == "__main__":
    main()
