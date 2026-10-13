"""Stage 1 + 2 of IndieRAG: evidence dependence graph, clustering, effective number of independent origins (ESS),
and a Bayesian truth-discovery variant with copy detection.

Estimators of P(doc_i and doc_j share an origin):
  none    : all independent (=> ESS == #docs, i.e. plain majority voting)
  H       : training-free heuristic on lexical / paraphrase / shared-rare-detail signals
  L       : gradient-boosted classifier on pairwise features, trained on EchoBench TRAIN questions only
  L+meta  : L with domain / date metadata
  L[abl]  : L trained on the feature-group ablation `abl` of ABLATIONS (e.g. L[-rare], L[lexical_only])
  LLM     : pairwise LLM judge on candidate pairs (others = 0)
  oracle  : ground-truth origins (upper bound)
  lex     : near-duplicate baseline, P = 1[5-gram shingle containment >= 0.5] (the exact quantity MinHash/LSH
            near-duplicate removal approximates)
  cos     : SemDedup-style baseline, P = 1[embedding cosine >= dependence.semdedup_cos]
"""
import argparse
import hashlib
import pickle
import time

import numpy as np

from .signals import FEATURE_GROUPS, DocStore, pair_features
from .utils import DATA, get_logger, load_config, save_json

log = get_logger("dependence")
MODEL_DIR = DATA / "features"


def feat_names(groups):
    return [f for g in groups for f in FEATURE_GROUPS[g]]


DEFAULT_GROUPS = ["lexical", "semantic", "rare", "aigc", "length"]


class DependenceModel:
    def __init__(self, store: DocStore, variant="L", groups=None, llm=None, cfg=None, model_suffix=""):
        self.store, self.variant, self.llm, self.cfg = store, variant, llm, cfg or {}
        self.kind = variant
        if variant.startswith("L["):  # feature-group ablation of the learned estimator
            groups = groups or ABLATIONS[variant[2:-1]]
            self.kind = "L+meta" if "meta" in groups else "L"
        self.groups = groups or (DEFAULT_GROUPS + (["meta"] if variant == "L+meta" else []))
        self.clf = None
        if self.kind not in ("none", "H", "L", "L+meta", "LLM", "oracle", "lex", "cos"):
            raise ValueError(f"unknown dependence estimator {variant}")
        self.time_sec, self.n_calls = 0.0, 0  # dependence-estimation cost (per instance = time_sec / n_calls)
        if self.kind in ("L", "L+meta"):
            p = MODEL_DIR / f"dep_clf_{'_'.join(self.groups)}{model_suffix}.pkl"
            with open(p, "rb") as f:
                self.clf = pickle.load(f)

    def matrix(self, docs, question):
        """docs: list of dicts with 'text' (+ 'domain','date','origin' optional) -> symmetric P [n,n], diag=1"""
        t0 = time.time()
        P = self._matrix(docs, question)
        self.time_sec += time.time() - t0
        self.n_calls += 1
        return P

    def _matrix(self, docs, question):
        n = len(docs)
        P = np.eye(n)
        if n < 2 or self.kind == "none":
            return P
        if self.kind == "oracle":
            for i in range(n):
                for j in range(i + 1, n):
                    P[i, j] = P[j, i] = float(docs[i]["origin"] == docs[j]["origin"])
            return P
        D = [self.store.doc(d["text"], question) for d in docs]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if self.kind in ("lex", "cos"):  # dedup baselines (hard decisions, no learning)
            thr = float((self.cfg.get("dependence") or {}).get("semdedup_cos", 0.95))
            for i, j in pairs:
                if self.kind == "lex":
                    a, b = D[i]["s5"], D[j]["s5"]
                    p = float(bool(a) and bool(b) and len(a & b) / min(len(a), len(b)) >= 0.5)
                else:
                    p = float(float(D[i]["e"] @ D[j]["e"]) >= thr)
                P[i, j] = P[j, i] = p
            return P
        F = [pair_features(self.store, D[i], D[j], docs[i], docs[j]) for i, j in pairs]
        if self.kind == "H":
            for (i, j), f in zip(pairs, F):
                p = max(f["c5"], 1.0 / (1.0 + np.exp(-12 * (f["rare"] - 0.3))) * float(f["cos"] > 0.7))
                P[i, j] = P[j, i] = min(1.0, p)
        elif self.kind in ("L", "L+meta"):
            names = feat_names(self.groups)
            X = np.array([[f[k] for k in names] for f in F])
            prob = self.clf.predict_proba(X)[:, 1]
            for (i, j), p in zip(pairs, prob):
                P[i, j] = P[j, i] = p
        elif self.kind == "LLM":
            from . import prompts as PR
            dc = self.cfg["dependence"]
            cand = [(i, j) for (i, j), f in zip(pairs, F) if f["cos"] >= dc["llm_pair_min_cos"] or f["j3"] >= dc["llm_pair_min_jacc"]]
            if cand:
                # canonical (hash) order of the two documents: the same pair hits the LLM cache in every condition
                hk = [hashlib.sha1(d["text"].encode("utf-8", "ignore")).hexdigest() for d in docs]
                cp = [(i, j) if hk[i] <= hk[j] else (j, i) for i, j in cand]
                outs = self.llm.chat([PR.PAIR_JUDGE.format(a=docs[i]["text"][:2500], b=docs[j]["text"][:2500]) for i, j in cp],
                                     max_tokens=3, tag="pair")
                for (i, j), o in zip(cand, outs):
                    P[i, j] = P[j, i] = 0.95 if "yes" in o.lower() else 0.02
        return P


# ---------------------------------------------------------------------------------------------- aggregation
def clusters(P, thr=0.5):
    n = len(P)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if P[i, j] >= thr:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def ess(idx, P, w=None):
    """effective number of independent sources among docs idx: (sum w)^2 / sum_ij w_i w_j rho_ij"""
    if len(idx) == 0:
        return 0.0
    w = np.ones(len(idx)) if w is None else np.asarray(w, float)
    R = P[np.ix_(idx, idx)]
    return float(w.sum() ** 2 / (w @ R @ w))


def bayes_truth(groups, P, n_iter=5, acc0=0.8, copy_rate=0.9):
    """ACCU-style truth discovery with copy detection (Dong et al. 2009), adapted to soft dependence probabilities.
    groups: list of answer-group id (or -1) per doc. Returns dict group -> posterior probability."""
    docs = [i for i, g in enumerate(groups) if g >= 0]
    vals = sorted({groups[i] for i in docs})
    if not vals:
        return {}
    n_false = max(len(vals), 2)
    acc = {i: acc0 for i in docs}
    post = {v: 1.0 / len(vals) for v in vals}
    for _ in range(n_iter):
        score = {v: 0.0 for v in vals}
        for v in vals:
            S = [i for i in docs if groups[i] == v]
            # order sources by accuracy; a later source's vote is discounted by its dependence on earlier ones
            S.sort(key=lambda i: -acc[i])
            for r, i in enumerate(S):
                tau = np.log(n_false * acc[i] / max(1e-6, 1 - acc[i]))
                indep = np.prod([1 - copy_rate * P[i, j] for j in S[:r]]) if r else 1.0
                score[v] += tau * indep
        m = max(score.values())
        z = sum(np.exp(s - m) for s in score.values())
        post = {v: float(np.exp(score[v] - m) / z) for v in vals}
        for i in docs:
            acc[i] = min(0.99, max(0.01, post[groups[i]]))
    return post


# ---------------------------------------------------------------------------------------------- training IndieRAG-L
def training_pairs(store, groups_list, exclude_gen=None, max_questions=None):
    """pairwise features + labels over the doc pools of EchoBench TRAIN questions.
    exclude_gen: drop copies written by this generator (cross-generator generalisation test).
    max_questions: only the first N train questions (sorted by qid; learning curve)."""
    from .conditions import load_bench
    qs, by_q = load_bench()
    X = {tuple(g): [] for g in groups_list}
    y = []
    rng = np.random.RandomState(0)
    train_ids = sorted(q["qid"] for q in qs if q["split"] == "train")
    keep = set(train_ids[:max_questions] if max_questions else train_ids)
    for q in sorted(qs, key=lambda q: q["qid"]):
        if q["split"] != "train" or q["qid"] not in keep:
            continue
        D = by_q[q["qid"]]
        # sub-sample copies to keep the pair set balanced across copy types
        docs = [d for d in D if d["role"] != "copy"] + [d for d in D if d["role"] == "copy" and rng.rand() < 0.5
                                                         and d.get("gen") != exclude_gen]
        S = [store.doc(d["text"], q["question"]) for d in docs]
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                f = pair_features(store, S[i], S[j], docs[i], docs[j])
                for g in groups_list:
                    X[tuple(g)].append([f[k] for k in feat_names(g)])
                y.append(int(docs[i]["origin"] == docs[j]["origin"]))
    return {k: np.array(v) for k, v in X.items()}, np.array(y)


ABLATIONS = {
    "full": DEFAULT_GROUPS,
    "full+meta": DEFAULT_GROUPS + ["meta"],
    "-lexical": [g for g in DEFAULT_GROUPS if g != "lexical"],
    "-semantic": [g for g in DEFAULT_GROUPS if g != "semantic"],
    "-rare": [g for g in DEFAULT_GROUPS if g != "rare"],
    "-aigc": [g for g in DEFAULT_GROUPS if g != "aigc"],
    "lexical_only": ["lexical"],
}


def train(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from .conditions import load_bench
    store = DocStore()
    groups_list = list(ABLATIONS.values())
    X, y = training_pairs(store, groups_list)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        raise RuntimeError("no usable training pairs (EchoBench train split empty?) - check eb_assemble.log")
    log.info(f"training pairs: {len(y)} (positive rate {y.mean():.3f})")
    info = {}
    for name, g in ABLATIONS.items():
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                             random_state=cfg["seed"])
        clf.fit(X[tuple(g)], y)
        with open(MODEL_DIR / f"dep_clf_{'_'.join(g)}.pkl", "wb") as f:
            pickle.dump(clf, f)
        info[name] = {"groups": g, "n_pairs": int(len(y))}
    # classifier that never saw copies of the second generator (tested on them in dep_eval)
    Xn, yn = training_pairs(store, [DEFAULT_GROUPS], exclude_gen="builder2")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                         random_state=cfg["seed"]).fit(Xn[tuple(DEFAULT_GROUPS)], yn)
    with open(MODEL_DIR / f"dep_clf_{'_'.join(DEFAULT_GROUPS)}_nobuilder2.pkl", "wb") as f:
        pickle.dump(clf, f)
    info["full_without_builder2_copies"] = {"groups": DEFAULT_GROUPS, "n_pairs": int(len(yn))}
    # learning curve: IndieRAG-L trained on the first N train questions only (evaluated in dep_eval)
    for nq in cfg["dependence"].get("learning_curve", []):
        Xq, yq = training_pairs(store, [DEFAULT_GROUPS], max_questions=nq)
        if len(set(yq.tolist())) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                             random_state=cfg["seed"]).fit(Xq[tuple(DEFAULT_GROUPS)], yq)
        with open(MODEL_DIR / f"dep_clf_{'_'.join(DEFAULT_GROUPS)}_n{nq}.pkl", "wb") as f:
            pickle.dump(clf, f)
        info[f"learning_curve_n{nq}"] = {"groups": DEFAULT_GROUPS, "n_train_questions": nq, "n_pairs": int(len(yq))}
    info["n_train_questions_all"] = sum(1 for q in load_bench()[0] if q["split"] == "train")
    save_json(info, MODEL_DIR / "dep_clf_info.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    train(load_config(a.config))


if __name__ == "__main__":
    main()
