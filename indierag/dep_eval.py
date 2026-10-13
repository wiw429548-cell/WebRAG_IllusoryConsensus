"""RQ3: how well is dependence (shared origin) detected?  Pairwise AUC / AP, by copy type and by hard-negative type,
feature-group ablations, clustering ARI and ESS error, on the doc pools of EchoBench TEST questions."""
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

from .conditions import doc_index, make_instances
from .dependence import ABLATIONS, DEFAULT_GROUPS, MODEL_DIR, DependenceModel, clusters, ess
from .signals import DocStore
from .utils import DATA, ROOT, get_logger, load_config, load_json, save_json

log = get_logger("dep_eval")


def pair_type(a, b):
    if a["origin"] == b["origin"]:
        t = sorted([a["copy_type"] or "origin", b["copy_type"] or "origin"])
        return "dep:" + "+".join(t)
    roles = tuple(sorted([a["role"], b["role"]]))
    return "indep:" + "+".join(roles)


def metrics_block(y, s, q, n_boot=200, seed=0):
    """AUC / AP / F1@0.5 with 95% CIs from a question-level cluster bootstrap (pairs of a pool are not independent)"""
    out = {"AUC": float("nan"), "AP": float("nan"), "F1@0.5": float("nan")}
    if len(set(y.tolist())) < 2:
        return out

    def f1(yy, ss):
        return float(2 * ((ss >= .5) & (yy == 1)).sum() / max(1, (ss >= .5).sum() + (yy == 1).sum()))
    out = {"AUC": float(roc_auc_score(y, s)), "AP": float(average_precision_score(y, s)), "F1@0.5": f1(y, s)}
    uq = np.unique(q)
    by_q = {u: np.where(q == u)[0] for u in uq}
    rng = np.random.RandomState(seed)
    bs = {k: [] for k in out}
    for _ in range(n_boot):
        ix = np.concatenate([by_q[u] for u in rng.choice(uq, len(uq), replace=True)])
        yy, ss = y[ix], s[ix]
        if len(set(yy.tolist())) < 2:
            continue
        bs["AUC"].append(roc_auc_score(yy, ss))
        bs["AP"].append(average_precision_score(yy, ss))
        bs["F1@0.5"].append(f1(yy, ss))
    for k, v in bs.items():
        if v:
            out[k + "_ci"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--llm", default=None, help="also evaluate the LLM pairwise judge with this model")
    ap.add_argument("--max_questions", type=int, default=600)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = ROOT / "results" / "dependence"
    store = DocStore()
    idx = doc_index()
    insts = make_instances(cfg, "test", ("main", "types", "indep"))
    # one instance per (question, condition) is too redundant -> evaluate on the union pool of each question
    pools = defaultdict(set)
    for it in insts:
        pools[(it["qid"], it["question"])].update(it["doc_ids"])
    keys = sorted(pools)[:a.max_questions]
    variants = {"H": DependenceModel(store, "H", cfg=cfg), "lex": DependenceModel(store, "lex", cfg=cfg),
                "cos": DependenceModel(store, "cos", cfg=cfg)}  # lex / cos: near-dup and SemDedup baselines
    for name, g in ABLATIONS.items():
        variants[f"L[{name}]"] = DependenceModel(store, "L+meta" if "meta" in g else "L", groups=g, cfg=cfg)
    for nq in cfg["dependence"].get("learning_curve", []):  # IndieRAG-L trained on fewer train questions
        sfx = f"_n{nq}"
        if (MODEL_DIR / f"dep_clf_{'_'.join(DEFAULT_GROUPS)}{sfx}.pkl").exists():
            variants[f"L[n_train={nq}]"] = DependenceModel(store, "L", cfg=cfg, model_suffix=sfx)
    n_llm = cfg["dependence"].get("llm_judge_questions", 100)
    llm_keys = set(keys[:n_llm])
    if a.llm:
        from .llm import LLM
        llm = LLM(a.llm, max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
        variants["LLM"] = DependenceModel(store, "LLM", llm=llm, cfg=cfg)
    scores = defaultdict(list)
    labels, ptypes, qpair = [], [], []
    ari = defaultdict(list)
    for qn, (qid, question) in enumerate(keys):
        docs = [idx[d] for d in sorted(pools[(qid, question)])]
        n = len(docs)
        iu = np.triu_indices(n, 1)
        lab = np.array([[float(docs[i]["origin"] == docs[j]["origin"]) for j in range(n)] for i in range(n)])[iu]
        labels.append(lab)
        qpair.append(np.full(len(lab), qn))
        ptypes += [pair_type(docs[i], docs[j]) for i, j in zip(*iu)]
        true_cl = [d["origin"] for d in docs]
        for v, m in variants.items():
            if v == "LLM" and (qid, question) not in llm_keys:  # expensive judge: first n_llm pools only
                scores[v].append(np.full(len(lab), np.nan))
                continue
            P = m.matrix(docs, question)
            scores[v].append(P[iu])
            ari[v].append(adjusted_rand_score(true_cl, clusters(P, cfg["dependence"]["hard_threshold"])))
    y = np.concatenate(labels)
    qpair = np.concatenate(qpair)
    ptypes = np.array(ptypes)
    sub = qpair < n_llm  # the LLM-judge subset (all estimators are also reported on it)
    res = {"n_pairs": int(len(y)), "pos_rate": float(y.mean()), "n_questions": len(keys),
           "n_questions_llm_subset": int(min(n_llm, len(keys))), "variants": {}, "variants_llm_subset": {},
           "time_sec_per_pool": {v: m.time_sec / max(1, m.n_calls) for v, m in variants.items()}}
    p2 = DATA / "echobench" / "phase2_meta.json"
    res["builder2_used"] = bool(load_json(p2).get("builder2_used")) if p2.exists() else None
    for v in variants:
        s = np.concatenate(scores[v])
        ok = ~np.isnan(s)
        r = metrics_block(y[ok], s[ok], qpair[ok])
        r["ARI"] = float(np.mean(ari[v])) if ari[v] else float("nan")
        r["n_pairs"] = int(ok.sum())
        r["by_type"] = {}
        for t in sorted(set(ptypes)):
            m = (ptypes == t) & ok
            if not m.any():
                continue
            # positives: recall at 0.5 ; negatives: false-positive rate at 0.5
            r["by_type"][t] = {"n": int(m.sum()), "mean_p": float(s[m].mean()),
                               ("recall@0.5" if t.startswith("dep") else "fpr@0.5"): float((s[m] >= .5).mean())}
        res["variants"][v] = r
        if (sub & ok).any():
            res["variants_llm_subset"][v] = metrics_block(y[sub & ok], s[sub & ok], qpair[sub & ok])
        log.info(f"{v:20s} AUC={r['AUC']:.4f} AP={r['AP']:.4f} ARI={r['ARI']:.4f}")
    # ESS error on the canonical echo instances (role-based claim groups -> isolates dependence estimation)
    ess_err = defaultdict(list)
    for it in insts:
        if it["setting"] != "A" or not it["cond"].startswith("A_mixed"):
            continue
        docs = [idx[d] for d in it["doc_ids"]]
        wrong_idx = [i for i, d in enumerate(docs) if d["role"] in ("origin", "copy")]
        corr_idx = [i for i, d in enumerate(docs) if d["role"] == "correct"]
        for v in ["H", "L[full]"] + (["LLM"] if "LLM" in variants and (it["qid"], it["question"]) in llm_keys else []):
            P = variants[v].matrix(docs, it["question"])
            ess_err[v].append(abs(ess(wrong_idx, P) - 1.0) + abs(ess(corr_idx, P) - len(corr_idx)))
        ess_err["none (#pages)"].append(abs(len(wrong_idx) - 1.0))
        if len(ess_err["H"]) >= 3000:
            break
    res["ess_abs_error"] = {k: float(np.mean(v)) for k, v in ess_err.items()}
    res["ess_abs_error_n"] = {k: len(v) for k, v in ess_err.items()}

    # ---- generalisation across copy GENERATORS (builder vs. a different LLM vs. rule-based syndication)
    gen_rec = defaultdict(list)
    Lf = variants["L[full]"]
    Lnb = DependenceModel(store, "L", cfg=cfg, model_suffix="_nobuilder2")
    for qid, question in keys:
        docs = [idx[d] for d in sorted(pools[(qid, question)])]
        org = [d for d in docs if d["role"] == "origin"]
        if not org:
            continue
        cps = [d for d in docs if d["role"] == "copy"]
        P = Lf.matrix(org + cps, question)
        Pn = Lnb.matrix(org + cps, question)
        for j, c in enumerate(cps):
            gen_rec[f"trained-on-all | {c['gen']}/{c['copy_type']}"].append(float(P[0, j + 1] >= 0.5))
            gen_rec[f"trained-without-builder2 | {c['gen']}/{c['copy_type']}"].append(float(Pn[0, j + 1] >= 0.5))
    # without second-generator copies (builder2 unavailable) the cross-generator test is not applicable
    res["recall_vs_origin_by_generator"] = {k: {"n": len(v), "recall@0.5": float(np.mean(v))} for k, v in sorted(gen_rec.items())}
    res["cross_generator_applicable"] = any("| builder2/" in k for k in gen_rec)

    # ---- cross-dataset generalisation: classifier trained on TriviaQA-web pools, tested on HotpotQA (Wikipedia) echoes
    hp_q, hp_d = DATA / "hotpot_echo" / "questions.jsonl", DATA / "hotpot_echo" / "docs.jsonl"
    if hp_q.exists() and hp_d.exists():
        from .utils import read_jsonl
        echo = defaultdict(list)
        for d in read_jsonl(hp_d):
            echo[d["qid"]].append(d)
        ys, ps = defaultdict(list), defaultdict(list)
        for q in read_jsonl(hp_q)[:300]:
            docs = echo[q["qid"]] + [{"text": p, "origin": f"para{i}", "role": "paragraph"} for i, p in enumerate(q["paragraphs"])]
            if not all(store.has(d["text"]) for d in docs):
                continue
            n = len(docs)
            iu = np.triu_indices(n, 1)
            lab = np.array([[float(docs[i]["origin"] == docs[j]["origin"]) for j in range(n)] for i in range(n)])[iu]
            for v in ["H", "L[full]"]:
                ys[v].append(lab)
                ps[v].append(variants[v].matrix(docs, q["question"])[iu])
        res["cross_dataset_hotpotqa"] = {
            v: {"n_pairs": int(sum(len(x) for x in ys[v])),
                "AUC": float(roc_auc_score(np.concatenate(ys[v]), np.concatenate(ps[v]))),
                "AP": float(average_precision_score(np.concatenate(ys[v]), np.concatenate(ps[v])))}
            for v in ys if len(set(np.concatenate(ys[v]).tolist())) > 1}

    # ---- retriever source bias: where does a dense retriever rank LLM-written copies vs. the real correct pages?
    rb = defaultdict(list)
    for qid, question in keys:
        qv = store.query_emb(question)
        if qv is None:
            continue
        docs = [idx[d] for d in sorted(pools[(qid, question)]) if idx[d]["role"] in ("correct", "origin", "copy")]
        sc = np.array([float(store.get(d["text"])[0] @ qv) for d in docs])
        order = np.argsort(-sc)
        rank = np.empty(len(docs), int)
        rank[order] = np.arange(len(docs))
        for d, r_ in zip(docs, rank):
            rb[d["copy_type"] or d["role"]].append(r_ / max(1, len(docs) - 1))
        top3 = [docs[i]["role"] for i in order[:3]]
        rb["_top3_has_llm_copy"].append(float(any(r in ("copy", "origin") for r in top3)))
    res["retriever_source_bias"] = {k: {"n": len(v), "mean_normalised_rank (0=top)": float(np.mean(v))} for k, v in rb.items()}
    save_json(res, out / "metrics.json")


if __name__ == "__main__":
    main()
