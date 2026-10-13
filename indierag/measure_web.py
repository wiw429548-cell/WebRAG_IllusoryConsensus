"""RQ5: measurement on REAL web search results (TriviaQA rc.web, Bing top results).

phase prepare : sample questions (disjoint from EchoBench) -> data/web/pages.jsonl
phase run     : per-page claims + dependence -> how often is the apparent (page-count) majority different from the
                independent-origin majority? how inflated is page-level support? does IndieRAG hurt on natural data?
"""
import argparse
import glob
import random

import numpy as np

from .claims import extract_claims, group_answers
from .dependence import DependenceModel, clusters, ess
from .llm import LLM
from .methods import Ctx, run_method
from .signals import DocStore
from .textutil import clean_page, containment, domain, shingles, window_around
from .utils import DATA, ROOT, contains, get_logger, load_config, mentions, read_jsonl, save_json, write_jsonl

log = get_logger("web")


def prepare(cfg):
    import pandas as pd
    wc = cfg["web_measure"]
    used = {q["qid"] for q in read_jsonl(DATA / "echobench" / "selected.jsonl")}
    rows = []
    for f in sorted(glob.glob(str(DATA / "raw" / "triviaqa" / "rc.web" / "validation-*.parquet"))):
        df = pd.read_parquet(f, columns=["question", "question_id", "answer", "search_results"])
        for _, r in df.iterrows():
            if r["question_id"] in used:
                continue
            ans = r["answer"]
            aliases = [a for a in list(ans["aliases"]) + [ans["value"]] if a]
            sr = r["search_results"]
            pages = [{"url": u, "domain": domain(u), "title": t, "text": clean_page(c)}
                     for u, t, c in zip(sr["url"], sr["title"], sr["search_context"]) if c and len(c.split()) > 50]
            if len(pages) >= 3:
                rows.append({"qid": r["question_id"], "question": r["question"], "aliases": aliases, "pages": pages})
    random.Random(cfg["seed"]).shuffle(rows)
    rows = rows[:wc["n_questions"]]
    out = []
    for r in rows:
        for j, p in enumerate(r["pages"][:wc["max_pages"]]):
            txt = window_around(p["text"], r["aliases"], 300) if mentions(p["text"], r["aliases"]) else " ".join(p["text"].split()[:300])
            out.append({"qid": r["qid"], "question": r["question"], "aliases": r["aliases"], "doc_id": f"{r['qid']}#{j}",
                        "domain": p["domain"], "url": p["url"], "text": txt[:wc["page_chars"]]})
    write_jsonl(out, DATA / "web" / "pages.jsonl")
    log.info(f"web measurement: {len(rows)} questions, {len(out)} pages")


def boot_ci(vals, groups=None, n_boot=1000, seed=0):
    """95% CI of the mean; resamples whole questions (groups) when given (pages of a question are not independent)"""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    groups = np.arange(len(vals)) if groups is None else np.asarray(groups)
    uq = np.unique(groups)
    idx = {u: np.where(groups == u)[0] for u in uq}
    rng = np.random.RandomState(seed)
    bs = [vals[np.concatenate([idx[u] for u in rng.choice(uq, len(uq))])].mean() for _ in range(n_boot)]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def run(cfg, reader, out):
    pages = read_jsonl(DATA / "web" / "pages.jsonl")
    by_q = {}
    for p in pages:
        by_q.setdefault(p["qid"], {"question": p["question"], "aliases": p["aliases"], "docs": []})["docs"].append(p)
    store = DocStore()
    llm = LLM(reader, max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    pairs = sorted({(q["question"], d["text"]) for q in by_q.values() for d in q["docs"]})
    u0 = llm.usage_snapshot()
    claims = dict(zip(pairs, extract_claims(llm, pairs)))
    u1 = llm.usage_snapshot()
    claim_cost = {k: u1[k] - u0[k] for k in u1}
    dep = DependenceModel(store, "L", cfg=cfg)
    wc = cfg["web_measure"]
    conflict_q = set()
    thr = cfg["dependence"]["hard_threshold"]
    st = {"n_questions": len(by_q), "n_pages": len(pages)}
    dup_lex, dup_dep, inflation, disagree, gold_lead_page, gold_lead_ess, n_cmp = [], [], [], [], [], [], 0
    page_q, cluster_sizes = [], []
    insts = []
    for qn, (qid, q) in enumerate(by_q.items()):
        docs = q["docs"]
        P = dep.matrix(docs, q["question"])
        cl = clusters(P, thr)
        sizes = np.bincount(cl)
        dup_dep += [float(sizes[c] > 1) for c in cl]
        page_q += [qn] * len(docs)
        cluster_sizes += sizes.tolist()
        S = [shingles(d["text"]) for d in docs]
        dup_lex += [float(any(containment(S[i], S[j]) >= 0.5 for j in range(len(docs)) if j != i)) for i in range(len(docs))]
        ans = [claims[(q["question"], d["text"])] for d in docs]
        gid, labels = group_answers(ans)
        if labels:
            cnt = np.bincount([g for g in gid if g >= 0], minlength=len(labels))
            E = np.array([ess([i for i, g in enumerate(gid) if g == k], P) for k in range(len(labels))])
            gold_g = [k for k, l in enumerate(labels) if contains(l, q["aliases"])]
            if gold_g:
                g = gold_g[0]
                inflation.append(cnt[g] / max(E[g], 1e-6))
            if len(labels) >= 2:
                n_cmp += 1
                conflict_q.add(qn)
                disagree.append(float(int(np.argmax(cnt)) != int(np.argmax(E))))
                gold_lead_page.append(float(contains(labels[int(np.argmax(cnt))], q["aliases"])))
                gold_lead_ess.append(float(contains(labels[int(np.argmax(E))], q["aliases"])))
        insts.append({"iid": qid, "question": q["question"], "docs": docs, "golds": q["aliases"], "wrongs": [],
                      "setting": "W", "m": 0, "k": 0, "cond": "web"})
    st.update(frac_pages_lexical_duplicate=float(np.mean(dup_lex)), frac_pages_dependent_cluster=float(np.mean(dup_dep)),
              mean_support_inflation_gold=float(np.mean(inflation)) if inflation else None,
              n_questions_with_conflict=n_cmp,
              page_vs_independent_majority_disagree=float(np.mean(disagree)) if disagree else None,
              acc_page_majority=float(np.mean(gold_lead_page)) if gold_lead_page else None,
              acc_independent_majority=float(np.mean(gold_lead_ess)) if gold_lead_ess else None,
              mean_pages_per_question=float(len(pages) / max(1, len(by_q))),
              mean_dependence_cluster_size=float(np.mean(cluster_sizes)) if cluster_sizes else None)
    st["ci95"] = {"frac_pages_lexical_duplicate": boot_ci(dup_lex, page_q),
                  "frac_pages_dependent_cluster": boot_ci(dup_dep, page_q),
                  "mean_support_inflation_gold": boot_ci(inflation),
                  "page_vs_independent_majority_disagree": boot_ci(disagree),
                  "acc_page_majority": boot_ci(gold_lead_page), "acc_independent_majority": boot_ci(gold_lead_ess)}
    st["source"] = "TriviaQA rc.web validation: cached Bing top results (offline snapshot, no live web access)"
    # natural-data utility check (no injected misinformation): IndieRAG should not hurt; every baseline that runs with
    # the same reader is compared on the same questions (paired bootstrap), overall and on the questions whose pages
    # disagree (>= 2 answer groups), where aggregation actually matters
    methods = list(wc.get("methods") or WEB_METHODS)
    variants = {m.split(":")[1] for m in methods if ":" in m}
    ctx = Ctx(llm, store, {"L": dep, **{v: DependenceModel(store, v, cfg=cfg) for v in variants - {"L"}},
                           "none": DependenceModel(store, "none")}, claims, cfg)
    cache, accs = {}, {}
    conflict = np.array([q in conflict_q for q in range(len(insts))])
    st["n_conflict_questions"] = int(conflict.sum())
    st["cost_per_question"], st["paired"] = {}, {}
    for m in ["vanilla"] + [x for x in methods if x != "vanilla"]:
        u0 = llm.usage_snapshot()
        res = run_method(m, insts, ctx, cache)
        u1 = llm.usage_snapshot()
        if m == "vanilla":
            cache["vanilla_by_iid"] = {it["iid"]: r for it, r in zip(insts, res)}
        acc = [contains(p, it["golds"]) for it, (_, p) in zip(insts, res)]
        accs[m] = np.array(acc, float)
        st[f"acc_{m}"] = float(np.mean(acc))
        st["ci95"][f"acc_{m}"] = boot_ci(acc)
        if conflict.any():
            st[f"accconf_{m}"] = float(accs[m][conflict].mean())
            st["ci95"][f"accconf_{m}"] = boot_ci(accs[m][conflict])
        st["cost_per_question"][m] = {k: (u1[k] - u0[k]) / max(1, len(insts)) for k in u1}
        write_jsonl([{"qid": it["iid"], "pred": p, "acc": a, "conflict": bool(c), "raw": r[:1000]}
                     for it, (r, p), a, c in zip(insts, res, acc, conflict)], out / "preds" / f"{m.replace(':', '_')}.jsonl")
        log.info(f"web {m:18s} acc={st[f'acc_{m}']:.4f}")
    st["claim_extraction_cost_per_question"] = {k: v / max(1, len(insts)) for k, v in claim_cost.items()}
    ours = "indierag:L" if "indierag:L" in accs else None
    for m in accs:
        if m == "vanilla":
            continue
        st["paired"][f"{m}|vanilla"] = paired_boot(accs[m], accs["vanilla"], conflict)
        if ours and m != ours:
            st["paired"][f"{ours}|{m}"] = paired_boot(accs[ours], accs[m], conflict)
    return st


WEB_METHODS = ["vanilla", "dedup", "semdedup", "majority", "indievote:lex", "indievote:cos", "trustrag", "astute",
               "instructrag", "cag", "indiebayes:none", "indievote:L", "indiebayes:L", "indierag:L"]


def paired_boot(a, b, conflict=None, n_boot=5000, seed=0):
    """paired bootstrap over questions (one instance per question): mean diff, 95% CI, two-sided p; also on the
    conflict subset"""
    def one(x, y):
        d = np.asarray(x, float) - np.asarray(y, float)
        if len(d) == 0:
            return None
        rng = np.random.RandomState(seed)
        bs = d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(1)
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        return {"delta": float(d.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)),
                "p": float(min(1.0, max(p, 1.0 / n_boot))), "n": int(len(d))}
    r = {"all": one(a, b)}
    if conflict is not None and np.any(conflict):
        r["conflict"] = one(np.asarray(a)[conflict], np.asarray(b)[conflict])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prepare", "run"])
    ap.add_argument("--reader", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="", help="non-empty: write to results/web_measure/<tag>/ (additional readers)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.phase == "prepare":
        prepare(cfg)
    else:
        out = ROOT / "results" / "web_measure" / a.tag if a.tag else ROOT / "results" / "web_measure"
        st = run(cfg, a.reader, out)
        st["reader"] = a.reader
        save_json(st, out / "metrics.json")


if __name__ == "__main__":
    main()
