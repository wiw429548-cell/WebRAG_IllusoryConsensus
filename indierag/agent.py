"""RQ4: search agents under illusory consensus, with a local dense search engine.

policies : ReAct (instruction model, Search-R1 prompt) | Search-R1 (RL-trained checkpoint) | IndieSearch (ours)
IndieSearch = ReAct + <evidence_status> feedback after every search (answer candidates with #pages and #independent
origins, from the dependence model) + a forced `seek_independent` step when the leading answer is backed by fewer
than tau independent origins (retrieval then excludes already-seen near-duplicates).

python -m indierag.agent --policy Qwen/Qwen2.5-7B-Instruct --mode indie --bench echobench
"""
import argparse
from collections import defaultdict

import numpy as np

from . import prompts as PR
from .claims import extract_claims, group_answers
from .conditions import doc_index, make_instances
from .dependence import DependenceModel, ess
from .llm import LLM
from .signals import DocStore
from .utils import DATA, ROOT, contains, em, f1, get_logger, load_config, read_jsonl, save_json, slug, write_jsonl

QUERY_INSTR = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, store, embedder):
        from sentence_transformers import SentenceTransformer
        self.pool = read_jsonl(DATA / "agent" / "pool.jsonl")
        self.store = store
        self.E = np.stack([store.get(p["text"])[0] for p in self.pool])
        self.enc = SentenceTransformer(embedder, device="cuda")

    def encode(self, queries):
        return self.enc.encode([QUERY_INSTR + q for q in queries], normalize_embeddings=True, batch_size=64)

    def search(self, qv, extra, k, exclude=(), diversify_from=()):
        """extra: instance-specific docs (dicts with text). Returns k docs (dicts with 'text')."""
        sc = self.E @ qv
        top = np.argsort(-sc)[:k * 6]
        cand = [(float(sc[i]), {"text": self.pool[i]["text"]}) for i in top]
        for d in extra:
            cand.append((float(self.store.get(d["text"])[0] @ qv), d))
        cand.sort(key=lambda x: -x[0])
        res = []
        div = [self.store.get(t)[0] for t in diversify_from]
        for s, d in cand:
            if d["text"] in exclude:
                continue
            if div and max(float(self.store.get(d["text"])[0] @ v) for v in div) > 0.9:
                continue  # near-duplicate of evidence already seen -> look for an independent source
            res.append(d)
            if len(res) == k:
                break
        return res


def load_bench(cfg, bench):
    n = cfg["agent"]["n_questions"]
    out = []
    if bench == "echobench":
        idx = doc_index()
        insts = [it for it in make_instances(cfg, "test", ("main",)) if it["cond"] in ("A_mixed_m0_k2", "A_mixed_m4_k2")]
        qids = sorted({it["qid"] for it in insts})[:n]
        for it in insts:
            if it["qid"] in qids:
                docs = [idx[d] for d in it["doc_ids"] if idx[d]["role"] != "distractor"]
                out.append({"iid": it["iid"], "question": it["question"], "golds": it["aliases"], "wrong": it["wrong"],
                            "m": it["m"], "extra": docs})
    else:
        qs = read_jsonl(DATA / "hotpot_echo" / "questions.jsonl")[:n]
        docs = defaultdict(list)
        for d in read_jsonl(DATA / "hotpot_echo" / "docs.jsonl"):
            docs[d["qid"]].append(d)
        for q in qs:
            for m in (0, 4):
                echo = docs[q["qid"]][:1 + m]
                out.append({"iid": f"{q['qid']}|m{m}", "question": q["question"], "golds": q["aliases"],
                            "wrong": q["wrong"], "m": m, "extra": echo})
    return out


def evidence_status(st, ctx):
    gid, labels = group_answers(st["claims"])
    if not labels:
        return "<evidence_status>No retrieved document states an explicit answer yet.</evidence_status>", 0.0
    P = ctx["dep"].matrix(st["seen"], st["question"])
    rows = []
    for k, lab in enumerate(labels):
        idx = [i for i, g in enumerate(gid) if g == k]
        rows.append((ess(idx, P), len(idx), lab))
    rows.sort(key=lambda x: -x[0])
    txt = "; ".join(f"\"{lab}\": {n} page(s), ~{e:.1f} independent origin(s)" for e, n, lab in rows)
    return f"<evidence_status>{txt}</evidence_status>", rows[0][0]


def run(cfg, policy, mode, bench, log, tau=None):
    store = DocStore()
    ret = Retriever(store, cfg["models"]["embedder"])
    llm = LLM(policy, max_model_len=cfg["models"]["max_model_len"], gpu_mem=0.75)
    dep = DependenceModel(store, "L", cfg=cfg)
    ac = cfg["agent"]
    tau = cfg["dependence"]["ess_tau"] if tau is None else float(tau)  # tau <= 0: feedback only, never forced
    insts = load_bench(cfg, bench)
    states = []
    for it in insts:
        instr = PR.REACT_AGENT.format(q=it["question"]) + (PR.INDIE_AGENT_NOTE if mode == "indie" else "")
        states.append({**it, "prefix": llm.render([{"role": "user", "content": instr}]), "done": False, "n_search": 0,
                       "seen": [], "claims": [], "forced": False, "answer": "", "seek": False})
    ctx = {"dep": dep}
    for turn in range(ac["max_turns"] + 2):
        act = [s for s in states if not s["done"]]
        if not act:
            break
        last = turn >= ac["max_turns"] + 1
        if last:
            for s in act:
                s["prefix"] += "\n<answer>"
        outs = llm.chat([s["prefix"] for s in act], raw=True, max_tokens=400, temperature=0.0,
                        stop=["</search>", "</answer>"], tag=f"agent{turn}")
        searches = []
        for s, o in zip(act, outs):
            if last or ("<answer>" in o and ("<search>" not in o or o.rfind("<answer>") > o.rfind("<search>"))):
                ans = o.split("<answer>")[-1].strip() if "<answer>" in o else o.strip()
                if mode == "indie" and not last and not s["forced"] and s["n_search"] < ac["max_turns"]:
                    _, lead = evidence_status(s, ctx)
                    if tau > 0 and lead < tau:
                        s["forced"], s["seek"] = True, True
                        s["prefix"] += o.split("<answer>")[0] + (
                            f"\n<evidence_status>The leading answer is backed by only ~{lead:.1f} independent origin(s). "
                            "Search again with a differently phrased query to find an independent source before "
                            "answering.</evidence_status>\n")
                        continue
                s["answer"], s["done"] = ans.split("\n")[0], True
                s["prefix"] += o + "</answer>"
            elif "<search>" in o:
                q = o.split("<search>")[-1].strip()
                s["prefix"] += o + "</search>"
                searches.append((s, q))
            else:  # no tag: treat the text as the answer
                s["answer"], s["done"] = o.strip().split("\n")[0], True
        if searches:
            qv = ret.encode([q for _, q in searches])
            new_pairs = []
            for (s, q), v in zip(searches, qv):
                seen_t = {d["text"] for d in s["seen"]}
                docs = ret.search(v, s["extra"], ac["topk"], exclude=seen_t if s["seek"] else (),
                                  diversify_from=[d["text"] for d in s["seen"]] if s["seek"] else ())
                s["seek"] = False
                s["n_search"] += 1
                s["_new"] = docs
                new_pairs += [(s["question"], d["text"]) for d in docs]
            if mode == "indie":
                cl = dict(zip(new_pairs, extract_claims(llm, new_pairs)))
            for s, _ in searches:
                docs = s.pop("_new")
                info = "\n".join(f"Doc {i + 1}: {d['text'][:1500]}" for i, d in enumerate(docs))
                s["prefix"] += f"\n\n<information>{info}</information>\n\n"
                for d in docs:
                    if d["text"] not in {x["text"] for x in s["seen"]}:
                        s["seen"].append(d)
                        if mode == "indie":
                            s["claims"].append(cl[(s["question"], d["text"])])
                if mode == "indie":
                    s["prefix"] += evidence_status(s, ctx)[0] + "\n\n"
    rows, agg = [], defaultdict(lambda: defaultdict(list))
    for s in states:
        sc = {"acc": contains(s["answer"], s["golds"]), "wrong": contains(s["answer"], [s["wrong"]]),
              "em": em(s["answer"], s["golds"]), "f1": f1(s["answer"], s["golds"]), "n_search": s["n_search"]}
        rows.append({"iid": s["iid"], "m": s["m"], "answer": s["answer"], **sc, "trace": s["prefix"][-6000:]})
        for k, v in sc.items():
            agg[f"m{s['m']}"][k].append(v)
            agg["_all"][k].append(v)
    return rows, {c: {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["acc"])} for c, d in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", default="react", choices=["react", "indie"])
    ap.add_argument("--bench", default="echobench", choices=["echobench", "hotpot"])
    ap.add_argument("--tau", type=float, default=None,
                    help="IndieSearch ablation: ESS threshold of the forced seek step (0 = feedback only); "
                         "default dependence.ess_tau, output dir gets the suffix _tau<value>")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    sfx = "" if a.tau is None else f"_tau{a.tau:g}"
    out = ROOT / "results" / "agent" / f"{a.bench}_{slug(a.policy)}_{a.mode}{sfx}"
    log = get_logger(f"agent/{a.bench}/{a.mode}{sfx}", out / "run.log")
    rows, met = run(cfg, a.policy, a.mode, a.bench, log, tau=a.tau)
    write_jsonl(rows, out / "preds.jsonl")
    save_json({"policy": a.policy, "mode": a.mode, "bench": a.bench, "tau": a.tau, "metrics": met}, out / "metrics.json")
    log.info(met)


if __name__ == "__main__":
    main()
