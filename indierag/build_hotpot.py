"""Multi-hop echo setting (HotpotQA distractor dev) + the shared local search index for the agent experiments.

phase build : wrong answer + wrong ORIGIN rewrite of the answer-bearing gold paragraph + 4 copies of it
phase pool  : data/agent/pool.jsonl = EchoBench distractor pages + HotpotQA paragraphs (local search engine corpus)
"""
import argparse
import random

from . import prompts as P
from .build_echobench import FAKE_SITES, _valid_wrong
from .utils import DATA, get_logger, load_config, mentions, normalize_answer, read_jsonl, write_jsonl

log = get_logger("hotpot")
OUT = DATA / "hotpot_echo"


def _paragraphs(row):
    ctx = row["context"]
    return [(t, " ".join(s).strip()) for t, s in zip(ctx["title"], ctx["sentences"])]


def build(cfg):
    import pandas as pd
    from .llm import LLM
    rng = random.Random(cfg["seed"])
    df = pd.read_parquet(DATA / "raw" / "hotpotqa" / "distractor" / "validation-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        ans = r["answer"]
        if normalize_answer(ans) in ("yes", "no") or not normalize_answer(ans):
            continue
        sup = set(r["supporting_facts"]["title"])
        paras = _paragraphs(r)
        gold = [(t, p) for t, p in paras if t in sup and mentions(p, [ans])]
        if gold:
            rows.append({"qid": r["id"], "question": r["question"], "answer": ans, "type": r["type"],
                         "gold_title": gold[0][0], "gold_text": f"{gold[0][0]}: {gold[0][1]}",
                         "paragraphs": [f"{t}: {p}" for t, p in paras]})
    rng.shuffle(rows)
    rows = rows[:int(cfg["agent"]["n_questions"] * 1.5)]
    llm = LLM(cfg["models"]["builder"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    outs = llm.generate([P.WRONG_ANSWER.format(q=r["question"], a=r["answer"]) for r in rows], temperature=0.8,
                        max_tokens=24, n=4, tag="hp_wrong")
    keep = []
    for r, o in zip(rows, outs):
        c = [x.strip().strip('."').split("\n")[0] for x in o]
        c = [x for x in c if x and not mentions(x, [r["answer"]]) and normalize_answer(r["answer"]) not in normalize_answer(x)]
        if c:
            r["wrong"] = c[0]
            keep.append(r)
    rows = keep
    outs = llm.generate([P.REWRITE_WRONG.format(q=r["question"], wrong=r["wrong"], gold=r["answer"], passage=r["gold_text"])
                         for r in rows], temperature=0.3, max_tokens=400, n=3, tag="hp_origin")
    keep = []
    for r, o in zip(rows, outs):
        ok = [x.strip() for x in o if _valid_wrong(x, r["wrong"], [r["answer"]])]
        if ok:
            r["origin_text"] = ok[0]
            keep.append(r)
    rows = keep[:cfg["agent"]["n_questions"]]
    src = [r["origin_text"] for r in rows]
    para = llm.generate([P.PARAPHRASE.format(text=s) for s in src], temperature=0.9, max_tokens=400, n=2, tag="hp_para")
    farm = llm.generate([P.AI_FARM.format(persona=P.AI_FARM_PERSONAS[0], text=s) for s in src], temperature=0.9,
                        max_tokens=400, tag="hp_farm")
    summ = llm.generate([P.SUMMARY.format(text=s) for s in src], temperature=0.9, max_tokens=200, tag="hp_summ")
    docs, qs = [], []
    for i, r in enumerate(rows):
        cps = [("paraphrase", para[i][0]), ("ai_farm", farm[i][0]), ("summary", summ[i][0]), ("paraphrase", para[i][1]),
               ("verbatim", f"{rng.choice(P.VERBATIM_HEADERS)} {r['origin_text']} Share this.")]
        D = [{"role": "origin", "origin": f"{r['qid']}:wrong", "text": r["origin_text"], "copy_type": None}]
        D += [{"role": "copy", "origin": f"{r['qid']}:wrong", "text": t.strip(), "copy_type": ct}
              for ct, t in cps if _valid_wrong(t, r["wrong"], [r["answer"]])]
        for j, d in enumerate(D):
            d.update(doc_id=f"{r['qid']}#{j}", qid=r["qid"], domain=rng.choice(FAKE_SITES), date=0)
        docs += D
        qs.append({"qid": r["qid"], "question": r["question"], "aliases": [r["answer"]], "wrong": r["wrong"],
                   "paragraphs": r["paragraphs"], "gold_text": r["gold_text"]})
    write_jsonl(qs, OUT / "questions.jsonl")
    write_jsonl(docs, OUT / "docs.jsonl")
    log.info(f"hotpot echo: {len(qs)} questions, {len(docs)} echo docs")


def pool(cfg):
    import pandas as pd
    rng = random.Random(cfg["seed"])
    n = cfg["agent"]["global_pool"]
    eb = read_jsonl(DATA / "echobench" / "distractor_pool.jsonl")[:n]
    rows = [{"pid": f"eb{i}", "text": d["text"], "title": d.get("title", "")} for i, d in enumerate(eb)]
    hq = read_jsonl(OUT / "questions.jsonl")
    seen = set()
    for q in hq:
        for p in q["paragraphs"]:
            if p not in seen:
                seen.add(p)
                rows.append({"pid": f"hq{len(rows)}", "text": p, "title": p.split(":")[0]})
    df = pd.read_parquet(DATA / "raw" / "hotpotqa" / "distractor" / "validation-00000-of-00001.parquet")
    extra = []
    for _, r in df.iterrows():
        for t, p in _paragraphs(r):
            s = f"{t}: {p}"
            if s not in seen:
                seen.add(s)
                extra.append(s)
    rng.shuffle(extra)
    rows += [{"pid": f"hx{i}", "text": s, "title": s.split(":")[0]} for i, s in enumerate(extra[:n // 2])]
    write_jsonl(rows, DATA / "agent" / "pool.jsonl")
    log.info(f"agent pool: {len(rows)} passages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["build", "pool"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"build": build, "pool": pool}[a.phase](cfg)


if __name__ == "__main__":
    main()
