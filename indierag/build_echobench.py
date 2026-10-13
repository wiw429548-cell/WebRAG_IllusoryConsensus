"""EchoBench: controlled illusory-consensus benchmark built from REAL web search results (TriviaQA rc.web).

phase select : choose questions with >= min_answer_pages distinct-domain pages mentioning the gold answer
               (pairwise 5-gram Jaccard < 0.3 so that the correct pages are genuinely independent)
phase gen1   : builder LLM -> wrong answer, wrong ORIGIN document (rewrite of a real page), independent wrong
               documents (rewrites of *other* real pages), copies of the origin (backtranslation, paraphrase,
               AI-farm articles, summaries) + rule-based verbatim copies with boilerplate
phase gen2   : second LLM (content-farm diversity) -> more paraphrase / AI-farm copies
phase assemble -> data/echobench/{questions.jsonl, docs.jsonl}; every doc has an `origin` id (dependence label)
"""
import argparse
import glob
import random
from collections import defaultdict

from . import prompts as P
from .textutil import clean_page, domain, jaccard, shingles, window_around
from .utils import DATA, get_logger, load_config, mentions, normalize_answer, read_jsonl, save_json, write_jsonl

log = get_logger("echobench")
OUT = DATA / "echobench"
LANGS = ["French", "German", "Spanish", "Italian", "Portuguese"]
FAKE_SITES = ["dailyfacts-hub.com", "trivia-world.net", "knowitall-blog.com", "quickanswers.info", "factsnack.org",
              "the-infopage.com", "newsbrief24.net", "learnsomething.today", "wikiwiseguide.com", "answerbase-pro.com",
              "topfacts.site", "histobuzz.com", "nuggetsofknowledge.net", "mindfeed.news", "curio-central.com"]


def select(cfg):
    import pandas as pd
    ec = cfg["echobench"]
    rng = random.Random(cfg["seed"])
    files = sorted(glob.glob(str(DATA / "raw" / "triviaqa" / "rc.web" / "validation-*.parquet")))
    cands, pool = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["question", "question_id", "answer", "search_results"])
        for _, r in df.iterrows():
            ans = r["answer"]
            aliases = sorted({a for a in list(ans["aliases"]) + [ans["value"]] if a and normalize_answer(a)}, key=lambda a: (len(a), a))
            sr = r["search_results"]
            pages = []
            for url, title, ctx in zip(sr["url"], sr["title"], sr["search_context"]):
                if ctx and len(ctx.split()) > 80:
                    pages.append({"url": url, "title": title, "domain": domain(url), "text": clean_page(ctx)})
            ans_pages, non_ans = [], []
            seen_dom = set()
            for p in pages:
                if mentions(p["text"], aliases):
                    if p["domain"] in seen_dom:
                        continue
                    sh = shingles(window_around(p["text"], aliases, ec["passage_words"]))
                    if any(jaccard(sh, q["_sh"]) >= 0.3 for q in ans_pages):
                        continue  # real-web near-duplicate: not independent
                    p["_sh"] = sh
                    ans_pages.append(p)
                    seen_dom.add(p["domain"])
                else:
                    non_ans.append(p)
            item = {"qid": r["question_id"], "question": r["question"], "aliases": aliases, "value": ans["value"],
                    "ans_pages": ans_pages, "non_ans": non_ans}
            if len(ans_pages) >= ec["min_answer_pages"]:
                cands.append(item)
            else:
                pool.extend(non_ans[:2])
    log.info(f"{len(cands)} candidate questions with >= {ec['min_answer_pages']} independent answer pages")
    rng.shuffle(cands)
    chosen = cands[:ec["n_questions"]]
    pool.extend(p for c in cands[ec["n_questions"]:] for p in c["non_ans"][:2])
    qs = []
    for i, c in enumerate(chosen):
        mk = lambda p: {"title": p["title"], "domain": p["domain"], "url": p["url"],
                        "text": window_around(p["text"], c["aliases"], ec["passage_words"])}
        ap = [mk(p) for p in c["ans_pages"]]
        qs.append({"qid": c["qid"], "question": c["question"], "aliases": c["aliases"], "value": c["value"],
                   "split": "train" if i < ec["n_train"] else "test",
                   "correct": ap[:ec["max_correct"]], "origin_base": ap[ec["max_correct"]],
                   "indep_bases": ap[ec["max_correct"] + 1:] + [mk(p) for p in c["non_ans"]],
                   })
    write_jsonl(qs, OUT / "selected.jsonl")
    pool = [{"title": p["title"], "domain": p["domain"], "text": " ".join(p["text"].split()[:ec["passage_words"]])}
            for p in pool]
    rng.shuffle(pool)
    write_jsonl(pool[:100000], OUT / "distractor_pool.jsonl")
    log.info(f"selected {len(qs)} questions; distractor pool {min(len(pool), 100000)}")


def _valid_gold(text, wrong, aliases):
    return bool(text) and mentions(text, aliases) and not mentions(text, [wrong])


def _valid_wrong(text, wrong, aliases):
    return bool(text) and mentions(text, [wrong]) and not mentions(text, aliases)


def gen1(cfg):
    from .llm import LLM
    ec = cfg["echobench"]
    qs = read_jsonl(OUT / "selected.jsonl")
    llm = LLM(cfg["models"]["builder"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    nc = ec["copies_per_type"]
    # (a) wrong answers
    outs = llm.generate([P.WRONG_ANSWER.format(q=q["question"], a=q["value"]) for q in qs], temperature=0.8,
                        max_tokens=24, n=4, tag="wrong")
    keep = []
    for q, o in zip(qs, outs):
        cand = [x.strip().strip('."').split("\n")[0] for x in o]
        cand = [x for x in cand if x and normalize_answer(x) and not mentions(x, q["aliases"])
                and not any(normalize_answer(a) in normalize_answer(x) for a in q["aliases"])]
        if cand:
            q["wrong"] = cand[0]
            keep.append(q)
    log.info(f"gen1: {len(keep)}/{len(qs)} questions with a valid wrong answer")
    qs = keep
    # (b) origin + independent wrong rewrites
    rw = lambda base, q: P.REWRITE_WRONG.format(q=q["question"], wrong=q["wrong"], gold=q["value"], passage=base["text"])
    outs = llm.generate([rw(q["origin_base"], q) for q in qs], temperature=0.3, max_tokens=512, n=3, tag="origin")
    keep = []
    for q, o in zip(qs, outs):
        ok = [x.strip() for x in o if _valid_wrong(x, q["wrong"], q["aliases"])]
        if ok:
            q["origin_text"] = ok[0]
            keep.append(q)
    log.info(f"gen1: {len(keep)}/{len(qs)} questions with a valid wrong origin document")
    qs = keep
    jobs = [(i, b) for i, q in enumerate(qs) for b in q["indep_bases"][:ec["n_indep_wrong"] + 3]]
    outs = llm.generate([rw(b, qs[i]) for i, b in jobs], temperature=0.3, max_tokens=512, n=1, tag="indep")
    for q in qs:
        q["indep_wrong"] = []
    for (i, b), o in zip(jobs, outs):
        q = qs[i]
        if len(q["indep_wrong"]) < ec["n_indep_wrong"] and _valid_wrong(o[0], q["wrong"], q["aliases"]):
            q["indep_wrong"].append({"text": o[0].strip(), "base_domain": b["domain"]})
    # (c) copies of the origin
    for q in qs:
        q["copies"] = defaultdict(list)
    src = [q["origin_text"] for q in qs]
    # back-translation (two hops, each copy through a different pivot language)
    langs = [LANGS[j % len(LANGS)] for j in range(nc)]
    step1 = llm.generate([P.TRANSLATE.format(lang=l, text=s) for s in src for l in langs], temperature=0.7,
                         max_tokens=700, tag="bt1")
    step2 = llm.generate([P.TRANSLATE_BACK.format(text=t[0]) for t in step1], temperature=0.0, max_tokens=600, tag="bt2")
    for qi, q in enumerate(qs):
        for j in range(nc):
            q["copies"]["backtrans"].append({"text": step2[qi * nc + j][0].strip(), "gen": "builder"})
    half = (nc + 1) // 2
    para = llm.generate([P.PARAPHRASE.format(text=s) for s in src], temperature=0.9, max_tokens=600, n=half, tag="para")
    farm = llm.generate([P.AI_FARM.format(persona=P.AI_FARM_PERSONAS[j % 4], text=s) for s in src for j in range(half)],
                        temperature=0.9, max_tokens=600, tag="farm")
    summ = llm.generate([P.SUMMARY.format(text=s) for s in src], temperature=0.9, max_tokens=300, n=nc, tag="summ")
    for qi, q in enumerate(qs):
        q["copies"]["paraphrase"] += [{"text": t.strip(), "gen": "builder"} for t in para[qi]]
        q["copies"]["ai_farm"] += [{"text": farm[qi * half + j][0].strip(), "gen": "builder"} for j in range(half)]
        q["copies"]["summary"] += [{"text": t.strip(), "gen": "builder"} for t in summ[qi]]
    # rule-based verbatim copies (syndication with boilerplate)
    rng = random.Random(cfg["seed"])
    for q in qs:
        for j in range(nc):
            site = rng.choice(FAKE_SITES)
            h = rng.choice(P.VERBATIM_HEADERS)
            f = rng.choice(P.VERBATIM_FOOTERS).format(year=rng.randint(2015, 2024), site=site)
            body = q["origin_text"]
            if rng.random() < 0.5:  # light truncation, as scrapers often do
                ws = body.split()
                body = " ".join(ws[:max(40, int(len(ws) * rng.uniform(0.75, 1.0)))])
            q["copies"]["verbatim"].append({"text": f"{h} {body} {f}", "gen": "rule"})
    for q in qs:
        q["copies"] = dict(q["copies"])
    # (d) controls: (i) style-matched correct pages = LLM paraphrase of every correct page (removes the
    #     "human page = true, LLM page = false" shortcut); (ii) copies of a CORRECT page (setting C: syndicated truth)
    jobs = [(i, j, c["text"]) for i, q in enumerate(qs) for j, c in enumerate(q["correct"])]
    outs = llm.generate([P.PARAPHRASE.format(text=t) for _, _, t in jobs], temperature=0.7, max_tokens=600, tag="cpara")
    for q in qs:
        q["correct_para"], q["correct_copies"] = {}, []
    for (i, j, _), o in zip(jobs, outs):
        if _valid_gold(o[0].strip(), qs[i]["wrong"], qs[i]["aliases"]):
            qs[i]["correct_para"][str(j)] = o[0].strip()
    src0 = [q["correct"][0]["text"] for q in qs]
    cp = llm.generate([P.PARAPHRASE.format(text=t) for t in src0], temperature=0.9, max_tokens=600, n=2, tag="ccopy_para")
    cf = llm.generate([P.AI_FARM.format(persona=P.AI_FARM_PERSONAS[1], text=t) for t in src0], temperature=0.9,
                      max_tokens=600, tag="ccopy_farm")
    cs = llm.generate([P.SUMMARY.format(text=t) for t in src0], temperature=0.9, max_tokens=300, tag="ccopy_summ")
    for qi, q in enumerate(qs):
        for ct, t in [("paraphrase", cp[qi][0]), ("ai_farm", cf[qi][0]), ("summary", cs[qi][0]), ("paraphrase", cp[qi][1])]:
            if _valid_gold(t.strip(), q["wrong"], q["aliases"]):
                q["correct_copies"].append({"text": t.strip(), "copy_type": ct})
    write_jsonl(qs, OUT / "phase1.jsonl")
    log.info(f"gen1 done: {len(qs)} questions")


def gen2(cfg):
    from .llm import LLM
    ec = cfg["echobench"]
    qs = read_jsonl(OUT / "phase1.jsonl")
    try:
        llm = LLM(cfg["models"]["builder2"], max_model_len=min(4096, cfg["models"]["max_model_len"]),
                  gpu_mem=cfg["models"]["gpu_mem"])
    except Exception as e:  # second generator is optional (copy diversity); never block the pipeline
        log.warning(f"builder2 unavailable ({e}); continuing with builder copies only")
        save_json({"builder2_used": False, "model": cfg["models"]["builder2"], "error": str(e)[:500]},
                  OUT / "phase2_meta.json")
        write_jsonl([{"qid": q["qid"], "paraphrase": [], "ai_farm": []} for q in qs], OUT / "phase2.jsonl")
        return
    nc = ec["copies_per_type"]
    rest = nc - (nc + 1) // 2
    src = [q["origin_text"] for q in qs]
    para = llm.generate([P.PARAPHRASE.format(text=s) for s in src], temperature=0.9, max_tokens=600, n=rest, tag="para2")
    farm = llm.generate([P.AI_FARM.format(persona=P.AI_FARM_PERSONAS[(j + 2) % 4], text=s) for s in src for j in range(rest)],
                        temperature=0.9, max_tokens=600, tag="farm2")
    rows = []
    for qi, q in enumerate(qs):
        rows.append({"qid": q["qid"],
                     "paraphrase": [{"text": t.strip(), "gen": "builder2"} for t in para[qi]],
                     "ai_farm": [{"text": farm[qi * rest + j][0].strip(), "gen": "builder2"} for j in range(rest)]})
    save_json({"builder2_used": True, "model": cfg["models"]["builder2"], "engine": llm.engine_info},
              OUT / "phase2_meta.json")
    write_jsonl(rows, OUT / "phase2.jsonl")


def assemble(cfg):
    ec = cfg["echobench"]
    rng = random.Random(cfg["seed"] + 1)
    qs = read_jsonl(OUT / "phase1.jsonl")
    p2 = {r["qid"]: r for r in read_jsonl(OUT / "phase2.jsonl")} if (OUT / "phase2.jsonl").exists() else {}
    pool = list({p["text"]: p for p in read_jsonl(OUT / "distractor_pool.jsonl")}.values())  # unique texts
    docs, questions = [], []
    stats = defaultdict(int)

    def date(lo, hi):
        return int(rng.uniform(lo, hi))  # days since 2010-01-01

    # re-split among the questions that survived generation (keeps the train/test ratio of the config)
    n_tr = int(round(len(qs) * ec["n_train"] / ec["n_questions"]))
    for i, q in enumerate(qs):
        q["split"] = "train" if i < n_tr else "test"
    for q in qs:
        qid = q["qid"]
        D = []
        for j, c in enumerate(q["correct"]):
            D.append({"role": "correct", "origin": f"{qid}:correct{j}", "text": c["text"], "domain": c["domain"],
                      "date": date(700, 4000), "copy_type": None, "gen": "web"})
        od = date(3300, 4000)
        D.append({"role": "origin", "origin": f"{qid}:wrong", "text": q["origin_text"], "domain": rng.choice(FAKE_SITES),
                  "date": od, "copy_type": None, "gen": "builder"})
        copies = {k: list(v) for k, v in q["copies"].items()}
        for k in ["paraphrase", "ai_farm"]:
            copies[k] = copies.get(k, []) + p2.get(qid, {}).get(k, [])
        for ct, lst in copies.items():
            for c in lst:
                if not _valid_wrong(c["text"], q["wrong"], q["aliases"]):
                    stats[f"invalid_{ct}"] += 1
                    continue
                D.append({"role": "copy", "origin": f"{qid}:wrong", "text": c["text"], "domain": rng.choice(FAKE_SITES),
                          "date": od + rng.randint(1, 400), "copy_type": ct, "gen": c["gen"]})
                stats[f"valid_{ct}"] += 1
        for j, t in q.get("correct_para", {}).items():
            D.append({"role": "correct_para", "origin": f"{qid}:correct{j}", "text": t, "domain": rng.choice(FAKE_SITES),
                      "date": date(700, 4000), "copy_type": None, "gen": "builder", "para_of": int(j)})
        for c in q.get("correct_copies", []):
            D.append({"role": "correct_copy", "origin": f"{qid}:correct0", "text": c["text"], "domain": rng.choice(FAKE_SITES),
                      "date": date(3300, 4400), "copy_type": c["copy_type"], "gen": "builder"})
        for j, w in enumerate(q["indep_wrong"]):
            D.append({"role": "indep_wrong", "origin": f"{qid}:indepwrong{j}", "text": w["text"],
                      "domain": rng.choice(FAKE_SITES), "date": date(700, 4400), "copy_type": None, "gen": "builder"})
        # sampled WITHOUT replacement: two copies of the same pool page would be dependent but carry distinct origins
        n_d = min(len(pool), ec.get("n_distractor_pool", ec["n_distractors"] + 2))
        for j, p in enumerate(rng.sample(pool, n_d)):
            D.append({"role": "distractor", "origin": f"{qid}:distr{j}", "text": p["text"], "domain": p["domain"],
                      "date": date(700, 4400), "copy_type": None, "gen": "web"})
        for j, d in enumerate(D):
            d.update(doc_id=f"{qid}#{j}", qid=qid)
        docs.extend(D)
        questions.append({k: q[k] for k in ["qid", "question", "aliases", "value", "wrong", "split"]})
    write_jsonl(questions, OUT / "questions.jsonl")
    write_jsonl(docs, OUT / "docs.jsonl")
    log.info(f"assembled {len(questions)} questions, {len(docs)} docs; copy validity {dict(stats)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["select", "gen1", "gen2", "assemble"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    OUT.mkdir(parents=True, exist_ok=True)
    {"select": select, "gen1": gen1, "gen2": gen2, "assemble": assemble}[a.phase](cfg)


if __name__ == "__main__":
    main()
