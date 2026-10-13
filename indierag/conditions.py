"""Evaluation instances = (question, ordered list of documents) for every grid condition.

setting A ('echo') : 1 wrong origin + m copies of it   vs  k independent correct pages  (+ distractors)
setting B ('indep'): m independent wrong sources       vs  k independent correct pages  (+ distractors)
                     -> here the wrong answer IS the independent majority when m > k (sanity control: a good method
                        must NOT become contrarian; we report agreement with the independent majority)
group 'ranked'      : setting A, documents ordered by retriever score instead of randomly (source-bias realism)
group 'noisy'       : setting A with 0 / 6 / 10 irrelevant distractor pages (robustness to retrieval noise)
group 'style'       : setting A with STYLE-MATCHED correct pages (LLM paraphrases of the correct pages), so that
                      "LLM-written = wrong" cannot be exploited
setting C ('symC')  : copies of a CORRECT page (syndicated truth) vs w independent wrong sources -> shows where
                      counting independent origins costs accuracy (the independent majority is wrong when w > 1)
every instance stores n_indep_correct / n_indep_wrong -> agreement with the independent majority is reported
"""
import random
from collections import defaultdict

from .utils import DATA, read_jsonl

EB = DATA / "echobench"


def load_bench():
    qs = read_jsonl(EB / "questions.jsonl")
    docs = read_jsonl(EB / "docs.jsonl")
    by_q = defaultdict(list)
    for d in docs:
        by_q[d["qid"]].append(d)
    return qs, by_q


def test_qids(cfg, qs=None, n=None):
    """the first n test questions (sorted by qid) = the questions every EchoBench RAG run is evaluated on"""
    qs = qs if qs is not None else load_bench()[0]
    ids = sorted(q["qid"] for q in qs if q["split"] == "test")
    n = (cfg.get("rag") or {}).get("n_test_questions") if n is None else n
    return set(ids[:n] if n else ids)


def make_instances(cfg, split="test", groups=("main", "types", "indep"), max_questions=None):
    """max_questions: restrict to the first N questions of the split (sorted by qid); None = all"""
    qs, by_q = load_bench()
    cc, ec = cfg["conditions"], cfg["echobench"]
    inst = []
    keep = None
    if max_questions:
        keep = set(sorted(q["qid"] for q in qs if q["split"] == split)[:max_questions])
    for q in qs:
        if q["split"] != split or (keep is not None and q["qid"] not in keep):
            continue
        D = by_q[q["qid"]]
        correct = [d for d in D if d["role"] == "correct"]
        origin = [d for d in D if d["role"] == "origin"]
        copies = defaultdict(list)
        for d in D:
            if d["role"] == "copy":
                copies[d["copy_type"]].append(d)
        indep_wrong = [d for d in D if d["role"] == "indep_wrong"]
        all_distr = [d for d in D if d["role"] == "distractor"]
        distr = all_distr[:ec["n_distractors"]]
        rng = random.Random(f"{cfg['seed']}-{q['qid']}")

        def add(cond, docs, setting, m, k, order="random", group=None):
            docs = list(docs)
            rng.shuffle(docs)
            ic = len({d["origin"] for d in docs if d["role"] in ("correct", "correct_para", "correct_copy")})
            iw = len({d["origin"] for d in docs if d["role"] in ("origin", "copy", "indep_wrong")})
            inst.append({"iid": f"{q['qid']}|{cond}", "qid": q["qid"], "question": q["question"],
                         "aliases": q["aliases"], "wrong": q["wrong"], "cond": cond, "group": group, "setting": setting,
                         "m": m, "k": k,
                         "doc_ids": [d["doc_id"] for d in docs], "order": order, "n_indep_correct": ic, "n_indep_wrong": iw})

        mixed = [d for t in ec["copy_types"] for d in copies[t]]
        random.Random(f"{cfg['seed']}-{q['qid']}-mixed").shuffle(mixed)

        if "main" in groups:
            for m in cc["main"]["m"]:
                for k in cc["main"]["k"]:
                    pool = [d for t in ec["copy_types"] for d in copies[t]]
                    rng2 = random.Random(f"{cfg['seed']}-{q['qid']}-mixed")
                    rng2.shuffle(pool)  # nested: the m copies for m=4 include those for m=2
                    if len(pool) >= m and len(correct) >= k:
                        add(f"A_mixed_m{m}_k{k}", correct[:k] + origin + pool[:m] + distr, "A", m, k, group="main")
        if "types" in groups:
            for t in cc["types"]["type"]:
                for m in cc["types"]["m"]:
                    for k in cc["types"]["k"]:
                        if len(copies[t]) >= m and len(correct) >= k:  # a condition never has fewer copies than its m
                            add(f"A_{t}_m{m}_k{k}", correct[:k] + origin + copies[t][:m] + distr, "A", m, k, group="types")
        if "ranked" in groups and "ranked" in cc:
            for m in cc["ranked"]["m"]:
                for k in cc["ranked"]["k"]:
                    if len(mixed) >= m and len(correct) >= k:
                        add(f"R_mixed_m{m}_k{k}", correct[:k] + origin + mixed[:m] + distr, "A", m, k, order="ranked",
                            group="ranked")
        if "noisy" in groups and "noisy" in cc:
            for m in cc["noisy"]["m"]:
                for k in cc["noisy"]["k"]:
                    for nd in cc["noisy"]["d"]:
                        if len(all_distr) >= nd and len(mixed) >= m and len(correct) >= k:
                            add(f"N_mixed_m{m}_k{k}_d{nd}", correct[:k] + origin + mixed[:m] + all_distr[:nd], "A", m, k,
                                group="noisy")
        cpara = [d for d in D if d["role"] == "correct_para"]
        ccopies = [d for d in D if d["role"] == "correct_copy"]
        if "style" in groups and "style" in cc and len(cpara) >= max(cc["style"]["k"]):
            for m in cc["style"]["m"]:
                for k in cc["style"]["k"]:
                    if len(mixed) >= m:
                        add(f"S_mixed_m{m}_k{k}", cpara[:k] + origin + mixed[:m] + distr, "A", m, k, group="style")
        if "symC" in groups and "symC" in cc:
            for m in cc["symC"]["m"]:
                for w in cc["symC"]["w"]:
                    if len(ccopies) >= m and len(indep_wrong) >= w:
                        add(f"C_m{m}_w{w}", correct[:1] + ccopies[:m] + indep_wrong[:w] + distr, "C", m, w, group="symC")
        if "indep" in groups:
            for m in cc["indep"]["m_indep"]:
                for k in cc["indep"]["k"]:
                    if len(indep_wrong) >= m and len(correct) >= k:
                        add(f"B_indep_m{m}_k{k}", correct[:k] + indep_wrong[:m] + distr, "B", m, k, group="indep")
    return inst


def doc_index():
    return {d["doc_id"]: d for d in read_jsonl(EB / "docs.jsonl")}
