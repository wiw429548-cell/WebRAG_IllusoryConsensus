"""Validity of the automatic accuracy metric: an LLM judge (models.judge, a different model than the primary reader)
re-scores a random sample of the primary reader's predictions (same instances for every method) on EchoBench
(setting A, m>=1) and RAMDocs. Reports agreement / Cohen's kappa between the judge and the string-match metric and
the judge accuracy per method (does the method ranking survive a semantic judge?).

python -m indierag.judge_eval --reader Qwen/Qwen2.5-7B-Instruct
-> results/judge/metrics.json
"""
import argparse
import random
import re

import numpy as np

from .llm import LLM
from .utils import DATA, ROOT, get_logger, load_config, read_jsonl, save_json, slug, write_jsonl

JUDGE_ECHO = (
    "You are grading a question-answering system.\n\nQuestion: {q}\nReference answer (any of these aliases is "
    "correct): {golds}\nKnown incorrect answer: {wrong}\nSystem answer: {pred}\n\n"
    "Is the system answer correct, i.e. it refers to the same entity/value as the reference answer and does not also "
    "assert the known incorrect answer? Answer with exactly one word: YES or NO.")
JUDGE_RAMDOCS = (
    "You are grading a question-answering system on an ambiguous question that may have several valid answers.\n\n"
    "Question: {q}\nValid answers (ALL of them should be given): {golds}\nMisinformation answers (must NOT be given): "
    "{wrong}\nSystem answer: {pred}\n\nDoes the system answer give every valid answer (paraphrases and answers with or "
    "without units count) and none of the misinformation answers? Answer with exactly one word: YES or NO.")


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return None
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)


def gold_info(cfg, ds):
    """iid -> (question, golds, wrongs)"""
    if ds == "ramdocs":
        rows = read_jsonl(DATA / "raw" / "ramdocs" / "RAMDocs_test.jsonl")
        return {f"ramdocs{i}": (r["question"], r.get("gold_answers", []), r.get("wrong_answers", []))
                for i, r in enumerate(rows)}
    from .conditions import make_instances
    return {it["iid"]: (it["question"], it["aliases"], [it["wrong"]])
            for it in make_instances(cfg, "test", ("main",), max_questions=(cfg.get("rag") or {}).get("n_test_questions"))}


def run(cfg, reader, judge_model, log):
    jc = cfg.get("judge_eval") or {}
    methods = jc.get("methods") or ["vanilla", "majority", "astute", "indievote:L", "indierag:L"]
    llm = None
    res = {"reader": reader, "judge": judge_model, "datasets": {}}
    rows_out = []
    for ds, n, key in (("echobench", jc.get("n_echobench", 100), "acc_strict"), ("ramdocs", jc.get("n_ramdocs", 60), "acc_all")):
        base = ROOT / "results" / "rag" / slug(reader) / ds / "preds"
        P = {m: {r["iid"]: r for r in read_jsonl(base / f"{m.replace(':', '_')}.jsonl")} for m in methods
             if (base / f"{m.replace(':', '_')}.jsonl").exists()}
        P = {m: v for m, v in P.items() if v}
        if not P:
            log.info(f"{ds}: no predictions of {reader} -> skipped")
            continue
        ok = lambda r: ds != "echobench" or bool(re.match(r"A_mixed_m[1-9]", r.get("cond", "")))  # noqa: E731
        shared = sorted(set.intersection(*[{i for i, r in v.items() if ok(r)} for v in P.values()]))
        if not shared:  # a method with a narrow scope (e.g. MADAM-RAG) would empty the intersection -> drop it
            P = {m: v for m, v in P.items() if len([i for i, r in v.items() if ok(r)]) >= n} or P
            shared = sorted(set.intersection(*[{i for i, r in v.items() if ok(r)} for v in P.values()]))
        info = gold_info(cfg, ds)
        shared = [i for i in shared if i in info]
        ids = random.Random(cfg["seed"]).sample(shared, min(n, len(shared)))
        if not ids:
            continue
        if llm is None:
            llm = LLM(judge_model, max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
        tmpl = JUDGE_ECHO if ds == "echobench" else JUDGE_RAMDOCS
        items = [(m, i) for m in P for i in ids]
        prompts = []
        for m, i in items:
            q, g, w = info[i]
            prompts.append([{"role": "user", "content": tmpl.format(
                q=q, golds="; ".join(g[:8]) or "(none)", wrong="; ".join(w[:4]) or "(none)",
                pred=(P[m][i].get("pred") or "(no answer)")[:500])}])
        outs = llm.chat(prompts, max_tokens=4, tag="judge_metric")
        judge = [float("yes" in o.lower()) for o in outs]
        metric = [float(P[m][i].get(key, 0.0)) for m, i in items]
        loose = [float(P[m][i].get("acc", 0.0)) for m, i in items]
        per = {}
        for m in P:
            sel = [k for k, (mm, _) in enumerate(items) if mm == m]
            per[m] = {"judge_acc": float(np.mean([judge[k] for k in sel])), "metric_acc": float(np.mean([metric[k] for k in sel])),
                      "loose_acc": float(np.mean([loose[k] for k in sel])), "n": len(sel)}
        jm, mm_ = np.array(judge), np.array(metric)
        from scipy.stats import kendalltau
        tau = kendalltau([per[m]["judge_acc"] for m in per], [per[m]["metric_acc"] for m in per])[0] if len(per) >= 3 else None
        res["datasets"][ds] = {"metric": key, "n_instances": len(ids), "n_judgements": len(items), "methods": list(P),
                               "agreement": float((jm == mm_).mean()), "kappa": kappa(jm, mm_),
                               "agreement_loose_acc": float((jm == np.array(loose)).mean()),
                               "kappa_loose_acc": kappa(jm, np.array(loose)),
                               "judge_yes_metric_no": float(((jm == 1) & (mm_ == 0)).mean()),
                               "judge_no_metric_yes": float(((jm == 0) & (mm_ == 1)).mean()),
                               "kendall_tau_method_ranking": None if tau is None or tau != tau else float(tau),
                               "per_method": per}
        rows_out += [{"ds": ds, "method": m, "iid": i, "pred": P[m][i].get("pred"), "metric": x, "judge": j, "judge_raw": o}
                     for (m, i), x, j, o in zip(items, metric, judge, outs)]
        log.info(f"{ds}: agreement={res['datasets'][ds]['agreement']:.3f} kappa={res['datasets'][ds]['kappa']}")
    return res, rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, help="reader whose predictions are judged")
    ap.add_argument("--judge", default=None, help="default: models.judge")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    judge = a.judge or cfg["models"].get("judge") or "Qwen/Qwen3-8B"
    if slug(judge) == slug(a.reader):
        raise SystemExit("the judge must differ from the reader whose predictions it judges")
    out = ROOT / "results" / "judge"
    log = get_logger("judge_eval", out / "run.log")
    res, rows = run(cfg, a.reader, judge, log)
    write_jsonl(rows, out / "judgements.jsonl")
    save_json(res, out / "metrics.json")


if __name__ == "__main__":
    main()
