"""Collect every result into results/summary.md, results/latex/*.tex (booktabs tables \\input by paper/main.tex)
and results/figures/*.{pdf,png}.

Only real outputs are used; a missing number is printed as "--". Averages over a set of conditions are only reported
when the method was run on ALL of them (methods with a reduced condition scope get "--" instead of a biased mean).
Readers whose vLLM engine needed a fallback (results/engine_log.jsonl: reduced context, TP clamp, eager mode) are
marked with a dagger.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .utils import ROOT, load_config, slug

R = ROOT / "results"
TEX = R / "latex"
FIG = R / "figures"

NAMES = {
    "vanilla": "Vanilla RAG", "dedup": "Dedup (5-gram)", "semdedup": "SemDedup (emb.)", "majority": "Majority vote",
    "indievote:lex": "Near-dup + vote", "indievote:cos": "SemDedup + vote", "astute": "Astute RAG",
    "instructrag": "InstructRAG-ICL", "trustrag": "TrustRAG (filter)", "madam": "MADAM-RAG", "cag": "CAG (prompted)",
    "indiebayes:none": "ACCU (no copy det.)", "indievote:H": "IndieVote-H", "indievote:L": "IndieVote-L",
    "indievote:LLM": "IndieVote-LLM", "indievote:oracle": "IndieVote-oracle", "indiebayes:L": "IndieBayes-L",
    "indierag:H": "IndieRAG-H", "indierag:L": "IndieRAG-L", "indierag:L+meta": "IndieRAG-L+meta",
    "indierag:LLM": "IndieRAG-LLM", "indierag:oracle": "IndieRAG-oracle",
    "indierag_table:L": "IndieRAG-L: table + all docs", "indierag_compress:L": "IndieRAG-L: grouped docs, no table",
}
BASELINES = ["vanilla", "dedup", "semdedup", "majority", "indievote:lex", "indievote:cos", "astute", "instructrag",
             "trustrag", "madam", "cag", "indiebayes:none"]
OURS = ["indievote:H", "indievote:L", "indievote:LLM", "indiebayes:L", "indierag:H", "indierag:L", "indierag:L+meta",
        "indierag:LLM"]
ORACLE = ["indievote:oracle", "indierag:oracle"]
KEY = ["vanilla", "majority", "astute", "indievote:L", "indiebayes:L", "indierag:L"]
COPY_TYPES = ["verbatim", "backtrans", "paraphrase", "ai_farm", "summary"]


def name(m):
    if m in NAMES:
        return NAMES[m]
    if m.startswith("indievote:L["):
        return "IndieVote-L " + m[len("indievote:L"):]
    if "#" in m:
        b, t = m.split("#")
        return f"{NAMES.get(b, b)} ($\\theta$={t})"
    return m


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None
    except json.JSONDecodeError:
        return None


def read_jsonl(p):
    if not Path(p).exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def isnum(x):
    return x is not None and isinstance(x, (int, float)) and x == x


def pct(x, nd=1):
    return "--" if not isnum(x) else f"{100 * x:.{nd}f}"


def num(x, nd=3):
    return "--" if not isnum(x) else f"{x:.{nd}f}"


def tex_esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


# ---------------------------------------------------------------------------------------------- metric access
def get(mt, meth, cond, key="acc"):
    v = ((mt or {}).get(meth) or {}).get(cond) or {}
    return v.get(key)


def avg(mt, meth, conds, key="acc"):
    """mean over conds; None unless the method has ALL of them (no biased partial averages)"""
    vals = [get(mt, meth, c, key) for c in conds]
    return float(np.mean(vals)) if conds and all(isnum(v) for v in vals) else None


def avg_avail(mt, meth, conds, key):
    vals = [get(mt, meth, c, key) for c in conds]
    vals = [v for v in vals if isnum(v)]
    return float(np.mean(vals)) if vals else None


def slope(mt, meth, ms, k=2, key="acc"):
    ys = [get(mt, meth, f"A_mixed_m{m}_k{k}", key) for m in ms]
    if len(ms) < 2 or not all(isnum(y) for y in ys):
        return None
    return float(np.polyfit(np.log2(np.array(ms) + 1), ys, 1)[0])


def tokens(mt, meth):
    c = ((mt or {}).get(meth) or {}).get("_cost_per_instance")
    return (c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)) if c else None


# ---------------------------------------------------------------------------------------------- tables
def best_rows(rows, col, exclude=(), higher=True):
    vals = [(r["_vals"].get(col), i) for i, r in enumerate(rows)
            if r["_key"] not in exclude and isnum(r["_vals"].get(col))]
    if not vals:
        return set()
    b = max(v for v, _ in vals) if higher else min(v for v, _ in vals)
    return {i for v, i in vals if abs(v - b) < 1e-12}


def write_table(fname, caption, label, header, rows, fmt, bold_cols=(), lower_better=(), notes="", wide=False,
                exclude_bold=tuple(ORACLE)):
    """rows: list of {"_key", "_label", "_vals": {col: value}}; header: list of (col, title); fmt: col -> fn"""
    TEX.mkdir(parents=True, exist_ok=True)
    bold = {c: best_rows(rows, c, exclude_bold, higher=c not in lower_better) for c in bold_cols}
    env = "table*" if wide else "table"
    L = [f"\\begin{{{env}}}[tbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{l" + "r" * len(header) + "}", "\\toprule",
         " & ".join(["Method"] + [h for _, h in header]) + " \\\\", "\\midrule"]
    prev_group = None
    for i, r in enumerate(rows):
        if r.get("_group") is not None and prev_group is not None and r["_group"] != prev_group:
            L.append("\\midrule")
        prev_group = r.get("_group")
        cells = []
        for c, _ in header:
            s = fmt.get(c, pct)(r["_vals"].get(c))
            if i in bold.get(c, set()) and s != "--":
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        L.append(" & ".join([r["_label"]] + cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        L.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}")
    L.append(f"\\end{{{env}}}")
    (TEX / fname).write_text("\n".join(L) + "\n", encoding="utf-8")


def md_table(header, rows, fmt):
    L = ["| method | " + " | ".join(h for _, h in header) + " |", "|---" * (len(header) + 1) + "|"]
    for r in rows:
        L.append(f"| {r['_label_md']} | " + " | ".join(fmt.get(c, pct)(r["_vals"].get(c)) for c, _ in header) + " |")
    return L


def mkrow(key, vals, group=None):
    return {"_key": key, "_label": tex_esc(name(key)) if "$" not in name(key) else name(key).replace("_", "\\_"),
            "_label_md": name(key), "_vals": vals, "_group": group}


# ---------------------------------------------------------------------------------------------- statistics
def preds(reader, ds, meth):
    rows = read_jsonl(R / "rag" / reader / ds / "preds" / f"{meth.replace(':', '_')}.jsonl")
    return {r["iid"]: r for r in rows}


def cluster_boot(diff_by_q, n_boot=5000, seed=0):
    """question-level cluster bootstrap of the mean paired difference -> (mean, lo, hi, p two-sided)"""
    qs = sorted(diff_by_q)
    sums = np.array([np.sum(diff_by_q[q]) for q in qs])
    cnts = np.array([len(diff_by_q[q]) for q in qs])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(qs), size=(n_boot, len(qs)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    mean = sums.sum() / cnts.sum()
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(mean), float(np.quantile(boots, .025)), float(np.quantile(boots, .975)), min(1.0, max(float(p), 1.0 / n_boot))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


def significance(reader, conds_ok, ours="indierag:L", baselines=BASELINES, ds="echobench", key="acc"):
    from scipy.stats import binomtest
    o = preds(reader, ds, ours)
    if not o:
        return []
    out = []
    for base in baselines:
        b = preds(reader, ds, base)
        ids = [i for i in o if i in b and conds_ok(o[i]["cond"]) and isnum(o[i].get(key)) and isnum(b[i].get(key))]
        if len(ids) < 10:
            continue
        d = defaultdict(list)
        n01 = n10 = 0
        for i in ids:
            q = o[i].get("qid") or i.split("|")[0]
            d[q].append(o[i][key] - b[i][key])
            n01 += int(o[i][key] == 1 and b[i][key] == 0)
            n10 += int(o[i][key] == 0 and b[i][key] == 1)
        mean, lo, hi, p = cluster_boot(d)
        pm = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        out.append({"baseline": base, "n_inst": len(ids), "n_q": len(d), "delta": mean, "lo": lo, "hi": hi, "p": p,
                    "p_mcnemar": float(pm)})
    if out:
        for r, a in zip(out, holm([r["p"] for r in out])):
            r["p_holm"] = a
    return out


def acc_ci(reader, meth, conds_ok, n_boot=2000):
    p = preds(reader, "echobench", meth)
    d = defaultdict(list)
    for i, r in p.items():
        if conds_ok(r["cond"]):
            d[r.get("qid") or i.split("|")[0]].append(r["acc"])
    if not d:
        return None
    _, lo, hi, _ = cluster_boot(d, n_boot)
    return lo, hi


def fp(p):
    return "--" if not isnum(p) else ("<0.001" if p < 1e-3 else f"{p:.3f}")


# ---------------------------------------------------------------------------------------------- engine fallbacks
def engine_fallbacks():
    fb = defaultdict(list)
    for e in read_jsonl(R / "engine_log.jsonl"):
        if e.get("fallback"):
            fb[slug(e.get("model", "?"))].append(
                f"tp={e.get('tp')}/{e.get('tp_requested')}, max_len={e.get('max_model_len')}, "
                f"eager={e.get('enforce_eager')}, gpu_mem={e.get('gpu_mem')} ({e.get('argv', '')[:120]})")
    return fb


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    M, cc = cfg["models"], cfg["conditions"]
    TEX.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fb = engine_fallbacks()
    dag = lambda r: "$^\\dagger$" if r in fb else ""  # noqa: E731
    readers_cfg = [slug(r) for r in M["readers"]]
    extra = [slug(r) for r in (M.get("extra_readers") or [])]
    large = slug(M["large_reader"])
    have = sorted(p.name for p in (R / "rag").glob("*")) if (R / "rag").exists() else []
    full_readers = [r for r in readers_cfg if r in have] or readers_cfg[:1]
    primary = readers_cfg[0]
    MS, KS = cc["main"]["m"], cc["main"]["k"]
    echo_conds = [f"A_mixed_m{m}_k{k}" for m in MS if m > 0 for k in KS]
    clean_conds = [f"A_mixed_m0_k{k}" for k in KS]
    B_conds = [f"B_indep_m{m}_k{k}" for m in cc["indep"]["m_indep"] for k in cc["indep"]["k"]]
    is_echo = lambda c: c in set(echo_conds)  # noqa: E731
    L = ["# IndieRAG — experiment summary (auto-generated by `python -m indierag.aggregate`)", "",
         "All numbers are computed from the files under `results/`; `--` = not available (job not run / failed, or the "
         "method was not run on every condition of the column). Accuracy = answer contains a gold alias.", ""]
    if fb:
        L += ["**Engine fallbacks** (results/engine_log.jsonl; marked with † in the LaTeX tables):", ""] + \
             [f"- {k}: " + "; ".join(v) for k, v in fb.items()] + [""]
    missing = []

    # ============================================================ RQ1/RQ2 main tables per reader
    for r in full_readers + [x for x in extra + [large] if x in have]:
        m = jl(R / "rag" / r / "echobench" / "metrics.json")
        if not m:
            missing.append(f"rag/{r}/echobench")
        mt = (m or {}).get("metrics", {})
        methods = [x for x in BASELINES + OURS + ORACLE if x in mt] if mt else BASELINES + OURS + ORACLE
        rows = []
        for meth in methods:
            vals = {"clean": avg(mt, meth, clean_conds), "m2": get(mt, meth, "A_mixed_m2_k2"),
                    "m8": get(mt, meth, "A_mixed_m8_k2"), "echo": avg(mt, meth, echo_conds),
                    "strict": avg(mt, meth, echo_conds, "acc_strict"), "wrong": avg(mt, meth, echo_conds, "wrong"),
                    "slope": slope(mt, meth, MS), "bagree": avg_avail(mt, meth, B_conds, "indep_majority_agree")
                    if all(isnum(get(mt, meth, c)) for c in B_conds) else None, "tok": tokens(mt, meth)}
            g = 0 if meth in BASELINES else (2 if meth in ORACLE else 1)
            rows.append(mkrow(meth, vals, g))
        header = [("clean", "Clean"), ("m2", "$m{=}2$"), ("m8", "$m{=}8$"), ("echo", "Echo avg"),
                  ("wrong", "Wrong$\\downarrow$"), ("slope", "Slope"), ("bagree", "B-agree"), ("tok", "Tok/inst")]
        fmt = {"slope": lambda x: num(x, 3), "tok": lambda x: "--" if not isnum(x) else f"{x / 1000:.1f}k"}
        nq = (m or {}).get("n_questions")
        fname = "tab_main.tex" if r == primary else f"tab_main_{r}.tex"
        write_table(fname, f"EchoBench setting A (reader {tex_esc(r)}{dag(r)}; {nq or '--'} test questions). "
                    "Accuracy (\\%) without copies (Clean, $m{=}0$), with $m$ copies of one wrong origin vs.\\ $k{=}2$ "
                    "independent correct pages, and averaged over $m\\geq1$, $k\\in\\{1,2,3\\}$ (Echo avg). Wrong: rate "
                    "of answering the injected wrong answer. Slope: $d\\,$acc$/d\\log_2(m{+}1)$ at $k{=}2$. B-agree: "
                    "agreement with the independent majority in control setting B. Tok/inst: reader tokens per "
                    "instance.", f"tab:main:{r}" if r != primary else "tab:main", header, rows, fmt,
                    bold_cols=["clean", "m2", "m8", "echo", "strict", "bagree"], lower_better=["wrong"], wide=True,
                    notes="-- = method not run on every condition of the column (reduced scope) or result missing.")
        L += [f"## RQ1/RQ2 — EchoBench, reader `{r}`{' †' if r in fb else ''} ({nq or '--'} questions, "
              f"{(m or {}).get('n_instances', '--')} instances)", ""] + md_table(header, rows, fmt) + [""]
        # CIs of echo accuracy (question-cluster bootstrap)
        ci_rows = []
        for meth in KEY + ["indierag:LLM", "indierag:oracle"]:
            if meth in mt and isnum(avg(mt, meth, echo_conds)):
                ci = acc_ci(r, meth, is_echo)
                if ci:
                    ci_rows.append(f"{name(meth)} {pct(avg(mt, meth, echo_conds))} [{pct(ci[0])}, {pct(ci[1])}]")
        if ci_rows:
            L += ["Echo-avg accuracy with 95% question-cluster bootstrap CIs: " + "; ".join(ci_rows), ""]
        if r not in full_readers:
            continue
        # ---- significance
        sig = significance(r, is_echo)
        if sig or r == primary:
            srows = [mkrow(s["baseline"], {"n": s["n_inst"], "d": s["delta"], "ci": (s["lo"], s["hi"]), "p": s["p"],
                                           "ph": s["p_holm"], "pm": s["p_mcnemar"]}) for s in sig] or \
                [mkrow(b, {}) for b in BASELINES]
            sh = [("n", "\\#inst"), ("d", "$\\Delta$acc"), ("ci", "95\\% CI"), ("p", "$p$"), ("ph", "$p_{\\text{Holm}}$"),
                  ("pm", "$p_{\\text{McNemar}}$")]
            sf = {"n": lambda x: "--" if x is None else str(x), "d": lambda x: "--" if not isnum(x) else f"{100 * x:+.1f}",
                  "ci": lambda x: "--" if not x else f"[{100 * x[0]:+.1f}, {100 * x[1]:+.1f}]", "p": fp, "ph": fp,
                  "pm": fp}
            write_table("tab_significance.tex" if r == primary else f"tab_significance_{r}.tex",
                        f"IndieRAG-L vs.\\ baselines on all setting-A instances with $m\\geq1$ shared by both methods "
                        f"(reader {tex_esc(r)}). $\\Delta$ in accuracy points; CI and $p$ from a question-level cluster "
                        "bootstrap (5000 resamples), Holm-corrected over baselines; McNemar on instances for reference.",
                        "tab:sig" if r == primary else f"tab:sig:{r}", sh, srows, sf)
            L += ["Paired significance, IndieRAG-L vs baseline (setting A, m>=1; question-cluster bootstrap, Holm):", ""] + \
                md_table(sh, srows, sf) + [""]
        # ---- copy types
        rows = [mkrow(meth, {t: get(mt, meth, f"A_{t}_m4_k2") for t in COPY_TYPES} | {"mixed": get(mt, meth, "A_mixed_m4_k2")})
                for meth in methods if not mt or any(isnum(get(mt, meth, f"A_{t}_m4_k2")) for t in COPY_TYPES)]
        header = [(t, tex_esc(t)) for t in COPY_TYPES] + [("mixed", "mixed")]
        write_table("tab_copytypes.tex" if r == primary else f"tab_copytypes_{r}.tex",
                    f"Accuracy (\\%) by copy type ($m{{=}}4$ copies, $k{{=}}2$; reader {tex_esc(r)}{dag(r)}).",
                    "tab:types" if r == primary else f"tab:types:{r}", header, rows, {}, bold_cols=[t for t, _ in header])
        L += ["Copy type (m=4, k=2):", ""] + md_table(header, rows, {}) + [""]
        # ---- style-matched control
        rows = []
        for meth in methods:
            vals = {}
            for x in cc["style"]["m"]:
                vals[f"A{x}"] = get(mt, meth, f"A_mixed_m{x}_k2")
                vals[f"S{x}"] = get(mt, meth, f"S_mixed_m{x}_k2")
            if not mt or any(isnum(vals[f"S{x}"]) for x in cc["style"]["m"]):
                rows.append(mkrow(meth, vals))
        header = [(f"{p}{x}", f"{'orig' if p == 'A' else 'style'} $m{{=}}{x}$") for x in cc["style"]["m"] for p in "AS"]
        write_table("tab_style.tex" if r == primary else f"tab_style_{r}.tex",
                    f"Style-matched control: correct pages are also LLM-paraphrased (style), so that ``LLM-written = "
                    f"wrong'' cannot be exploited; $k{{=}}2$ (reader {tex_esc(r)}{dag(r)}).",
                    "tab:style" if r == primary else f"tab:style:{r}", header, rows, {}, wide=True,
                    bold_cols=[c for c, _ in header])
        L += ["Style-matched control (orig vs style-matched correct pages, k=2):", ""] + md_table(header, rows, {}) + [""]
        # ---- setting C + ranked + noisy
        rows = []
        for meth in methods:
            vals = {f"C{mm}{w}": get(mt, meth, f"C_m{mm}_w{w}") for mm in cc["symC"]["m"] for w in cc["symC"]["w"]}
            vals["Cagree"] = get(mt, meth, f"C_m{max(cc['symC']['m'])}_w{max(cc['symC']['w'])}", "indep_majority_agree")
            if not mt or any(isnum(v) for v in vals.values()):
                rows.append(mkrow(meth, vals))
        header = [(f"C{mm}{w}", f"$m{{=}}{mm},w{{=}}{w}$") for mm in cc["symC"]["m"] for w in cc["symC"]["w"]] + \
                 [("Cagree", "indep.-agree")]
        write_table("tab_symC.tex" if r == primary else f"tab_symC_{r}.tex",
                    f"Setting C (limitation probe): $m$ copies of a TRUE page vs.\\ $w$ independent wrong sources "
                    f"(accuracy \\%); for $w{{=}}2$ the independent majority is wrong. indep.-agree: agreement with the "
                    f"independent-origin majority at the largest $m,w$ (reader {tex_esc(r)}{dag(r)}).",
                    "tab:symC" if r == primary else f"tab:symC:{r}", header, rows, {}, bold_cols=[c for c, _ in header])
        L += ["Setting C (copies of the TRUE page vs w independent wrong):", ""] + md_table(header, rows, {}) + [""]
        rows = []
        for meth in methods:
            vals = {f"R{x}": get(mt, meth, f"R_mixed_m{x}_k2") for x in cc["ranked"]["m"]}
            vals |= {f"N{d}": get(mt, meth, f"N_mixed_m4_k2_d{d}") for d in cc["noisy"]["d"]}
            if not mt or any(isnum(v) for v in vals.values()):
                rows.append(mkrow(meth, vals))
        header = [(f"R{x}", f"ranked $m{{=}}{x}$") for x in cc["ranked"]["m"]] + \
                 [(f"N{d}", f"$d{{=}}{d}$") for d in cc["noisy"]["d"]]
        write_table("tab_ranked_noisy.tex" if r == primary else f"tab_ranked_noisy_{r}.tex",
                    f"Robustness: documents in dense-retriever order (ranked, $k{{=}}2$) and with $d$ irrelevant "
                    f"distractor pages ($m{{=}}4,k{{=}}2$); accuracy \\% (reader {tex_esc(r)}{dag(r)}).",
                    "tab:robust" if r == primary else f"tab:robust:{r}", header, rows, {}, wide=True,
                    bold_cols=[c for c, _ in header])
        L += ["Retriever order (ranked) and retrieval noise (d distractors, m=4,k=2):", ""] + md_table(header, rows, {}) + [""]
        # ---- ablations (components, estimators, features)
        abl = ["indierag:L", "indierag_table:L", "indierag_compress:L", "indierag:H", "indierag:LLM", "indierag:L+meta",
               "indierag:oracle", "majority", "indievote:H", "indievote:L", "indievote:LLM", "indievote:oracle"] + \
            sorted(x for x in mt if x.startswith("indievote:L[")) + ["indiebayes:none", "indiebayes:L"]
        rows = []
        for meth in abl:
            g = 0 if meth.startswith("indierag") else (1 if meth.startswith("indievote") or meth == "majority" else 2)
            rows.append(mkrow(meth, {"clean": avg(mt, meth, clean_conds), "echo": avg(mt, meth, echo_conds),
                                     "bagree": avg_avail(mt, meth, B_conds, "indep_majority_agree")
                                     if all(isnum(get(mt, meth, c)) for c in B_conds) else None,
                                     "dep": ((mt.get(meth) or {}).get("_dep_sec_per_instance")),
                                     "tok": tokens(mt, meth)}, g))
        header = [("clean", "Clean"), ("echo", "Echo avg"), ("bagree", "B-agree"), ("dep", "dep.\\ ms/inst"),
                  ("tok", "Tok/inst")]
        fmt = {"dep": lambda x: "--" if not isnum(x) else f"{1000 * x:.0f}",
               "tok": lambda x: "--" if not isnum(x) else f"{x / 1000:.1f}k"}
        write_table("tab_ablation.tex" if r == primary else f"tab_ablation_{r}.tex",
                    f"Ablations (reader {tex_esc(r)}{dag(r)}): IndieRAG components (evidence table / origin-grouped "
                    "documents), dependence estimators (H heuristic, L learned, LLM judge, oracle), and IndieVote with "
                    "the learned estimator trained without one feature group. dep.\\ ms/inst: dependence-estimation "
                    "time per instance.", "tab:ablation" if r == primary else f"tab:ablation:{r}", header, rows, fmt,
                    bold_cols=["clean", "echo", "bagree"])
        L += ["Ablations:", ""] + md_table(header, rows, fmt) + [""]
        # ---- threshold sensitivity
        ths = sorted({float(x.split("#")[1]) for x in mt if x.startswith("indierag:L#")} | {cfg["dependence"]["hard_threshold"]})
        rows = []
        for t in ths:
            meth = "indierag:L" if t == cfg["dependence"]["hard_threshold"] else f"indierag:L#{t}"
            rows.append(mkrow(meth, {f"m{x}": get(mt, meth, f"A_mixed_m{x}_k2") for x in MS}))
        header = [(f"m{x}", f"$m{{=}}{x}$") for x in MS]
        write_table("tab_threshold.tex" if r == primary else f"tab_threshold_{r}.tex",
                    f"Sensitivity of IndieRAG-L to the clustering threshold $\\theta$ ($k{{=}}2$; reader "
                    f"{tex_esc(r)}{dag(r)}).", "tab:thr" if r == primary else f"tab:thr:{r}", header, rows, {})
        L += ["Clustering-threshold sensitivity (k=2):", ""] + md_table(header, rows, {}) + [""]
        if m and m.get("claim_extraction_accuracy"):
            L += ["Per-document claim extraction accuracy by document role: " +
                  ", ".join(f"{k}: {pct(v)}" for k, v in m["claim_extraction_accuracy"].items()), ""]

    # ============================================================ scale / family table (key methods, all readers)
    all_r = readers_cfg + extra + [large]
    rows = []
    for meth in KEY:
        vals = {}
        for r in all_r:
            mt = (jl(R / "rag" / r / "echobench" / "metrics.json") or {}).get("metrics", {})
            vals[f"{r}|c"] = avg(mt, meth, clean_conds)
            vals[f"{r}|e"] = avg(mt, meth, echo_conds)
        rows.append(mkrow(meth, vals))
    shown = [r for r in all_r if any(isnum(rw["_vals"][f"{r}|e"]) for rw in rows)] or readers_cfg[:1]
    header = []
    for r in shown:
        short = r.split("_", 1)[-1][:18]
        header += [(f"{r}|c", f"{tex_esc(short)}{dag(r)} clean"), (f"{r}|e", "echo")]
    write_table("tab_readers.tex", "Key methods across readers (model family and scale; main grid). Clean: $m{=}0$; "
                "echo: average over $m\\geq1$, $k\\in\\{1,2,3\\}$. Accuracy \\%.", "tab:readers", header, rows, {},
                wide=True, bold_cols=[c for c, _ in header])
    L += ["## Readers (family / scale), key methods: clean / echo-avg accuracy", ""] + md_table(header, rows, {}) + [""]
    for r in all_r:
        if not (R / "rag" / r / "echobench" / "metrics.json").exists():
            missing.append(f"rag/{r}/echobench")

    # ============================================================ RAMDocs
    rows = []
    rd_readers = [r for r in readers_cfg if (R / "rag" / r / "ramdocs" / "metrics.json").exists()] or readers_cfg[:1]
    rds = {r: (jl(R / "rag" / r / "ramdocs" / "metrics.json") or {}).get("metrics", {}) for r in rd_readers}
    for meth in BASELINES + [x for x in OURS if x != "indierag:L+meta"] + ["indierag_table:L", "indierag_compress:L"]:
        vals = {}
        for r in rd_readers:
            al = (rds[r].get(meth) or {}).get("_all") or {}
            vals[f"{r}|all"], vals[f"{r}|w"] = al.get("acc_all"), al.get("wrong")
        rows.append(mkrow(meth, vals, 0 if meth in BASELINES else 1))
    header = []
    for r in rd_readers:
        header += [(f"{r}|all", f"{tex_esc(r.split('_', 1)[-1][:18])}{dag(r)} acc"), (f"{r}|w", "wrong$\\downarrow$")]
    write_table("tab_ramdocs.tex", "RAMDocs (realistic ambiguity + misinformation + noise; 500 questions). acc: all "
                "valid answers and no misinformation answer; wrong: a misinformation answer is given (\\%).",
                "tab:ramdocs", header, rows, {}, bold_cols=[c for c, _ in header if c.endswith("|all")],
                lower_better=[c for c, _ in header if c.endswith("|w")])
    L += ["## RAMDocs", ""] + md_table(header, rows, {}) + [""]
    # paired significance on RAMDocs (one instance per question -> plain bootstrap over questions), primary reader
    sh = [("n", "\\#q"), ("d", "$\\Delta$acc"), ("ci", "95\\% CI"), ("p", "$p$"), ("ph", "$p_{\\text{Holm}}$"),
          ("pm", "$p_{\\text{McNemar}}$")]
    sf = {"n": lambda x: "--" if x is None else str(x), "d": lambda x: "--" if not isnum(x) else f"{100 * x:+.1f}",
          "ci": lambda x: "--" if not x else f"[{100 * x[0]:+.1f}, {100 * x[1]:+.1f}]", "p": fp, "ph": fp, "pm": fp}
    for r in rd_readers:
        sig = significance(r, lambda c: True, ds="ramdocs", key="acc_all")
        if not sig and r != primary:
            continue
        srows = [mkrow(s["baseline"], {"n": s["n_inst"], "d": s["delta"], "ci": (s["lo"], s["hi"]), "p": s["p"],
                                       "ph": s["p_holm"], "pm": s["p_mcnemar"]}) for s in sig] or \
            [mkrow(b, {}) for b in BASELINES]
        write_table("tab_significance_ramdocs.tex" if r == primary else f"tab_significance_ramdocs_{r}.tex",
                    f"RAMDocs: IndieRAG-L vs.\\ baselines (reader {tex_esc(r)}{dag(r)}; metric: all valid answers and "
                    "no misinformation answer). $\\Delta$ in points; CI and $p$ from a bootstrap over questions (5000 "
                    "resamples), Holm-corrected over baselines; McNemar for reference.",
                    "tab:sig:ramdocs" if r == primary else f"tab:sig:ramdocs:{r}", sh, srows, sf)
        L += [f"RAMDocs paired significance, IndieRAG-L vs baseline (reader {r}):", ""] + md_table(sh, srows, sf) + [""]

    # ============================================================ cost / efficiency (primary reader, EchoBench)
    ROUNDS = {"vanilla": 1, "dedup": 1, "semdedup": 1, "trustrag": 1, "instructrag": 1, "astute": 2, "cag": 2,
              "madam": 3, "majority": 1, "indievote": 1, "indiebayes": 1, "indierag": 2, "indierag_table": 2,
              "indierag_compress": 2}
    mt = (jl(R / "rag" / primary / "echobench" / "metrics.json") or {}).get("metrics", {})
    rows = []
    for meth in BASELINES + OURS:
        c = (mt.get(meth) or {}).get("_cost_per_instance") or {}
        rounds = ROUNDS.get(meth, ROUNDS.get(meth.split(":")[0]))
        rows.append(mkrow(meth, {"calls": c.get("calls"), "pt": c.get("prompt_tokens"), "ct": c.get("completion_tokens"),
                                 "rounds": rounds, "dep": (mt.get(meth) or {}).get("_dep_sec_per_instance"),
                                 "echo": avg(mt, meth, echo_conds)}, 0 if meth in BASELINES else 1))
    header = [("calls", "LLM calls"), ("pt", "prompt tok"), ("ct", "gen.\\ tok"), ("rounds", "seq.\\ rounds"),
              ("dep", "dep.\\ ms"), ("echo", "Echo acc")]
    kfmt = lambda x: "--" if not isnum(x) else (f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}")  # noqa: E731
    fmt = {"calls": lambda x: num(x, 1), "pt": kfmt, "ct": kfmt, "rounds": lambda x: "--" if x is None else str(x),
           "dep": lambda x: "--" if not isnum(x) else f"{1000 * x:.0f}"}
    write_table("tab_cost.tex", f"Inference cost per instance (EchoBench, reader {tex_esc(primary)}{dag(primary)}), "
                "counted as if uncached and stand-alone: reader LLM calls, prompt and generated tokens (including the "
                "per-document claim extraction of the vote/IndieRAG methods, the per-document credibility ratings of "
                "CAG and the LLM pair judgements of the -LLM variants), number of sequential LLM rounds (latency), "
                "and dependence-estimation time per instance.", "tab:cost",
                header, rows, fmt, bold_cols=["echo"])
    L += ["## Cost per instance (EchoBench, primary reader)", ""] + md_table(header, rows, fmt) + [""]
    for r in readers_cfg:
        if not (R / "rag" / r / "ramdocs" / "metrics.json").exists():
            missing.append(f"rag/{r}/ramdocs")

    # ============================================================ RQ3 dependence detection
    d = jl(R / "dependence" / "metrics.json")
    if not d:
        missing.append("dependence")
    dv = (d or {}).get("variants", {})
    order = ["H", "L[full]", "L[full+meta]", "LLM", "lex", "cos", "L[-lexical]", "L[-semantic]", "L[-rare]", "L[-aigc]",
             "L[lexical_only]"] + sorted(k for k in dv if k.startswith("L[n_train"))
    ci = lambda x: "--" if not x else f"[{100 * x[0]:.1f}, {100 * x[1]:.1f}]"  # noqa: E731
    dlab = {"lex": "near-dup (5-gram $\\geq$0.5)", "cos": "SemDedup (cos $\\geq$" +
            str((cfg.get("dependence") or {}).get("semdedup_cos", 0.95)) + ")"}
    rows = []
    for v in order:
        x = dv.get(v, {})
        sub = ((d or {}).get("variants_llm_subset") or {}).get(v, {})
        rows.append({"_key": v, "_label": dlab.get(v, tex_esc(v)), "_label_md": v, "_group": 0 if "n_train" not in v else 1,
                     "_vals": {"auc": x.get("AUC"), "auc_ci": x.get("AUC_ci"), "ap": x.get("AP"), "f1": x.get("F1@0.5"),
                               "ari": x.get("ARI"), "auc_sub": sub.get("AUC"),
                               "ms": ((d or {}).get("time_sec_per_pool") or {}).get(v)}})
    header = [("auc", "AUC"), ("auc_ci", "95\\% CI"), ("ap", "AP"), ("f1", "F1@0.5"), ("ari", "ARI"),
              ("auc_sub", "AUC (LLM subset)"), ("ms", "ms/pool")]
    fmt = {"auc_ci": ci, "ari": lambda x: num(x, 3), "ms": lambda x: "--" if not isnum(x) else f"{1000 * x:.0f}"}
    nq, nl = (d or {}).get("n_questions", "--"), (d or {}).get("n_questions_llm_subset", "--")
    write_table("tab_dependence.tex", f"Shared-origin detection on the document pools of {nq} EchoBench test questions "
                f"(pairwise AUC/AP/F1 in \\%, CI from a question-level bootstrap; ARI of the clustering at the default "
                f"threshold). The LLM judge is evaluated on the first {nl} pools only; all estimators are also reported "
                f"on that subset. Lower block: learning curve (number of training questions).", "tab:dep", header,
                rows, fmt, bold_cols=["auc", "ap", "f1", "ari", "auc_sub"], exclude_bold=())
    L += ["## RQ3 — dependence (shared-origin) detection", "",
          f"pairs: {(d or {}).get('n_pairs', '--')}, positive rate {num((d or {}).get('pos_rate'))}; "
          f"second copy generator used: {(d or {}).get('builder2_used', '--')}", ""] + md_table(header, rows, fmt) + [""]
    if d:
        v0 = "L[full]" if "L[full]" in dv else (next(iter(dv)) if dv else None)
        if v0:
            L += [f"Per pair type ({v0}): recall@0.5 for dependent pairs, FPR@0.5 for independent pairs", ""]
            for t, x in dv[v0].get("by_type", {}).items():
                L.append(f"- {t}: n={x['n']}, " + ", ".join(f"{k}={num(v)}" for k, v in x.items() if k != "n"))
        L += ["", "ESS absolute error (wrong group should be 1, correct group k): " +
              ", ".join(f"{k}={num(v)}" for k, v in d.get("ess_abs_error", {}).items()), ""]
        if d.get("cross_generator_applicable") is False or d.get("builder2_used") is False:
            L += ["Cross-generator generalisation: n/a (the second copy generator was not available).", ""]
        elif d.get("recall_vs_origin_by_generator"):
            L += ["Copy detection recall by copy generator (trained on all vs. trained WITHOUT the second generator):", ""] + \
                 [f"- {k}: n={v['n']}, recall={num(v['recall@0.5'])}" for k, v in d["recall_vs_origin_by_generator"].items()] + [""]
        if d.get("cross_dataset_hotpotqa"):
            L += ["Cross-dataset (TriviaQA-web pools -> HotpotQA/Wikipedia echoes): " +
                  ", ".join(f"{k}: AUC={num(v['AUC'])} AP={num(v['AP'])} (n={v['n_pairs']})"
                            for k, v in d["cross_dataset_hotpotqa"].items()), ""]
        if d.get("retriever_source_bias"):
            L += ["Retriever source bias (bge; mean normalised rank, 0 = top):", ""] + \
                 [f"- {k}: {num(v['mean_normalised_rank (0=top)'])} (n={v['n']})" for k, v in d["retriever_source_bias"].items()] + [""]
    # by copy type table for L[full] and H
    rows = []
    for v in ["lex", "cos", "H", "L[full]", "LLM"]:
        bt = (dv.get(v) or {}).get("by_type", {})
        vals = {}
        for t in COPY_TYPES:
            ks = [k for k in bt if k.startswith("dep:") and t in k and "origin" in k]
            vals[t] = bt[ks[0]].get("recall@0.5") if ks else None
        hn = [k for k in bt if k.startswith("indep:")]
        vals["fpr"] = float(np.average([bt[k]["fpr@0.5"] for k in hn], weights=[bt[k]["n"] for k in hn])) if hn else None
        rows.append({"_key": v, "_label": dlab.get(v, tex_esc(v)), "_label_md": v, "_vals": vals})
    header = [(t, tex_esc(t)) for t in COPY_TYPES] + [("fpr", "FPR (indep.)")]
    write_table("tab_dep_types.tex", "Recall@0.5 of origin--copy pairs by copy type, and false-positive rate on "
                "independent pairs (\\%).", "tab:deptypes", header, rows, {}, exclude_bold=())
    L += ["Origin-copy recall by copy type / FPR on independent pairs:", ""] + md_table(header, rows, {}) + [""]

    # ============================================================ RQ4 agents
    rows = []
    runs = sorted((R / "agent").glob("*/metrics.json")) if (R / "agent").exists() else []
    for p in runs:
        mt = (jl(p) or {}).get("metrics", {})
        pr = read_jsonl(p.parent / "preds.jsonl")
        by_q = defaultdict(list)
        for x in pr:
            if x.get("m", 0) > 0:
                by_q[x["iid"].split("|")[0]].append(x["acc"])
        c = cluster_boot(by_q, 2000) if by_q else None
        rows.append({"_key": p.parent.name, "_label": tex_esc(p.parent.name), "_label_md": p.parent.name,
                     "_vals": {"a0": (mt.get("m0") or {}).get("acc"), "a4": (mt.get("m4") or {}).get("acc"),
                               "ci": (c[1], c[2]) if c else None, "w4": (mt.get("m4") or {}).get("wrong"),
                               "ns": (mt.get("_all") or {}).get("n_search"), "n": (mt.get("m4") or {}).get("n")}})
    if not rows:
        missing.append("agent")
    header = [("a0", "acc $m{=}0$"), ("a4", "acc $m{=}4$"), ("ci", "95\\% CI"), ("w4", "wrong $m{=}4$"),
              ("ns", "\\#search"), ("n", "$n$")]
    fmt = {"ci": ci, "ns": lambda x: num(x, 2), "n": lambda x: "--" if x is None else str(x)}
    write_table("tab_agents.tex", "Search agents with a local dense search engine: ReAct (instruction model), Search-R1 "
                "(RL-trained) and IndieSearch (ReAct + evidence-status feedback + seek-independent step). $m$: injected "
                "copies of a wrong origin.", "tab:agents", header, rows, fmt, bold_cols=["a0", "a4"],
                lower_better=["w4"], exclude_bold=())
    L += ["## RQ4 — search agents", ""] + md_table(header, rows, fmt) + [""]
    # paired IndieSearch - ReAct difference (same policy model and benchmark), question-cluster bootstrap
    for p in runs:
        nm = p.parent.name
        if not (nm.endswith("_indie") or "_indie_tau" in nm):
            continue
        base = p.parent.parent / (nm[:nm.rindex("_indie")] + "_react") / "preds.jsonl"
        a_ = {x["iid"]: x for x in read_jsonl(p.parent / "preds.jsonl")}
        b_ = {x["iid"]: x for x in read_jsonl(base)}
        for mm in (0, 4):
            dq = defaultdict(list)
            for i, x in a_.items():
                if i in b_ and x.get("m") == mm:
                    dq[i.split("|")[0]].append(x["acc"] - b_[i]["acc"])
            if dq:
                mean, lo, hi, pv = cluster_boot(dq, 5000)
                L.append(f"- {p.parent.name} vs ReAct, m={mm}: Δacc={100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}], "
                         f"p={fp(pv)}, n={len(dq)} questions")
    L.append("")

    # ============================================================ RQ5 web measurement
    w = jl(R / "web_measure" / "metrics.json")
    if not w:
        missing.append("web_measure")
    wc = (w or {}).get("ci95", {})
    items = [("n_questions", "questions"), ("n_pages", "pages"), ("frac_pages_lexical_duplicate", "pages with a lexical near-duplicate"),
             ("frac_pages_dependent_cluster", "pages in a dependent cluster (IndieRAG-L)"),
             ("mean_support_inflation_gold", "support inflation of the gold answer (\\#pages / ESS)"),
             ("n_questions_with_conflict", "questions with $\\geq2$ answer groups"),
             ("page_vs_independent_majority_disagree", "page vs.\\ independent majority disagree"),
             ("acc_page_majority", "accuracy of page majority"), ("acc_independent_majority", "accuracy of independent majority"),
             ("acc_vanilla", "Vanilla RAG accuracy"), ("acc_majority", "Majority vote accuracy"),
             ("acc_indievote:L", "IndieVote-L accuracy"), ("acc_indierag:L", "IndieRAG-L accuracy")]

    def wfmt(k, v):
        if not isnum(v):
            return "--"
        if k.startswith(("frac", "acc", "page_vs")):
            s = pct(v)
            c = wc.get(k)
            return s + (f" [{pct(c[0])}, {pct(c[1])}]" if c else "")
        if k == "mean_support_inflation_gold":
            c = wc.get(k)
            return f"{v:.2f}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else "")
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"
    TL = ["\\begin{table}[tbp]", "\\centering", "\\small",
          "\\caption{Measurement on real web search results (TriviaQA rc.web cached Bing results; offline snapshot). "
          "Percentages with 95\\% bootstrap CIs.}", "\\label{tab:web}", "\\begin{tabular}{lr}", "\\toprule",
          "Quantity & Value \\\\", "\\midrule"]
    TL += [f"{lab} & {wfmt(k, (w or {}).get(k))} \\\\" for k, lab in items]
    TL += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "tab_web.tex").write_text("\n".join(TL) + "\n", encoding="utf-8")
    L += ["## RQ5 — real web search results (TriviaQA rc.web, offline snapshot)", ""] + \
         [f"- {lab.replace('$' + chr(92) + 'geq', '>=').replace('$', '').replace(chr(92), '')}: {wfmt(k, (w or {}).get(k))}" for k, lab in items] + [""]
    # every method on the natural web pages, per web reader: accuracy (all / conflict questions), paired delta vs.
    # vanilla RAG and tokens per question (claim extraction added for the claim-based methods)
    web_runs = ([(w.get("reader") or M["readers"][0], w)] if w else []) + \
        [(x.get("reader") or p.parent.name, x) for p in sorted((R / "web_measure").glob("*/metrics.json"))
         if (x := jl(p))]
    claim_based = lambda m: m == "majority" or m.split(":")[0] in ("indievote", "indiebayes", "indierag")  # noqa: E731
    for wi, (wr, wx) in enumerate(web_runs or [(M["readers"][0], {})]):
        wci, cpq, pr_ = wx.get("ci95", {}), wx.get("cost_per_question", {}), wx.get("paired", {})
        cc_ = wx.get("claim_extraction_cost_per_question") or {}
        wm = [m for m in BASELINES + OURS if f"acc_{m}" in wx] or ["vanilla", "majority", "indievote:L", "indierag:L"]
        rows = []
        for meth in wm:
            pv = (pr_.get(f"{meth}|vanilla") or {})
            po = (pr_.get(f"indierag:L|{meth}") or {})
            c = cpq.get(meth) or {}
            tok = (c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) +
                   ((cc_.get("prompt_tokens", 0) + cc_.get("completion_tokens", 0)) if claim_based(meth) else 0)) if c else None
            rows.append(mkrow(meth, {"acc": wx.get(f"acc_{meth}"), "acc_ci": wci.get(f"acc_{meth}"),
                                     "conf": wx.get(f"accconf_{meth}"), "dv": (pv.get("all") or {}),
                                     "dvc": (pv.get("conflict") or {}), "do": (po.get("conflict") or {}), "tok": tok},
                              0 if meth in BASELINES else 1))
        dfmt = lambda x: "--" if not x else (f"{100 * x['delta']:+.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]")  # noqa: E731
        header = [("acc", "acc"), ("acc_ci", "95\\% CI"), ("conf", "acc (conflict)"), ("dv", "$\\Delta$ vs.\\ vanilla"),
                  ("dvc", "$\\Delta$ vs.\\ vanilla (conflict)"), ("do", "IndieRAG-L $-$ method (conflict)"),
                  ("tok", "tok/q")]
        fmt = {"acc_ci": ci, "dv": dfmt, "dvc": dfmt, "do": dfmt,
               "tok": lambda x: "--" if not isnum(x) else (f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}")}
        fn = "tab_web_methods.tex" if wi == 0 else f"tab_web_methods_{slug(wr)}.tex"
        write_table(fn, f"All methods on real web search results (TriviaQA rc.web cached Bing pages, "
                    f"{wx.get('n_questions', '--')} questions, {wx.get('n_conflict_questions', '--')} of them with "
                    f"$\\geq$2 answer groups among the pages = ``conflict''; reader {tex_esc(slug(wr))}). Accuracy in \\%; "
                    "$\\Delta$ in points with paired bootstrap 95\\% CIs over questions; tokens per question as if uncached.",
                    "tab:webmethods" if wi == 0 else f"tab:webmethods:{slug(wr)}", header, rows, fmt,
                    bold_cols=["acc", "conf"], wide=True)
        L += [f"Web pages, all methods (reader {wr}):", ""] + md_table(header, rows, fmt) + [""]
        if not wx:
            missing.append("web_measure (methods)")

    # ============================================================ metric validity: LLM judge vs. string match
    jd = jl(R / "judge" / "metrics.json")
    if not jd:
        missing.append("judge")
    rows = []
    for ds in ("echobench", "ramdocs"):
        x = ((jd or {}).get("datasets") or {}).get(ds) or {}
        rows.append({"_key": ds, "_label": {"echobench": "EchoBench ($m\\geq1$, strict)", "ramdocs": "RAMDocs (all valid)"}[ds],
                     "_label_md": ds, "_vals": {"n": x.get("n_judgements"), "agr": x.get("agreement"), "kap": x.get("kappa"),
                                                "yn": x.get("judge_yes_metric_no"), "ny": x.get("judge_no_metric_yes"),
                                                "tau": x.get("kendall_tau_method_ranking")}})
    header = [("n", "\\#judgements"), ("agr", "agreement"), ("kap", "Cohen $\\kappa$"), ("yn", "judge yes / metric no"),
              ("ny", "judge no / metric yes"), ("tau", "Kendall $\\tau$ (ranking)")]
    fmt = {"n": lambda x: "--" if x is None else str(x), "kap": lambda x: num(x, 2), "tau": lambda x: num(x, 2)}
    write_table("tab_judge.tex", f"Validity of the string-match metric: an LLM judge ({tex_esc((jd or {}).get('judge') or M.get('judge') or '--')}) "
                f"re-grades predictions of reader {tex_esc(slug((jd or {}).get('reader') or M['readers'][0]))} (same random "
                "instances for every method). Agreement and disagreement rates in \\%; Kendall $\\tau$ between the method "
                "rankings under the judge and under the metric.", "tab:judge", header, rows, fmt, exclude_bold=())
    L += ["## Metric validity (LLM judge vs. string match)", ""] + md_table(header, rows, fmt) + [""]
    if jd:
        for ds, x in (jd.get("datasets") or {}).items():
            L.append(f"- {ds}: " + ", ".join(f"{m}: judge={pct(v['judge_acc'])} metric={pct(v['metric_acc'])}"
                                            for m, v in x.get("per_method", {}).items()))
        L.append("")

    # ============================================================ human validation, theory, cases
    hv = jl(R / "annotation" / "human_validation.json")
    L += ["## Human validation", ""] + ([f"- {k}: {v}" for k, v in hv.items()] if hv else
                                       ["- not available (annotate the CSVs in results/annotation/, then run "
                                        "`python -m indierag.annotate score`)"]) + [""]
    t = jl(R / "theory" / "simulation.json")
    if t:
        L += ["## Theory — copy-corrected Condorcet simulation (accuracy of majority vote)", ""]
        for k, dd in t.items():
            L.append(f"- {k}: " + "; ".join(f"m={mm}: naive {v['0.0']:.3f} / perfect detector {v['1.0']:.3f}"
                                           for mm, v in dd.items()))
        L.append("")
    L += ["## Case studies (IndieRAG-L correct, vanilla RAG wrong, m=4)", ""] + case_studies(primary)
    L += ["## Missing results", ""] + ([f"- {x}" for x in sorted(set(missing))] or ["- none"]) + [""]
    (R / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    figures(cfg, full_readers, all_r, primary)
    print("wrote", R / "summary.md", "and", TEX)


def case_studies(r, n=4):
    out = []
    o, v = preds(r, "echobench", "indierag:L"), preds(r, "echobench", "vanilla")
    picks = [i for i in o if i in v and o[i]["acc"] == 1 and v[i]["acc"] == 0 and "_m4_" in o[i]["cond"]][:n]
    for i in picks:
        out += [f"**{i}** — document roles: {', '.join(str(x) for x in (o[i].get('roles') or []))}", "",
                f"- vanilla RAG: {v[i]['pred']}", f"- IndieRAG-L: {o[i]['pred']}",
                f"- IndieRAG-L reasoning (excerpt): {o[i]['raw'][:400]}", ""]
    return out or ["- not available", ""]


def figures(cfg, readers, all_r, primary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(FIG / f"{stem}.pdf")
        fig.savefig(FIG / f"{stem}.png", dpi=150)
        plt.close(fig)
    MS, KS = cfg["conditions"]["main"]["m"], cfg["conditions"]["main"]["k"]
    for r in readers:
        m = jl(R / "rag" / r / "echobench" / "metrics.json")
        if not m:
            continue
        fig, axes = plt.subplots(1, len(KS), figsize=(3.4 * len(KS), 2.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, KS):
            for meth in ["vanilla", "majority", "dedup", "astute", "madam", "indievote:L", "indierag:L", "indierag:oracle"]:
                d = m["metrics"].get(meth)
                if not d:
                    continue
                ms = [x for x in MS if f"A_mixed_m{x}_k{k}" in d]
                if len(ms) >= 2:
                    ax.plot(ms, [100 * d[f"A_mixed_m{x}_k{k}"]["acc"] for x in ms], marker="o", ms=3, label=name(meth),
                            ls="--" if "oracle" in meth else "-")
            ax.set_xscale("symlog", base=2)
            ax.set_xticks(MS)
            ax.set_xticklabels([str(x) for x in MS])
            ax.set_title(f"k={k} independent correct pages", fontsize=9)
            ax.set_xlabel("#copies m of the wrong origin")
        axes[0].set_ylabel("accuracy (%)")
        axes[-1].legend(fontsize=6)
        save(fig, "susceptibility" if r == primary else f"susceptibility_{r}")
    # learning curve of the dependence classifier
    d = jl(R / "dependence" / "metrics.json")
    if d:
        info = jl(ROOT / "data" / "features" / "dep_clf_info.json") or {}
        pts = []
        for k, v in d["variants"].items():
            if k.startswith("L[n_train="):
                pts.append((int(k.split("=")[1].rstrip("]")), v["AUC"], v.get("AUC_ci")))
        if "L[full]" in d["variants"] and info.get("n_train_questions_all"):
            pts.append((info["n_train_questions_all"], d["variants"]["L[full]"]["AUC"], d["variants"]["L[full]"].get("AUC_ci")))
        if len(pts) >= 2:
            pts.sort()
            fig, ax = plt.subplots(figsize=(3.4, 2.5))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
            lo = [p[2][0] if p[2] else p[1] for p in pts]
            hi = [p[2][1] if p[2] else p[1] for p in pts]
            ax.fill_between([p[0] for p in pts], lo, hi, alpha=0.2)
            if "H" in d["variants"]:
                ax.axhline(d["variants"]["H"]["AUC"], ls="--", c="gray", label="H (training-free)")
                ax.legend(fontsize=7)
            ax.set_xscale("log")
            ax.set_xlabel("#training questions")
            ax.set_ylabel("pairwise AUC")
            save(fig, "dep_learning_curve")
    # theory
    t = jl(R / "theory" / "simulation.json")
    if t:
        fig, axes = plt.subplots(1, len(t), figsize=(3.4 * len(t), 2.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (k, dd) in zip(axes, t.items()):
            ms = sorted(dd, key=int)
            for rc in sorted(next(iter(dd.values())), key=float):
                ax.plot([int(x) for x in ms], [dd[x][rc] for x in ms], marker="o", ms=3, label=f"recall={rc}")
            ax.set_xscale("symlog", base=2)
            ax.set_title(k, fontsize=9)
            ax.set_xlabel("#copies m of one origin")
        axes[0].set_ylabel("majority-vote accuracy")
        axes[-1].legend(fontsize=6)
        save(fig, "theory_simulation")
    # scale / family: echo accuracy of key methods per reader
    vals = {}
    for r in all_r:
        mt = (jl(R / "rag" / r / "echobench" / "metrics.json") or {}).get("metrics")
        if mt:
            echo = [f"A_mixed_m{x}_k{k}" for x in MS if x > 0 for k in KS]
            vals[r] = {meth: avg(mt, meth, echo) for meth in KEY}
    if len(vals) >= 2:
        fig, ax = plt.subplots(figsize=(6.8, 2.6))
        w = 0.8 / len(KEY)
        for i, meth in enumerate(KEY):
            ys = [100 * vals[r][meth] if isnum(vals[r][meth]) else 0 for r in vals]
            ax.bar(np.arange(len(vals)) + i * w, ys, w, label=name(meth))
        ax.set_xticks(np.arange(len(vals)) + 0.4 - w / 2)
        ax.set_xticklabels([r.split("_", 1)[-1][:22] for r in vals], fontsize=7)
        ax.set_ylabel("echo accuracy (%)")
        ax.legend(fontsize=6, ncol=3)
        save(fig, "readers")


if __name__ == "__main__":
    main()
