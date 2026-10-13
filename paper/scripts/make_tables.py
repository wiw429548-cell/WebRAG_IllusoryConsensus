"""Styled paper tables (gray/white banding + colour accents) built from the auto-generated result tables.

Usage (from the repository root):  python paper/scripts/make_tables.py
Reads results/latex/*.tex (written by `python -m indierag.aggregate` from the raw result files) and re-emits a
selection of rows/columns into paper/tables/*.tex. Numbers are copied verbatim as strings; nothing is recomputed or
typed in by hand. Styling macros (\\best, \\scnd, \\hdr, \\ours, \\grp, \\sigp, \\sign, \\nsig) live in main.tex.
"""
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "latex"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SECOND = "Qwen_Qwen3_8B"

RENAME = {"Dedup (5-gram)": "Near-dup removal", "SemDedup (emb.)": "SemDeDup", "SemDedup + vote": "SemDeDup + vote", "ACCU (no copy det.)": "ACCU (no copy det.)",
          "TrustRAG (filter)": "TrustRAG (filter)", "CAG (prompted)": "CAG (prompted)"}
OURS_PREFIX = ("Indie",)


def load(name):
    """name -> OrderedDict(row label -> list of cell strings); also returns the header cells"""
    rows, header = OrderedDict(), None
    for line in open(SRC / f"{name}.tex", encoding="utf-8"):
        line = line.strip()
        if "&" not in line or not line.endswith("\\\\"):
            continue
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c.strip()) for c in line[:-2].split("&")]
        if cells[0] == "Method":
            header = cells
            continue
        rows[cells[0].replace("\_", "_")] = cells[1:]
    return rows, header


def fnum(s):
    try:
        return float(s.replace("k", "").replace("+", ""))
    except ValueError:
        return None


def nice(s):
    """pretty-print a numeric cell: true minus signs"""
    s = s.strip()
    if s in ("--", ""):
        return "--"
    return re.sub(r"(?<![\w$])-(?=\d)", "$-$", s)


def disp(label):
    lab = RENAME.get(label, label).replace("_", "\_")
    lab = lab.replace("IndieRAG", "\\sys").replace("IndieVote", "IndieVote").replace("IndieBayes", "IndieBayes")
    return lab


def mark_best(table, cols, higher=True, exclude=()):
    """table: list of (label, [cells]); returns {(row, col): 'best'|'scnd'}"""
    out = {}
    for c in cols:
        vals = sorted({fnum(r[1][c]) for r in table if r[0] not in exclude and r[1] and fnum(r[1][c]) is not None},
                      reverse=higher)
        if not vals:
            continue
        for i, (lab, cells) in enumerate(table):
            if lab in exclude or not cells:
                continue
            v = fnum(cells[c])
            if v is None:
                continue
            if v == vals[0]:
                out[(i, c)] = "best"
            elif len(vals) > 1 and v == vals[1]:
                out[(i, c)] = "scnd"
    return out


def render(table, marks, ncols, ours_rows=()):
    lines = []
    for i, (lab, cells) in enumerate(table):
        if cells is None:  # group header
            lines.append(f"\\grp{{{ncols}}}{{{lab}}}")
            continue
        out = []
        for c, x in enumerate(cells):
            x = nice(x)
            m = marks.get((i, c))
            out.append(f"\\best{{{x}}}" if m == "best" else f"\\scnd{{{x}}}" if m == "scnd" else x)
        pre = "\\ours " if lab in ours_rows or (lab.startswith(OURS_PREFIX) and "oracle" not in lab) else ""
        lines.append(pre + disp(lab) + " & " + " & ".join(out) + " \\\\")
    return "\n".join(lines)


def write(name, body):
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------- T1 main
def t_main():
    a, _ = load("tab_main")
    b, _ = load(f"tab_main_{SECOND}")
    keep = [0, 2, 3, 4, 5, 6]  # Clean, m=8, Echo avg, Wrong, Slope, B-agree   (dropping m=2, Tok)
    groups = [("Single-pass readers", ["Vanilla RAG", "Dedup (5-gram)", "SemDedup (emb.)", "TrustRAG (filter)",
                                       "Astute RAG", "InstructRAG-ICL", "CAG (prompted)"]),
              ("Per-document claims + aggregation", ["Majority vote", "Near-dup + vote", "SemDedup + vote",
                                                     "ACCU (no copy det.)", "MADAM-RAG"]),
              ("Ours: dependence-aware vote", ["IndieVote-H", "IndieVote-L", "IndieVote-LLM", "IndieBayes-L"]),
              ("Ours: dependence-aware reader", ["IndieRAG-H", "IndieRAG-L", "IndieRAG-L+meta", "IndieRAG-LLM"]),
              ("Oracle dependence (upper bound)", ["IndieVote-oracle", "IndieRAG-oracle"])]
    table = []
    for g, meths in groups:
        table.append((g, None))
        for m in meths:
            table.append((m, [a[m][i] for i in keep] + [b[m][i] for i in keep] + [a[m][7]]))
    excl = {"IndieVote-oracle", "IndieRAG-oracle", "IndieRAG-L+meta"}  # L+meta uses simulated metadata
    marks = {}
    for off in (0, 6):
        marks.update(mark_best(table, [off + 0, off + 1, off + 2, off + 4, off + 5], True, excl))
        marks.update(mark_best(table, [off + 3], False, excl))
    body = r"""\begin{table*}[t]
\centering
\caption{\textbf{Illusory consensus on EchoBench (setting A).} Accuracy (\%) without copies (Clean), with $m{=}8$
copies of one wrong origin against $k{=}2$ independent correct pages, and averaged over all $m\!\geq\!1$, $k\!\in\!\{1,2,3\}$
(Echo, 12 conditions $\times$ 200 test questions). Wrong$\downarrow$: rate of outputting the injected wrong answer
(Echo conditions). Slope: $d\,\mathrm{acc}/d\log_2(m{+}1)$ at $k{=}2$ (0 = flat). B-agr.: agreement with the
independent majority in control setting B ($m{=}1$ or $3$ independent wrong pages vs.\ $k{=}2$; no copies). Tok: reader tokens per instance (Qwen2.5-7B).
\best{Best} and \scnd{second best} exclude oracle rows and L+meta (simulated page metadata); ``--'': method run on a reduced condition set.}
\label{tab:main}
\setlength{\tabcolsep}{3.6pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rrrrrr rrrrrr r}
\toprule
\hdr & \multicolumn{6}{c}{\textbf{Reader: Qwen2.5-7B-Instruct}} & \multicolumn{6}{c}{\textbf{Reader: Qwen3-8B}} & \\
\cmidrule(lr){2-7}\cmidrule(lr){8-13}
\hdr \textbf{Method} & Clean & $m{=}8$ & Echo & Wrong$\downarrow$ & Slope & B-agr. & Clean & $m{=}8$ & Echo & Wrong$\downarrow$ & Slope & B-agr. & Tok \\
\midrule
""" + render(table, marks, 14) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("main", body)


# ---------------------------------------------------------------------------------------------------- T2 significance
def _sig_cell(delta, ci, p_holm):
    d = fnum(delta)
    p = 0.0005 if p_holm.startswith("<") else fnum(p_holm)
    star = "^{\\ddagger}" if p < 0.01 else "^{\\dagger}" if p < 0.05 else ""
    s = nice(delta)
    if p < 0.05:
        return (f"\\sigp{{{s}}}" if d > 0 else f"\\sign{{{s}}}") + (f"$\\,{star}$" if star else "")
    return f"\\nsig{{{s}}}"


def t_sig():
    srcs = ["tab_significance", f"tab_significance_{SECOND}", "tab_significance_ramdocs",
            f"tab_significance_ramdocs_{SECOND}"]
    T = [load(s)[0] for s in srcs]
    meths = list(T[0].keys())
    lines = []
    for m in meths:
        cells = [_sig_cell(t[m][1], t[m][2], t[m][4]) if m in t else "--" for t in T]
        lines.append(f"{disp(m)} & " + " & ".join(cells) + " \\\\")
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{Paired gains of \sys{}-L over each baseline} ($\Delta$acc in points). EchoBench: setting A, $m\!\geq\!1$
(2{,}400 instances; 400 for MADAM-RAG); RAMDocs: 500 questions. Question-clustered bootstrap, Holm-corrected over the
12 baselines: $^{\ddagger}p_{\text{Holm}}{<}0.01$, $^{\dagger}p_{\text{Holm}}{<}0.05$; \sigp{green}/\sign{red} =
significant gain/loss, \nsig{gray} = not significant. CIs in Appendix~\ref{app:tables}.}
\label{tab:sig}
\setlength{\tabcolsep}{4pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rr rr}
\toprule
\hdr & \multicolumn{2}{c}{\textbf{EchoBench}} & \multicolumn{2}{c}{\textbf{RAMDocs}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\hdr \textbf{vs.\ baseline} & Qwen2.5 & Qwen3 & Qwen2.5 & Qwen3 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("sig", body)


def t_sig_full():
    out = []
    for src, title in [("tab_significance", "EchoBench, Qwen2.5-7B"), (f"tab_significance_{SECOND}", "EchoBench, Qwen3-8B"),
                       ("tab_significance_ramdocs", "RAMDocs, Qwen2.5-7B"),
                       (f"tab_significance_ramdocs_{SECOND}", "RAMDocs, Qwen3-8B")]:
        t, _ = load(src)
        out.append(f"\\grp{{7}}{{{title}}}")
        for m, c in t.items():
            out.append(f"{disp(m)} & {c[0]} & {nice(c[1])} & {nice(c[2])} & {c[3]} & {c[4]} & {c[5]} \\\\")
    body = r"""\begin{table}[h]
\centering
\caption{Paired significance of \sys{}-L vs.\ every baseline: $\Delta$acc, 95\% question-clustered bootstrap CI,
bootstrap $p$, Holm-corrected $p$, and McNemar $p$ (reference).}
\label{tab:sigfull}
\setlength{\tabcolsep}{3pt}
\scriptsize
\rowcolors{2}{white}{rowc}
\begin{tabular}{l r r c r r r}
\toprule
\hdr \textbf{vs.\ baseline} & $n$ & $\Delta$acc & 95\% CI & $p$ & $p_{\text{Holm}}$ & $p_{\text{McN}}$ \\
\midrule
""" + "\n".join(out) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("sig_full", body)


# ---------------------------------------------------------------------------------------------------- T3 controls
def t_controls(reader_suffix="", name="controls", label="tab:controls", reader="Qwen2.5-7B"):
    st, _ = load("tab_style" + reader_suffix)
    rn, _ = load("tab_ranked_noisy" + reader_suffix)
    sc, _ = load("tab_symC" + reader_suffix)
    mn, _ = load("tab_main" + reader_suffix)
    meths = ["Vanilla RAG", "SemDedup (emb.)", "Majority vote", "SemDedup + vote", "Astute RAG", "CAG (prompted)",
             "IndieVote-L", "IndieBayes-L", "IndieRAG-L"]
    table = []
    for m in meths:
        g = lambda t, i: t[m][i] if m in t else "--"
        table.append((m, [g(st, 7), g(rn, 2), g(rn, 5), g(mn, 6), g(sc, 2), g(sc, 1), g(sc, 4)]))
    marks = mark_best(table, list(range(7)), True)
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{Controls and stress tests} (""" + reader + r""", accuracy \% unless noted).
\emph{Style}: correct pages also LLM-rewritten ($m{=}8$). \emph{Rank.}: documents in dense-retriever order ($m{=}8$).
\emph{Noise}: 10 distractor pages ($m{=}4$). \emph{B-agr.}: follows the independent majority when wrong sources are
independent. \emph{Setting C}: copies of the \emph{true} page vs.\ $w$ independent wrong pages; C-ind.: agreement
with the independent majority at $m{=}4$, $w{=}2$. Questions per condition: style 183, ranked and noise 200, setting C
93, 98 and 68. Shaded: ours.}
\label{""" + label + r"""}
\setlength{\tabcolsep}{1.6pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rrrr rrr}
\toprule
\hdr & \multicolumn{4}{c}{\textbf{Echoed misinformation}} & \multicolumn{3}{c}{\textbf{Setting C (echoed truth)}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
\hdr \textbf{Method} & Style & Rank. & Noise & B-agr. & {\scriptsize $m4,w1$} & {\scriptsize $m2,w2$} & C-ind. \\
\midrule
""" + render(table, marks, 8) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(name, body)


# ---------------------------------------------------------------------------------------------------- T4 ablation
def t_ablation():
    a, _ = load("tab_ablation")
    b, _ = load(f"tab_ablation_{SECOND}")
    groups = [("Reader input (\\sys{}-L)", ["IndieRAG-L", "IndieRAG-L: table + all docs",
                                            "IndieRAG-L: grouped docs, no table"]),
              ("Dependence estimator (\\sys{})", ["IndieRAG-H", "IndieRAG-LLM", "IndieRAG-L+meta", "IndieRAG-oracle"]),
              ("Feature groups (IndieVote-L)", ["IndieVote-L", "IndieVote-L [-lexical]", "IndieVote-L [-semantic]",
                                                "IndieVote-L [-rare]", "IndieVote-L [-aigc]",
                                                "IndieVote-L [lexical_only]"])]
    lab = {"IndieRAG-L": "\\sys{}-L (full)", "IndieRAG-L: table + all docs": "\\quad table + all docs",
           "IndieRAG-L: grouped docs, no table": "\\quad grouped docs, no table", "IndieRAG-H": "\\quad H (rule)",
           "IndieRAG-LLM": "\\quad LLM pair judge", "IndieRAG-L+meta": "\\quad L + page metadata$^{*}$",
           "IndieRAG-oracle": "\\quad oracle origins", "IndieVote-L": "IndieVote-L (all features)",
           "IndieVote-L [-lexical]": "\\quad $-$lexical", "IndieVote-L [-semantic]": "\\quad $-$semantic",
           "IndieVote-L [-rare]": "\\quad $-$rare facts", "IndieVote-L [-aigc]": "\\quad $-$machine-text score",
           "IndieVote-L [lexical_only]": "\\quad lexical only"}
    lines = []
    for g, meths in groups:
        lines.append(f"\\grp{{7}}{{{g}}}")
        base = None
        for m in meths:
            ca, cb = a[m], b[m]
            row = [ca[0], ca[1], ca[2], cb[1], ca[3], ca[4]]
            if base is None:
                base = row
                cells = [f"\\textbf{{{nice(x)}}}" for x in row]
            else:
                cells = []
                for j, (x, x0) in enumerate(zip(row, base)):
                    v, v0 = fnum(x), fnum(x0)
                    if j in (1, 3) and v is not None and v0 is not None and abs(v - v0) >= 1.0:
                        cells.append(f"\\sigp{{{nice(x)}}}" if v > v0 else f"\\sign{{{nice(x)}}}")
                    else:
                        cells.append(nice(x))
            lines.append(f"{lab[m]} & " + " & ".join(cells) + " \\\\")
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{Ablations} (accuracy \%). Echo columns: \sign{red}/\sigp{green} = at least 1 point below/above the
first row of the block. ms: dependence estimation per instance (CPU). $^{*}$Page metadata in EchoBench is simulated
(see \S\ref{sec:bench}), so L+meta is optimistic.}
\label{tab:ablation}
\setlength{\tabcolsep}{2.6pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rrr r rr}
\toprule
\hdr & \multicolumn{3}{c}{\textbf{Qwen2.5-7B}} & \textbf{Qwen3} & & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}
\hdr \textbf{Variant} & Clean & Echo & B-agr. & Echo & ms & Tok \\
\midrule
""".replace("{l rrr r rr}", "{l rrr r rr}") + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("ablation", body)


# ---------------------------------------------------------------------------------------------------- T5 dependence
def t_dependence():
    d, _ = load("tab_dependence")
    rows = [("near-dup (5-gram $\\geq$0.5)", "5-gram near-dup ($\\geq$0.5)"), ("SemDedup (cos $\\geq$0.95)", "SemDeDup (cos $\\geq$0.95)"),
            ("LLM", "LLM pair judge (Qwen2.5-7B)"), ("H", "Ours: H (rule, no training)"),
            ("L[full]", "Ours: L (learned)"), ("L[full+meta]", "Ours: L + metadata$^{*}$")]
    abl = [("L[-lexical]", "\\quad L $-$lexical"), ("L[-semantic]", "\\quad L $-$semantic"),
           ("L[-rare]", "\\quad L $-$rare facts"), ("L[-aigc]", "\\quad L $-$machine-text score"),
           ("L[lexical_only]", "\\quad L lexical only"), ("L[n_train=25]", "\\quad L, 25 training questions"),
           ("L[n_train=100]", "\\quad L, 100 training questions")]
    cols = [0, 2, 3, 4, 6]  # AUC, AP, F1, ARI, ms/pool
    table = [("Estimators", None)] + [(lab, [d[k][c] for c in cols]) for k, lab in rows] + \
            [("Ablations and learning curve of L", None)] + [(lab, [d[k][c] for c in cols]) for k, lab in abl]
    marks = mark_best(table[:6], [0, 1, 2, 3], True)  # L+meta (row 6) uses simulated metadata
    lines = []
    for i, (lab, cells) in enumerate(table):
        if cells is None:
            lines.append(f"\\grp{{6}}{{{lab}}}")
            continue
        cs = []
        for c, x in enumerate(cells):
            m = marks.get((i, c)) if i < 7 else None
            cs.append(f"\\best{{{x}}}" if m == "best" else f"\\scnd{{{x}}}" if m == "scnd" else nice(x))
        pre = "\\ours " if lab.startswith("Ours") else ""
        lines.append(pre + lab + " & " + " & ".join(cs) + " \\\\")
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{Shared-origin detection} on the pools of the 456 EchoBench test questions (189{,}746 pairs, 59.5\%
dependent). AUC/AP/F1@0.5 in \%; ARI of the origin clustering at $\theta{=}0.5$; ms per pool. $^{*}$Simulated metadata.}
\label{tab:dep}
\setlength{\tabcolsep}{3.5pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rrrrr}
\toprule
\hdr \textbf{Estimator} & AUC & AP & F1 & ARI & ms \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("dependence", body)


# ---------------------------------------------------------------------------------------------------- T6 RAMDocs
def t_ramdocs():
    r, _ = load("tab_ramdocs")
    meths = ["Vanilla RAG", "Dedup (5-gram)", "SemDedup (emb.)", "Majority vote", "Astute RAG", "InstructRAG-ICL",
             "TrustRAG (filter)", "MADAM-RAG", "CAG (prompted)", "ACCU (no copy det.)", "IndieVote-L", "IndieBayes-L",
             "IndieRAG-H", "IndieRAG-L", "IndieRAG-LLM"]
    table = [(m, r[m]) for m in meths]
    marks = mark_best(table, [0, 2], True)
    marks.update(mark_best(table, [1, 3], False))
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{RAMDocs} (500 questions with ambiguity, misinformation and noise; no synthetic copies). Acc: all
valid answers and no misinformation answer; Wrong$\downarrow$: a misinformation answer is output.}
\label{tab:ramdocs}
\setlength{\tabcolsep}{4.5pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rr rr}
\toprule
\hdr & \multicolumn{2}{c}{\textbf{Qwen2.5-7B}} & \multicolumn{2}{c}{\textbf{Qwen3-8B}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\hdr \textbf{Method} & Acc & Wrong$\downarrow$ & Acc & Wrong$\downarrow$ \\
\midrule
""" + render(table, marks, 5) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("ramdocs", body)


# ---------------------------------------------------------------------------------------------------- T7 agents
def t_agents():
    t, _ = load("tab_agents")
    key = lambda s: s.replace("\\_", "_")
    t = {key(k): v for k, v in t.items()}
    rows = [("ReAct (Qwen2.5-7B)", "Qwen_Qwen2_5_7B_Instruct_react"),
            ("Search-R1 (RL, 7B)", "PeterJinGo_SearchR1_nq_hotpotqa_train_qwen2_5_7b_em_ppo_react"),
            ("IndieSearch ($\\tau{=}1.5$)", "Qwen_Qwen2_5_7B_Instruct_indie"),
            ("\\quad feedback only ($\\tau{=}0$)", "Qwen_Qwen2_5_7B_Instruct_indie_tau0"),
            ("\\quad stricter ($\\tau{=}2$)", "Qwen_Qwen2_5_7B_Instruct_indie_tau2")]
    table = []
    for lab, k in rows:
        e, h = t[f"echobench_{k}"], t[f"hotpot_{k}"]
        table.append((lab, [e[0], e[1], e[3], e[4], h[0], h[1], h[4]]))
    marks = mark_best(table, [0, 1, 4, 5], True)
    marks.update(mark_best(table, [2], False))
    lines = []
    for i, (lab, cells) in enumerate(table):
        cs = []
        for c, x in enumerate(cells):
            m = marks.get((i, c))
            cs.append(f"\\best{{{x}}}" if m == "best" else f"\\scnd{{{x}}}" if m == "scnd" else nice(x))
        lines.append(("\\ours " if "Indie" in lab or "tau" in lab else "") + lab + " & " + " & ".join(cs) + " \\\\")
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{Search agents} (300 questions each, local dense search engine). $m$: copies of a wrong origin
injected into the index. \#S: searches per question.}
\label{tab:agents}
\setlength{\tabcolsep}{2.8pt}
\small
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rrrr rrr}
\toprule
\hdr & \multicolumn{4}{c}{\textbf{EchoBench}} & \multicolumn{3}{c}{\textbf{HotpotQA-echo}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
\hdr \textbf{Agent} & $m{=}0$ & $m{=}4$ & Wr.$\downarrow$ & \#S & $m{=}0$ & $m{=}4$ & \#S \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("agents", body)


# ---------------------------------------------------------------------------------------------------- T8 web methods
def _delta_cell(s):
    m = re.match(r"([+-]?[\d.]+)\s*\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]", s)
    if not m:
        return "--"
    d, lo, hi = map(float, m.groups())
    txt = nice(m.group(1))
    if lo > 0:
        return f"\\sigp{{{txt}}}"
    if hi < 0:
        return f"\\sign{{{txt}}}"
    return f"\\nsig{{{txt}}}"


def t_web():
    a, _ = load("tab_web_methods")
    b, _ = load(f"tab_web_methods_{SECOND}")
    meths = ["Vanilla RAG", "Dedup (5-gram)", "SemDedup (emb.)", "Majority vote", "SemDedup + vote", "Astute RAG",
             "InstructRAG-ICL", "TrustRAG (filter)", "CAG (prompted)", "IndieVote-L", "IndieBayes-L", "IndieRAG-L"]
    lines = []
    for m in meths:
        ca, cb = a[m], b[m]
        cells = [nice(ca[0]), _delta_cell(ca[3]), _delta_cell(ca[4]),
                 nice(cb[0]), _delta_cell(cb[3]), _delta_cell(cb[4])]
        pre = "\\ours " if m.startswith("Indie") else ""
        lines.append(pre + disp(m) + " & " + " & ".join(cells) + " \\\\")
    va, vb = a["Vanilla RAG"][2], b["Vanilla RAG"][2]
    body = r"""\begin{table}[!t]
\centering
\caption{\textbf{Natural data: unmodified web search results} (TriviaQA, cached Bing pages; 1{,}000 questions, 297 of
them with pages supporting $\geq$2 answers = ``conf.''). Accuracy (\%) on all questions and paired difference to
vanilla RAG ($\Delta$, points) on all and on conflict questions (vanilla: """ + va + "/" + vb + r"""\%).
\sigp{Green}/\sign{red}: 95\% CI above/below 0; \nsig{gray}: CI contains 0.}
\label{tab:web}
\setlength{\tabcolsep}{3pt}
\small
\rowcolors{4}{white}{rowc}
\begin{tabular}{l rrr rrr}
\toprule
\hdr & \multicolumn{3}{c}{\textbf{Qwen2.5-7B}} & \multicolumn{3}{c}{\textbf{Qwen3-8B}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\hdr \textbf{Method} & Acc & $\Delta$ & $\Delta$conf. & Acc & $\Delta$ & $\Delta$conf. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("web", body)


# ---------------------------------------------------------------------------------------------------- appendix tables
def t_cost():
    c, _ = load("tab_cost")
    table = [(m, v) for m, v in c.items()]
    body = r"""\begin{table}[h]
\centering
\caption{Cost per EchoBench instance (Qwen2.5-7B). Claim extraction (one short call per document) and CAG's
per-document credibility ratings are counted as if run stand-alone. Seq.: sequential LLM rounds; dep.: dependence
estimation (ms, CPU); Echo: echo-avg accuracy.}
\label{tab:cost}
\setlength{\tabcolsep}{3.5pt}
\scriptsize
\rowcolors{2}{white}{rowc}
\begin{tabular}{l rrrrrr}
\toprule
\hdr \textbf{Method} & Calls & Prompt & Gen. & Seq. & dep. & Echo \\
\midrule
""" + render(table, mark_best(table, [5], True), 7) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("cost", body)


def t_copytypes():
    lines = []
    for src, title in [("tab_copytypes", "Qwen2.5-7B"), (f"tab_copytypes_{SECOND}", "Qwen3-8B")]:
        t, _ = load(src)
        table = [(m, v) for m, v in t.items()]
        lines.append(f"\\grp{{7}}{{{title}}}")
        lines.append(render(table, mark_best(table, list(range(6)), True, {"IndieVote-oracle"}), 7))
    body = r"""\begin{table}[h]
\centering
\caption{Accuracy (\%) with $m{=}4$ copies of a single type vs.\ $k{=}2$ independent correct pages.}
\label{tab:copytypes}
\setlength{\tabcolsep}{3pt}
\scriptsize
\rowcolors{2}{white}{rowc}
\begin{tabular}{l rrrrrr}
\toprule
\hdr \textbf{Method} & verb. & b.-tr. & para. & farm & summ. & mixed \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("copytypes", body)


def t_readers():
    t, _ = load("tab_readers")
    table = [(m, v) for m, v in t.items()]
    marks = mark_best(table, list(range(8)), True)
    body = r"""\begin{table}[h]
\centering
\caption{Reader family and scale (key methods; Granite and 32B were run on setting A and B only). Clean and echo-avg
accuracy (\%).}
\label{tab:readers}
\setlength{\tabcolsep}{2.6pt}
\scriptsize
\rowcolors{3}{white}{rowc}
\begin{tabular}{l rr rr rr rr}
\toprule
\hdr & \multicolumn{2}{c}{Qwen2.5-7B} & \multicolumn{2}{c}{Qwen3-8B} & \multicolumn{2}{c}{Granite-3.3-8B} & \multicolumn{2}{c}{Qwen2.5-32B} \\
\hdr \textbf{Method} & clean & echo & clean & echo & clean & echo & clean & echo \\
\midrule
""" + render(table, marks, 9) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("readers", body)


def t_judge():
    t, _ = load("tab_judge")
    lines = [f"{lab} & " + " & ".join(nice(c) for c in cells) + " \\\\" for lab, cells in t.items()]
    body = r"""\begin{table}[h]
\centering
\caption{Validity of the string-match metric: an LLM judge (Qwen3-8B) re-grades the same random instances for seven
methods (Qwen2.5-7B reader). Agreement and disagreement rates in \%; Kendall $\tau$ between the method rankings under
judge and metric.}
\label{tab:judge}
\setlength{\tabcolsep}{3pt}
\scriptsize
\rowcolors{2}{white}{rowc}
\begin{tabular}{l rrrrrr}
\toprule
\hdr \textbf{Data} & $n$ & agree & $\kappa$ & J+/M$-$ & J$-$/M+ & $\tau$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("judge", body)


if __name__ == "__main__":
    t_judge()
    t_main()
    t_sig()
    t_sig_full()
    t_controls()
    t_controls(f"_{SECOND}", "controls_qwen3", "tab:controls3", "Qwen3-8B")
    t_ablation()
    t_dependence()
    t_ramdocs()
    t_agents()
    t_web()
    t_cost()
    t_copytypes()
    t_readers()
