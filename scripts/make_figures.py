"""Figures of the README, drawn directly from the result files under results/.

Usage (from the repository root):  python scripts/make_figures.py
Writes assets/figures/*.png. Every plotted value is read from results/rag/*/echobench/metrics.json, the per-instance
predictions in results/rag/*/echobench/preds/ (.jsonl or .jsonl.gz), results/dependence/, results/web_measure/ and
results/theory/simulation.json.
"""
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = "Qwen_Qwen2_5_7B_Instruct"
SECOND = "Qwen_Qwen3_8B"
READERS = [("Qwen_Qwen2_5_7B_Instruct", "Qwen2.5-7B"), ("Qwen_Qwen3_8B", "Qwen3-8B"),
           ("ibm_granite_granite_3_3_8b_instruct", "Granite-3.3-8B"), ("Qwen_Qwen2_5_32B_Instruct_AWQ", "Qwen2.5-32B")]
MS, KS = [0, 1, 2, 4, 8], [1, 2, 3]

# ------------------------------------------------------------------ style: one palette for the whole paper
C = {
    "vanilla": "#3b3b3b", "majority": "#9a9a9a", "semdedup": "#e69f00", "dedup": "#f0c27b", "astute": "#8e6bb0",
    "cag": "#b5835a", "instructrag": "#6aa84f", "indievote:L": "#4a90c8", "indierag:L": "#c8102e",
    "indierag:oracle": "#c8102e", "indievote:lex": "#c9c9c9", "indievote:cos": "#d9b36c",
}
LABEL = {
    "vanilla": "Vanilla RAG", "majority": "Majority vote", "semdedup": "SemDeDup", "dedup": "Near-dup removal",
    "astute": "Astute RAG", "cag": "CAG (prompted)", "instructrag": "InstructRAG-ICL", "indievote:L": "IndieVote-L",
    "indierag:L": "IndieRAG-L", "indierag:oracle": "IndieRAG-oracle", "indievote:lex": "Near-dup + vote",
    "indievote:cos": "SemDeDup + vote", "madam": "MADAM-RAG", "trustrag": "TrustRAG", "indiebayes:none": "ACCU",
    "indievote:H": "IndieVote-H", "indiebayes:L": "IndieBayes-L", "indierag:H": "IndieRAG-H",
    "indierag:LLM": "IndieRAG-LLM", "indievote:LLM": "IndieVote-LLM", "indierag:L+meta": "IndieRAG-L+meta",
}
MARK = {"vanilla": "o", "majority": "v", "semdedup": "s", "astute": "D", "cag": "P", "indievote:L": "^",
        "indierag:L": "o", "indierag:oracle": "o", "instructrag": "X"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4,
    "lines.markersize": 3.6, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def metrics(reader):
    return jl(RES / "rag" / reader / "echobench" / "metrics.json")["metrics"]


def acc(mt, meth, cond, key="acc"):
    v = (mt.get(meth) or {}).get(cond)
    return None if v is None or v.get(key) is None else 100 * v[key]


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=250)
    plt.close(fig)
    print("wrote", stem)


def curve(ax, mt, meth, conds, xs, **kw):
    ys = [acc(mt, meth, c) for c in conds]
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return
    style = dict(color=C.get(meth, "k"), marker=MARK.get(meth, "o"), label=LABEL.get(meth, meth), zorder=3)
    if meth == "indierag:L":
        style.update(lw=2.1, ms=4.2, zorder=5)
    if "oracle" in meth:
        style.update(ls=(0, (3, 2)), lw=1.0, marker=None, alpha=0.7)
    if meth == "majority":
        style.update(ls=(0, (1, 1.2)))
    style.update(kw)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], **style)


def band(ax, reader, meth, conds, xs, n_boot=2000, seed=0):
    """95% bootstrap band from the per-question predictions (one instance per question and condition)"""
    path = RES / "rag" / reader / "echobench" / "preds" / (meth.replace(":", "_") + ".jsonl")
    by = {}
    lines = open(path, encoding="utf-8") if path.exists() else gzip.open(str(path) + ".gz", "rt", encoding="utf-8")
    for line in lines:
        r = json.loads(line)
        by.setdefault(r["cond"], []).append(r["acc"])
    rng = np.random.RandomState(seed)
    lo, hi = [], []
    for c in conds:
        a = np.array(by[c], float)
        b = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(1)
        lo.append(100 * np.quantile(b, .025))
        hi.append(100 * np.quantile(b, .975))
    ax.fill_between(xs, lo, hi, color=C[meth], alpha=0.13, lw=0, zorder=1)


# ------------------------------------------------------------------ Fig: susceptibility curves (RQ1/RQ2)
def fig_susceptibility():
    mt = metrics(PRIMARY)
    meths = ["vanilla", "majority", "semdedup", "astute", "cag", "indievote:L", "indierag:L", "indierag:oracle"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
    xs = np.log2(np.array(MS) + 1)
    for ax, k in zip(axes[:3], KS):
        for meth in ("vanilla", "indierag:L"):
            band(ax, PRIMARY, meth, [f"A_mixed_m{m}_k{k}" for m in MS], xs)
        for meth in meths:
            curve(ax, mt, meth, [f"A_mixed_m{m}_k{k}" for m in MS], xs)
        ax.set_title(f"({'abc'[k - 1]}) setting A, $k{{=}}{k}$", pad=3)
    ax = axes[3]
    smeths = ["vanilla", "majority", "semdedup", "astute", "cag", "indievote:L", "indierag:L"]
    sm = [0, 2, 4, 8]
    for meth in ("vanilla", "indierag:L"):
        band(ax, PRIMARY, meth, [f"S_mixed_m{m}_k2" for m in sm], np.log2(np.array(sm) + 1))
    for meth in smeths:
        curve(ax, mt, meth, [f"S_mixed_m{m}_k2" for m in sm], np.log2(np.array(sm) + 1))
    ax.set_title("(d) style-matched, $k{=}2$", pad=3)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in MS])
        ax.set_xlabel("copies $m$ of the wrong origin")
        ax.set_ylim(15, 97)
    axes[0].set_ylabel("accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.10), handlelength=2.2, columnspacing=1.1)
    save(fig, "susceptibility")


# ------------------------------------------------------------------ Fig: readers (family / scale)
def fig_readers():
    meths = ["vanilla", "majority", "astute", "indievote:L", "indierag:L"]
    echo = [f"A_mixed_m{m}_k{k}" for m in MS if m > 0 for k in KS]
    clean = [f"A_mixed_m0_k{k}" for k in KS]
    fig, ax = plt.subplots(figsize=(3.35, 1.75))
    w = 0.16
    for j, meth in enumerate(meths):
        for i, (r, _) in enumerate(READERS):
            mt = metrics(r)
            e = [acc(mt, meth, c) for c in echo]
            cl = [acc(mt, meth, c) for c in clean]
            if any(v is None for v in e):
                continue
            x = i + (j - 2) * w
            ax.bar(x, np.mean(e), w * 0.92, color=C[meth], edgecolor="white", lw=0.3,
                   label=LABEL[meth] if i == 0 else None, zorder=3)
            ax.plot([x - w * 0.42, x + w * 0.42], [np.mean(cl)] * 2, color="black", lw=0.9, zorder=4,
                    label="clean ($m{=}0$)" if (i == 0 and j == 0) else None)
    ax.set_xticks(range(len(READERS)))
    ax.set_xticklabels([n.replace("-", "\n", 1) for _, n in READERS])
    ax.set_ylabel("echo-avg accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.33), handlelength=1.2, columnspacing=0.8)
    save(fig, "readers")


# ------------------------------------------------------------------ Fig: copy types (heatmaps)
def fig_copytypes():
    types =["verbatim", "backtrans", "paraphrase", "ai_farm", "summary"]
    tlab = ["verbatim", "back-transl.", "paraphrase", "content farm", "summary"]
    # (a) recall of origin-copy pairs per estimator -- taken from the generated LaTeX table (same source numbers)
    rows = {}
    for line in open(RES / "latex" / "tab_dep_types.tex", encoding="utf-8"):
        if "&" in line and "Method" not in line:
            cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            rows[cells[0]] = [float(c) for c in cells[1:]]
    est = [("near-dup (5-gram $\\geq$0.5)", "5-gram near-dup"), ("SemDedup (cos $\\geq$0.95)", "SemDeDup"),
           ("LLM", "LLM pair judge"), ("H", "Ours: H (rule)"), ("L[full]", "Ours: L (learned)")]
    A = np.array([rows[k][:5] for k, _ in est])
    fpr = [rows[k][5] for k, _ in est]
    # (b) end-to-end accuracy per copy type (m=4, k=2)
    mt = metrics(PRIMARY)
    meths = ["vanilla", "dedup", "semdedup", "majority", "indievote:lex", "indievote:cos", "indievote:L", "indierag:L"]
    B = np.array([[acc(mt, m, f"A_{t}_m4_k2") for t in types] for m in meths], float)

    cmap = LinearSegmentedColormap.from_list("gb", ["#f7f7f7", "#c6dbef", "#4a90c8", "#08306b"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.55})
    for ax, M, ylabels, title, unit in [
            (axes[0], A, [n for _, n in est], "(a) recall of origin–copy pairs (%)", ""),
            (axes[1], B, [LABEL[m] for m in meths], "(b) QA accuracy with 4 copies of one type, $k{=}2$ (%)", "")]:
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(tlab, rotation=25, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.0f}" if v >= 99.95 else f"{v:.1f}", ha="center", va="center",
                        fontsize=6.3, color="white" if v > 62 else "#222222")
        ax.set_title(title, pad=4)
    for i, f in enumerate(fpr):
        axes[0].text(len(types) - 0.35, i, f"{f:.1f}", ha="left", va="center", fontsize=6.3, color="#7a0019")
    axes[0].text(len(types) - 0.35, len(est) - 0.3, "FPR on\nindep.\npairs", ha="left", va="top", fontsize=5.8,
                 color="#7a0019", fontweight="bold")
    for lab in axes[0].get_yticklabels()[3:]:
        lab.set_fontweight("bold")
    for lab in axes[1].get_yticklabels()[6:]:
        lab.set_fontweight("bold")
    save(fig, "copytypes")


# ------------------------------------------------------------------ Fig: accuracy vs. cost
def fig_cost():
    mt = metrics(PRIMARY)
    echo = [f"A_mixed_m{m}_k{k}" for m in MS if m > 0 for k in KS]
    meths = ["vanilla", "dedup", "semdedup", "majority", "indievote:lex", "indievote:cos", "astute", "instructrag",
             "trustrag", "cag", "indievote:L", "indievote:LLM", "indierag:H", "indierag:L", "indierag:LLM"]
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    pts = []
    for meth in meths:
        e = [acc(mt, meth, c) for c in echo]
        cst = mt[meth].get("_cost_per_instance")
        if any(v is None for v in e) or not cst:
            continue
        tok = (cst["prompt_tokens"] + cst["completion_tokens"]) / 1000
        ours = meth.startswith("indie")
        col = C.get(meth, "#4a90c8" if meth.startswith("indievote") else ("#c8102e" if ours else "#7f7f7f"))
        ax.scatter(tok, np.mean(e), s=34 if meth == "indierag:L" else 20, color=col, zorder=4,
                   edgecolor="white", lw=0.5, marker="*" if meth == "indierag:L" else "o")
        pts.append((tok, np.mean(e), LABEL.get(meth, meth)))
    off = {"IndieRAG-L": (5, -2), "IndieRAG-H": (-30, 5), "IndieRAG-LLM": (-50, -9), "IndieVote-LLM": (-60, -2),
           "Majority vote": (5, -2), "Near-dup + vote": (5, -2), "SemDedup + vote": (5, -2), "Astute RAG": (3, 4),
           "CAG (prompted)": (4, -7), "InstructRAG-ICL": (-30, 7), "Vanilla RAG": (3, -8), "SemDedup": (-33, 2),
           "Near-dup removal": (-60, -1), "TrustRAG": (-42, 0), "IndieVote-L": (5, -2)}
    crowd = {"Vanilla RAG", "Near-dup removal", "SemDedup", "TrustRAG", "InstructRAG-ICL", "Astute RAG"}
    cp = [(x, y) for x, y, n in pts if n in crowd]
    if cp:
        from matplotlib.patches import Ellipse
        cx, cy = np.exp(np.mean(np.log([p[0] for p in cp]))), np.mean([p[1] for p in cp])
        ax.add_patch(Ellipse((cx, cy), 0.95, 10.5, transform=ax.transData, fill=False, ec="#999999", lw=0.6,
                             ls=(0, (2, 1.5))))
        ax.annotate("single-call baselines:\nVanilla, Near-dup, SemDeDup,\nTrustRAG, InstructRAG-ICL, Astute",
                    (cx, cy - 5.2), xytext=(1.95, 57.5), textcoords="data", fontsize=5.8, color="#444444", va="top",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
    for x, y, n in pts:
        if n in crowd:
            continue
        dx, dy = off.get(n, (4, 2))
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6,
                    color="#c8102e" if n.startswith("IndieRAG") else "#333333",
                    fontweight="bold" if n == "IndieRAG-L" else "normal")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.85, 15)
    ax.set_xticks([2, 3, 5, 10])
    ax.set_xticklabels(["2k", "3k", "5k", "10k"])
    ax.set_xlabel("reader tokens per instance (log)")
    ax.set_ylabel("echo-avg accuracy (%)")
    ax.set_ylim(30, 84)
    save(fig, "cost")


# ------------------------------------------------------------------ Fig: theory + web measurement
def fig_theory_web():
    t = jl(RES / "theory" / "simulation.json")
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1]})
    ax = axes[0]
    dd = t["p=0.65"]
    ms = sorted(dd, key=int)
    recs = sorted(next(iter(dd.values())), key=float)
    cm = LinearSegmentedColormap.from_list("r", ["#bdbdbd", "#c8102e"])
    xs = np.log2(np.array([int(m) for m in ms]) + 1)
    for i, rc in enumerate(recs):
        ax.plot(xs, [100 * dd[m][rc] for m in ms], color=cm(i / (len(recs) - 1)), marker="o", ms=2.6, lw=1.2,
                label=f"{float(rc):.2f}")
    ax.set_xticks(xs[::2])
    ax.set_xticklabels([ms[i] for i in range(0, len(ms), 2)])
    ax.set_xlabel("copies $m$ of one origin")
    ax.set_ylabel("majority accuracy (%)")
    ax.set_title("(a) Condorcet model, $p{=}0.65$", pad=3)
    ax.legend(title="detector recall", title_fontsize=6, fontsize=5.8, loc="lower left", handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.2)
    ax = axes[1]
    w = jl(RES / "web_measure" / "metrics.json")
    ci = w["ci95"]
    items = [("frac_pages_lexical_duplicate", "lexical\nnear-dup", "#9a9a9a"),
             ("frac_pages_dependent_cluster", "shared\norigin (L)", "#c8102e")]
    for i, (k, n, col) in enumerate(items):
        v = 100 * w[k]
        ax.bar(i, v, 0.6, color=col, zorder=3)
        ax.errorbar(i, v, yerr=[[v - 100 * ci[k][0]], [100 * ci[k][1] - v]], color="black", capsize=2, lw=0.8,
                    zorder=4)
        ax.text(i, 100 * ci[k][1] + 1.2, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for _, n, _ in items])
    ax.set_ylabel("% of real web pages")
    ax.set_ylim(0, 35)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) TriviaQA Bing pages", pad=3)
    save(fig, "theory_web")


# ------------------------------------------------------------------ Fig: natural-data forest plot (two readers)
def fig_web_forest():
    meths = ["dedup", "semdedup", "majority", "astute", "instructrag", "cag", "indievote:L", "indierag:L"]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (path, title) in zip(axes, [(RES / "web_measure" / "metrics.json", "Qwen2.5-7B"),
                                        (RES / "web_measure" / SECOND / "metrics.json", "Qwen3-8B")]):
        w = jl(path)
        for i, meth in enumerate(meths):
            for sub, dy, col_alpha in [("all", -0.14, 1.0), ("conflict", 0.14, 0.45)]:
                p = w["paired"].get(f"{meth}|vanilla", {}).get(sub)
                if not p:
                    continue
                col = C.get(meth, "#7f7f7f")
                y = i + dy
                ax.plot([100 * p["lo"], 100 * p["hi"]], [y, y], color=col, alpha=col_alpha, lw=1.3)
                ax.plot(100 * p["delta"], y, "o" if sub == "all" else "s", color=col, alpha=col_alpha, ms=3.0,
                        mec="white", mew=0.3)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, pad=3)
        ax.set_xlabel("$\\Delta$acc vs. vanilla (pp)")
        ax.set_xlim(-31, 11)
        ax.set_xticks([-20, -10, 0, 10])
        ax.set_xticklabels(["$-$20", "$-$10", "0", "+10"])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(meths)))
    axes[0].set_yticklabels([LABEL[m] for m in meths])
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], color="#555", marker="o", ms=3, lw=1.2),
                    Line2D([], [], color="#555", alpha=0.45, marker="s", ms=3, lw=1.2)],
                   ["all 1000 q.", "297 conflict q."], loc="upper left", fontsize=6, handlelength=1.4)
    save(fig, "web_forest")


if __name__ == "__main__":
    fig_susceptibility()
    fig_readers()
    fig_copytypes()
    fig_cost()
    fig_theory_web()
    fig_web_forest()
