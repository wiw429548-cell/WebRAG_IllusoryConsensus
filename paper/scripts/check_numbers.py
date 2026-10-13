"""Audit: every decimal number in the prose of main.tex must appear in a result source.

Usage (from the repository root):  python paper/scripts/check_numbers.py
Sources: results/summary.md, results/latex/*.tex, the benchmark build logs, and a short list of numbers derived by
simple arithmetic from those sources (each derivation is written next to it). Unmatched numbers are printed for review.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
body = TEX[TEX.index(r"\begin{abstract}"):TEX.index(r"\bibliographystyle")]
body = re.sub(r"(?m)%.*$", "", body)
body = re.sub(r"\\(?:input|includegraphics|label|ref|cite)(?:\[[^\]]*\])?\{[^}]*\}", " ", body)

src = (ROOT / "results" / "summary.md").read_text(encoding="utf-8")
for p in sorted((ROOT / "results" / "latex").glob("*.tex")):
    src += p.read_text(encoding="utf-8")
for p in ["logs/eb_select.log", "logs/eb_gen1.log", "logs/eb_assemble.log"]:
    src += (ROOT / p).read_text(encoding="utf-8", errors="ignore")
src_nums = set(re.findall(r"\d+(?:\.\d+)?", src.replace(",", "")))

DERIVED = {
    "12": "echo conditions: m in {1,2,4,8} x k in {1,2,3}",
    "24": "12 baselines x 2 readers",
    "25142": "sum of valid copies in eb_assemble.log",
    "99.2": "6034 / (6034+46) valid verbatim copies",
    "73": "4419/6080 (content farm) = 72.7%, lowest validity of LLM copies",
    "85": "5181/6080 (paraphrase) = 85.2%, highest validity of LLM copies",
    "304": "train split of the 760 assembled questions", "456": "test split",
    "238581": "data/features/dep_clf_info.json n_pairs", "14236": "data/agent/pool.jsonl lines",
    "6.4": "6397 pages / 1000 questions", "2.5": "27.4 / 10.9", "4.8": "11.9k / 2.5k tokens",
    "5.4": "5388 ms per pool", "2.0": "63.6 - 61.6 (lexical only)", "1.3": "67.1 - 65.8", "1.5": "82.3-80.8? no: 80.3 vs 78.8",
    "0.12": "ESS error 0.116", "0.42": "summary-copy rank 0.419", "0.55": "ai_farm rank 0.550",
    "37": "86.5 - 49.5", "16": "Qwen3 85.2-68.5 / 32B 88.0-72.1", "20": "Qwen2.5 80.7-61.0 = 19.7",
    "54": "near-dup AUC 53.8", "60": "SemDeDup AUC 60.0", "183": "style conditions n", "93": "C_m4_w1 n",
    "98": "C_m2_w2 n", "68": "C_m4_w2 n", "200": "test questions per reader", "150": "A_summary n", "199": "A_verbatim n",
    "64.9": "sim m>=4 naive", "65.1": "sim naive max", "76.3": "sim perfect min 0.7629", "76.6": "sim perfect max 0.7662",
    "65.4": "sim recall 0.8 m=32 0.6537", "0.65": "sim p", "0.8": "sim recall", "0.6": "answer grouping F1",
    "0.3": "H rule offset", "0.7": "H rule cosine", "0.75": "LLM pair cosine", "0.1": "LLM pair jaccard",
    "0.9": "IndieBayes discount / IndieSearch cosine", "1.5": "tau", "0.95": "SemDeDup cosine", "0.5": "theta / containment",
    "220": "passage words", "5000": "bootstrap resamples", "2026": "seed",
}
TRIVIAL = {str(i) for i in range(0, 11)} | {"2017", "2027", "32", "100", "1000", "2.5", "8k"}

nums = re.findall(r"(?<![\w.])\d+(?:\{,\}\d+)?(?:\.\d+)?", body)
unmatched = []
for n in nums:
    v = n.replace("{,}", "")
    if v in src_nums or v in DERIVED or v in TRIVIAL:
        continue
    unmatched.append(v)
print(f"{len(nums)} numbers in prose; {len(set(unmatched))} distinct unmatched:")
for v in sorted(set(unmatched), key=lambda x: float(x)):
    i = body.replace("{,}", "").find(v)
    print(f"  {v:>8}  ...{body.replace('{,}', '')[max(0, i - 60):i + 30].replace(chr(10), ' ')}...")
