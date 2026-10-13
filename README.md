# Counting Sources, Not Pages

Code, data and results for the anonymous submission *Counting Sources, Not Pages: Dependence-Aware Evidence
Aggregation against Illusory Consensus in Web RAG*.

A RAG reader that sees seven pages giving one answer and three giving another usually goes with the seven. On the
Web those seven pages are often one article that was syndicated, mirrored, translated or rewritten by an LLM. We call
this *illusory consensus*. This repository contains:

- EchoBench, a benchmark built from real TriviaQA web search results, in which one wrong origin and up to eight
  copies of it compete with independent correct pages;
- IndieRAG, which estimates whether two pages share an origin and weighs each answer by its effective number of
  independent sources instead of its page count;
- the full experiment pipeline (12 baselines, 4 readers, dependence detection, search agents, RAMDocs, real web
  pages) and every result file behind the tables and figures of the paper.

The paper is in [`paper/main.pdf`](paper/main.pdf).

## Main results

EchoBench, setting A: accuracy averaged over 12 echo conditions (m ∈ {1, 2, 4, 8} copies of a wrong origin,
k ∈ {1, 2, 3} independent correct pages, 200 test questions each). Clean = no copies.

| Method | Qwen2.5-7B clean | Qwen2.5-7B echo | Qwen3-8B clean | Qwen3-8B echo |
|---|---|---|---|---|
| Vanilla RAG | 80.7 | 61.0 | 85.2 | 68.5 |
| Majority vote over pages | 75.7 | 35.0 | 78.2 | 40.6 |
| SemDeDup, then read | 80.5 | 66.2 | 85.2 | 73.0 |
| CAG (prompted), strongest baseline | 86.8 | 68.1 | 84.7 | 73.7 |
| IndieVote-L | 75.8 | 63.6 | 78.0 | 67.1 |
| IndieRAG-L | 80.3 | **78.3** | 84.7 | **79.5** |

IndieRAG-L is significantly better than each of the 12 baselines for both readers (question-clustered bootstrap,
Holm-corrected). The learned shared-origin detector reaches 99.6 pairwise AUC on the EchoBench test pools, against
53.8 for 5-gram near-duplicate detection and 60.0 for SemDeDup, and runs in about 24 ms per pool on a CPU.
On 6,397 real Bing result pages, it places 27.4% of the pages in an origin cluster with another page of the same
result list. All numbers are in [`results/summary.md`](results/summary.md).

The paper also reports the cases where the method does not help: when the echoed page is the correct one (setting C),
with Granite-3.3-8B as reader (use IndieVote there), for prompted search agents, and on the unmodified 2017 TriviaQA
web pages, where IndieRAG-L is about 2 points below vanilla RAG while IndieVote-L matches it.

## Repository layout

```
indierag/            all code (benchmark construction, methods, experiments, aggregation)
configs/default.yaml every setting used in the paper
run_all.py           runs the whole pipeline
data/
  echobench/         EchoBench (questions, documents, intermediate generation outputs)
  hotpot_echo/       HotpotQA-echo, 300 multi-hop questions for the search agents
  agent/pool.jsonl   passages of the local search index used by the agents
  web/pages.jsonl    1,000 TriviaQA questions with their unmodified cached Bing pages
  features/          trained shared-origin classifiers (dep_clf_*.pkl) and their training info
results/             every result file of the paper (metrics, per-instance predictions, tables, figures)
logs/                build logs with the EchoBench statistics quoted in the paper
paper/               LaTeX source, figures, tables and the scripts that produce them
scripts/unpack.py    decompresses the shipped .jsonl.gz files
```

Large `.jsonl` files are stored as `.jsonl.gz`. Run `python scripts/unpack.py` once after cloning.

## Reproducing the tables and figures (CPU only)

This uses the shipped results and needs no GPU or model download.

```bash
pip install numpy scipy matplotlib pyyaml
python scripts/unpack.py
python -m indierag.aggregate          # rewrites results/summary.md, results/latex/*.tex, results/figures/
sh paper/build.sh                     # paper figures and tables, number check, LaTeX build
```

`paper/build.sh` runs three scripts and then `latexmk`:

- `paper/scripts/make_figures.py` draws every figure from `results/**/metrics.json` and the per-instance predictions;
- `paper/scripts/make_tables.py` re-styles the tables written by `indierag.aggregate`, copying the numbers unchanged;
- `paper/scripts/check_numbers.py` checks that every number in the text of `paper/main.tex` appears in the result
  files or follows from them by the arithmetic listed in the script.

Running `indierag.aggregate` on the shipped files reproduces `results/summary.md` and all of `results/latex/`
exactly (on Windows the files are written with CRLF line endings).

## Running the full pipeline

Our runs used Linux, Python 3.12, PyTorch 2.13 with CUDA 13.0, vLLM, and two GPUs with 45 GiB of memory each.

```bash
pip install -r requirements.txt
python scripts/unpack.py
CUDA_VISIBLE_DEVICES=0,1 nohup python run_all.py > run.log 2>&1 &
```

`run_all.py` downloads TriviaQA rc.web, HotpotQA and RAMDocs into `data/raw/` (set `HF_ENDPOINT` to use a mirror),
then schedules 31 jobs over the visible GPUs. A job is skipped when its output file already exists, so with the
shipped `data/` and `results/` it only rebuilds the document features (`data/features/emb.npy` and friends are not
included because the embedding matrix is larger than GitHub's file limit) and then aggregates. To rerun an
experiment, delete its output (for example `results/rag/Qwen_Qwen2_5_7B_Instruct/echobench/`); to rebuild everything,
start from a copy without `data/` and `results/`. LLM calls are cached in `data/cache/*.sqlite`, and the RAG jobs
resume per method. Progress is written to `run.log` and `logs/_status.json`.

Options: `--quick` (one reader, main experiments only), `--config path`, `--skip_download`. Our estimate for the
complete grid is about 55 GPU-hours. Expensive methods run on subsets of the conditions listed under `rag.scope` in
the config; every average in the tables uses only conditions that all compared methods ran on.

Models used: Qwen2.5-7B-Instruct (benchmark builder, primary reader, agent policy), Phi-3.5-mini-instruct (second copy
generator), Qwen3-8B (second reader and LLM judge), Granite-3.3-8B-Instruct, Qwen2.5-32B-Instruct-AWQ,
the Search-R1 checkpoint `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`, bge-base-en-v1.5 (embeddings)
and Qwen2.5-1.5B (Fast-DetectGPT score). Llama-3.1-8B is listed in the config but was skipped because its weights
are gated; it runs automatically if `HF_TOKEN` has access.

## EchoBench

760 TriviaQA questions (304 train, 456 test) with 42,902 documents. `data/echobench/questions.jsonl` has one line per
question (`qid`, `question`, gold `aliases`, gold `value`, injected `wrong` answer, `split`). `docs.jsonl` has one
line per document:

| field | meaning |
|---|---|
| `role` | `correct` (real page), `origin` (real page rewritten to state the wrong answer), `copy` (copy of the origin), `indep_wrong` (wrong page written independently), `correct_para` (LLM paraphrase of a correct page, style-matched control), `correct_copy` (copy of a correct page, setting C), `distractor` (page from another question) |
| `origin` | origin id; two documents share an origin exactly when this field is equal |
| `copy_type` | `verbatim`, `backtrans`, `paraphrase`, `ai_farm` (content-farm rewrite) or `summary` |
| `gen` | `web` (real page), `builder` (Qwen2.5-7B), `builder2` (Phi-3.5-mini) or `rule` (verbatim copy) |
| `domain`, `date` | real domain for real pages; a fictitious site and a simulated date (days since 2010-01-01) for generated pages |

Counts: 760 origins, 25,142 copies (6,034 verbatim, 5,181 paraphrase, 5,061 back-translation, 4,447 summary, 4,419
content farm), 2,280 correct pages, 1,196 independent wrong pages, 1,980 style-matched and 2,424 copies of correct
pages, and 9,120 distractors. A copy is kept only if it still states the wrong answer and does not mention a gold
alias. The evaluation conditions (settings A, B, C, style, ranked, noisy) are built from these documents by
`indierag/conditions.py`.

The copies and wrong origins state false answers on purpose. They are meant for evaluating retrieval and
aggregation, not for training general-purpose models.

## Code map

| module | role |
|---|---|
| `build_echobench.py`, `build_hotpot.py` | benchmark construction |
| `conditions.py` | evaluation instances for every setting |
| `claims.py` | per-page answer extraction and answer grouping |
| `signals.py` | pair features (lexical, semantic, rare shared facts, machine-text score) |
| `dependence.py` | shared-origin estimators (H, L, L+meta, LLM, oracle, 5-gram, SemDeDup), clustering, ESS, IndieBayes |
| `methods.py` | IndieRAG, IndieVote and all baselines |
| `run_rag.py` | RAG experiments on EchoBench and RAMDocs |
| `dep_eval.py` | shared-origin detection, transfer, learning curve |
| `agent.py` | ReAct, Search-R1 and IndieSearch agents |
| `measure_web.py` | measurement and all methods on real web pages |
| `judge_eval.py` | LLM-judge check of the string-match metric |
| `simulate.py` | copy-corrected Condorcet simulation |
| `annotate.py` | export and scoring of optional human annotation sheets |
| `aggregate.py` | `results/summary.md`, LaTeX tables, figures |
| `experiments.py`, `scheduler.py`, `runtime.py`, `llm.py` | job registry, GPU scheduling, vLLM engine and caching |

Method names in the code map to the paper as follows: `vanilla`, `dedup` (near-duplicate removal), `semdedup`,
`majority`, `indievote:lex` (near-dup + vote), `indievote:cos` (SemDeDup + vote), `astute`, `instructrag`,
`trustrag`, `madam`, `cag`, `indiebayes:none` (ACCU), and ours: `indievote:{H,L,LLM,oracle}`, `indiebayes:L`,
`indierag:{H,L,L+meta,LLM,oracle}`.

## Data sources

EchoBench and `data/web/` are derived from the cached Bing search results distributed with
[TriviaQA](https://nlp.cs.washington.edu/triviaqa/) (rc.web validation split). HotpotQA-echo is derived from
[HotpotQA](https://hotpotqa.github.io/) (distractor dev). [RAMDocs](https://github.com/HanNight/RAMDocs) is
downloaded at run time and not redistributed here. Please follow the terms of these datasets when using the files
in `data/`.

## Citation

The citation will be added after the review period.
