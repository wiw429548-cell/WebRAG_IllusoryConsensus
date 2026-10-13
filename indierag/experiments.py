"""Registry of every experiment of the IndieRAG paper -> scheduler jobs (see README.md)."""
import os

from .scheduler import Job, py
from .utils import DATA, ROOT, load_config, slug



def gated_accessible(model):
    """True if a gated HF model can actually be downloaded with the current HF_TOKEN (checked once per model)."""
    import urllib.request
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return False
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    req = urllib.request.Request(f"{base}/{model}/resolve/main/config.json", method="HEAD",
                                 headers={"Authorization": f"Bearer {tok}"})
    try:
        urllib.request.urlopen(req, timeout=30)
        return True
    except Exception:
        return False

def build_jobs(config_path=None, quick=False):
    cfg = load_config(config_path)
    c = ["--config", config_path] if config_path else []
    M = cfg["models"]
    R = ROOT / "results"
    J = []
    J.append(Job("eb_select", py("indierag.build_echobench", "--phase", "select", *c), DATA / "echobench/selected.jsonl", n_gpus=0))
    J.append(Job("web_prepare", py("indierag.measure_web", "--phase", "prepare", *c), DATA / "web/pages.jsonl", ["eb_select"], n_gpus=0))
    J.append(Job("eb_gen1", py("indierag.build_echobench", "--phase", "gen1", *c), DATA / "echobench/phase1.jsonl", ["eb_select"]))
    J.append(Job("eb_gen2", py("indierag.build_echobench", "--phase", "gen2", *c), DATA / "echobench/phase2.jsonl", ["eb_gen1"]))
    J.append(Job("eb_assemble", py("indierag.build_echobench", "--phase", "assemble", *c), DATA / "echobench/docs.jsonl", ["eb_gen2"], n_gpus=0))
    J.append(Job("hp_build", py("indierag.build_hotpot", "--phase", "build", *c), DATA / "hotpot_echo/docs.jsonl"))
    J.append(Job("agent_pool", py("indierag.build_hotpot", "--phase", "pool", *c), DATA / "agent/pool.jsonl",
                 ["eb_select", "hp_build"], n_gpus=0))
    J.append(Job("featurize", py("indierag.signals", *c), DATA / "features/docstore.json",
                 ["eb_assemble", "web_prepare", "agent_pool"]))
    J.append(Job("dep_train", py("indierag.dependence", *c), DATA / "features/dep_clf_info.json", ["featurize"], n_gpus=0))
    J.append(Job("simulate", py("indierag.simulate"), R / "theory/simulation.json", n_gpus=0))
    J.append(Job("annotation_export", py("indierag.annotate", "export", *c), R / "annotation/echo_copies.csv",
                 ["dep_train"], n_gpus=0))
    readers = list(M["readers"])
    extra = list(M.get("extra_readers") or [])
    gated = [m for m in readers + extra if m.startswith("meta-llama/") and not gated_accessible(m)]
    if gated:
        print(f"[experiments] no access to gated models {gated} (HF_TOKEN missing or access not granted) -> skipped", flush=True)
    readers = [m for m in readers if m not in gated]
    extra = [m for m in extra if m not in gated and m not in readers]
    if quick:
        readers, extra = readers[:1], []
    rc = cfg.get("rag") or {}
    red = ["--groups", rc.get("reduced_groups", "main,indep"),
           "--methods", rc.get("reduced_methods", "vanilla,majority,astute,indievote:L,indiebayes:L,indierag:L")]
    # RQ1/RQ2: all methods x readers on EchoBench (longest jobs first); RAMDocs generalisation
    for r in readers:
        J.append(Job(f"rag_{slug(r)}_echobench", py("indierag.run_rag", "--reader", r, "--dataset", "echobench", *c),
                     R / "rag" / slug(r) / "echobench" / "metrics.json", ["dep_train"]))
    for r in readers:
        J.append(Job(f"rag_{slug(r)}_ramdocs", py("indierag.run_rag", "--reader", r, "--dataset", "ramdocs", *c),
                     R / "rag" / slug(r) / "ramdocs" / "metrics.json", ["dep_train"]))
    # validity of the string-match accuracy: an LLM judge (different model) re-scores a sample of the primary
    # reader's predictions (EchoBench + RAMDocs)
    judge = M.get("judge") or "Qwen/Qwen3-8B"
    if readers and slug(judge) != slug(readers[0]):
        J.append(Job("judge_eval", py("indierag.judge_eval", "--reader", readers[0], "--judge", judge, *c),
                     R / "judge/metrics.json",
                     [f"rag_{slug(readers[0])}_echobench", f"rag_{slug(readers[0])}_ramdocs"]))
    if not quick:
        # other model family (reduced scope: main + indep groups, key methods)
        for r in extra:
            J.append(Job(f"rag_{slug(r)}_echobench", py("indierag.run_rag", "--reader", r, "--dataset", "echobench", *red, *c),
                         R / "rag" / slug(r) / "echobench" / "metrics.json", ["dep_train"]))
        # RQ3: dependence detection quality + ablations + learning curve (LLM judge with the first reader)
        J.append(Job("dep_eval", py("indierag.dep_eval", "--llm", readers[0], *c), R / "dependence/metrics.json", ["dep_train"]))
        # RQ4: search agents
        for bench in ["echobench", "hotpot"]:
            for pol, mode in [(M["agent_policy"], "react"), (M["agent_policy"], "indie"), (M["searchr1"], "react")]:
                J.append(Job(f"agent_{bench}_{slug(pol)}_{mode}",
                             py("indierag.agent", "--policy", pol, "--mode", mode, "--bench", bench, *c),
                             R / "agent" / f"{bench}_{slug(pol)}_{mode}" / "metrics.json", ["dep_train"]))
            # IndieSearch ablation: feedback only (tau=0, never forced to seek) and a stricter seek rule
            for tau in (cfg.get("agent") or {}).get("tau_ablation") or []:
                pol = M["agent_policy"]
                J.append(Job(f"agent_{bench}_{slug(pol)}_indie_tau{float(tau):g}",
                             py("indierag.agent", "--policy", pol, "--mode", "indie", "--bench", bench,
                                "--tau", float(tau), *c),
                             R / "agent" / f"{bench}_{slug(pol)}_indie_tau{float(tau):g}" / "metrics.json", ["dep_train"]))
        # RQ5: real web measurement (+ natural-data utility of every method; second reader writes to a sub-dir)
        J.append(Job("web_run", py("indierag.measure_web", "--phase", "run", "--reader", readers[0], *c),
                     R / "web_measure/metrics.json", ["dep_train"]))
        for r in readers[1:int((cfg.get("web_measure") or {}).get("n_readers", 1))]:
            J.append(Job(f"web_run_{slug(r)}", py("indierag.measure_web", "--phase", "run", "--reader", r,
                                                  "--tag", slug(r), *c),
                         R / "web_measure" / slug(r) / "metrics.json", ["dep_train"]))
        # scale analysis: a 32B (AWQ) reader, tensor-parallel over both GPUs (starts once both GPUs are free)
        J.append(Job("rag_large_echobench",
                     py("indierag.run_rag", "--reader", M["large_reader"], "--dataset", "echobench", "--tp", "all",
                        *red, *c),
                     R / "rag" / slug(M["large_reader"]) / "echobench" / "metrics.json", ["dep_train"], n_gpus=2))
    return J
