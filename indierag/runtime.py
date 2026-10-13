"""Helpers that keep long multi-GPU runs from failing on a shared server.

  * vLLM's FlashInfer top-k/top-p sampler JIT-compiles CUDA code at warm-up and needs `nvcc`. setup_env() sets
    VLLM_USE_FLASHINFER_SAMPLER=0, which selects the PyTorch-native sampler (same outputs) and needs no compiler.
  * Upgrading torch while other jobs use it can crash them with SIGBUS. ensure_packages() installs only missing
    packages, never upgrades torch, and holds a file lock while pip runs.
  * Jobs started at the same time could land on the same GPU. Every job takes one flock per physical GPU, so
    concurrent launches queue instead of competing for memory.
  * build_vllm() clamps tensor parallelism to the visible GPUs, adapts gpu_memory_utilization to the free memory,
    and retries with eager mode or a shorter context before giving up. Every fallback is written to
    results/engine_log.jsonl.
"""
import json
import re
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows (local smoke tests): locks become no-ops
    fcntl = None

ENV_DEFAULTS = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",  # FlashInfer sampler needs nvcc for JIT -> use the PyTorch sampler
    # vLLM forks its engine-core process unless told otherwise; if anything in the job touched CUDA before the engine
    # was built (e.g. torch.cuda.is_available()), the forked child dies with "Cannot re-initialize CUDA in forked
    # subprocess" (observed in the rag_* jobs). spawn is immune to that; every entry point has a __main__ guard.
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "1",  # torch.cuda.is_available() via NVML: does not initialise CUDA
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",   # CUDA ids == nvidia-smi ids (used for locks / memory checks)
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUTF8": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def setup_env():
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)


def _user():
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return "user"


LOCK_DIR = Path(os.environ.get("WWW_LOCK_DIR") or os.path.join(tempfile.gettempdir(), f"www_gpu_locks_{_user()}"))


# ---------------------------------------------------------------------------------------------- GPU locks
def _lock_path(gpu):
    return LOCK_DIR / f"gpu{str(gpu).replace('/', '_')}.lock"


def try_lock_gpus(gpus, tag=""):
    """Non-blocking exclusive lock on every GPU in `gpus`. Returns a list of fds (possibly empty if locking is
    unavailable/disabled) or None if at least one GPU is held by another process."""
    if fcntl is None or not gpus or os.environ.get("WWW_NO_GPU_LOCK"):
        return []
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fds = []
    for g in gpus:
        fd = os.open(str(_lock_path(g)), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            release_gpus(fds)
            return None
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} cwd={os.getcwd()} job={tag} t={time.ctime()}\n".encode())
        except OSError:
            pass
        fds.append(fd)
    return fds


def lock_gpus_blocking(gpus, tag="", poll=30, log=print):
    """Wait until all `gpus` can be locked (for long-lived processes such as a model server)."""
    t0, warned = time.time(), False
    while True:
        fds = try_lock_gpus(gpus, tag)
        if fds is not None:
            if warned:
                log(f"[gpu-lock] acquired {gpus} after {(time.time() - t0) / 60:.1f} min")
            return fds
        if not warned:
            log(f"[gpu-lock] GPUs {gpus} are in use by another job ({lock_holders(gpus)}); waiting ...")
            warned = True
        time.sleep(poll)


def release_gpus(fds):
    for fd in fds or []:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except OSError:
            pass


def lock_holders(gpus):
    out = {}
    for g in gpus:
        try:
            out[g] = _lock_path(g).read_text().strip()
        except OSError:
            out[g] = "?"
    return out


# ---------------------------------------------------------------------------------------------- GPU memory
def gpu_memory():
    """{nvidia-smi index or UUID: (used_MiB, total_MiB)}; {} if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    res = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            try:
                res[p[0]] = res[p[1]] = (float(p[2]), float(p[3]))
            except ValueError:
                pass
    return res


def visible_physical_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    mem = gpu_memory()
    return sorted({k for k in mem if k.isdigit()}, key=int)


def free_fraction(gpus):
    """min over gpus of free/total memory, None if unknown."""
    mem = gpu_memory()
    fr = [(mem[g][1] - mem[g][0]) / mem[g][1] for g in gpus if g in mem and mem[g][1] > 0]
    return min(fr) if fr else None


def report_gpus(log=print):
    mem = gpu_memory()
    vis = visible_physical_gpus()
    log(f"[run_all] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} -> GPUs {vis}")
    for g in vis:
        if g in mem:
            used, tot = mem[g]
            held = try_lock_gpus([g], "probe")
            busy = "" if held is not None else f"  (currently used by another WWW job: {lock_holders([g])[g]})"
            release_gpus(held)
            log(f"[run_all]   gpu{g}: {used / 1024:.1f}/{tot / 1024:.1f} GiB used{busy}")
            if used > 2048 and held is not None:
                log(f"[run_all]   WARNING gpu{g} already has {used / 1024:.1f} GiB in use by another process; "
                    f"jobs will wait up to {os.environ.get('WWW_GPU_WAIT_MIN', '30')} min for it to free up")


# ---------------------------------------------------------------------------------------------- packages
def _importable(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def ensure_packages(reqs, log=print):
    """reqs: list of (import_name, pip_spec). Installs ONLY the missing ones (no --upgrade), keeping the installed
    torch pinned (unless vLLM itself must be installed), under a file lock. Returns still-missing mods."""
    missing = [(m, s) for m, s in reqs if not _importable(m)]
    if not missing:
        return []
    if os.environ.get("WWW_NO_PIP"):
        log(f"[run_all] missing {[m for m, _ in missing]} and WWW_NO_PIP is set -> not installing")
        return [m for m, _ in missing]
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lf = open(LOCK_DIR / "pip.lock", "a+")
    try:
        if fcntl is not None:
            log("[run_all] waiting for the pip lock ...")
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        missing = [(m, s) for m, s in missing if not _importable(m)]  # maybe installed meanwhile
        if missing:
            specs = [s for _, s in missing]
            cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--progress-bar", "off"]
            cons = None
            if not any(s.split(">")[0].split("=")[0].strip().lower() == "vllm" for s in specs):
                try:
                    from importlib.metadata import version
                    cons = Path(tempfile.gettempdir()) / f"www_constraints_{os.getpid()}.txt"
                    cons.write_text(f"torch=={version('torch')}\n")
                    cmd += ["-c", str(cons)]
                except Exception:
                    cons = None
            else:
                log("[run_all] WARNING: vLLM is missing and will be installed now; this may change the torch "
                    "version. Do not run other jobs while this happens (better: pip install vllm beforehand).")
            log(f"[run_all] pip installing missing packages: {specs}")
            for attempt in range(2):  # NFS 'Stale file handle' is transient
                rc = subprocess.call(cmd + specs)
                if rc == 0:
                    break
                time.sleep(10)
            if cons is not None:
                cons.unlink(missing_ok=True)
            import importlib
            importlib.invalidate_caches()
    finally:
        if fcntl is not None:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        lf.close()
    still = [m for m, _ in missing if not _importable(m)]
    if still:
        log(f"[run_all] WARNING: could not install {still}; jobs that need them will fail (see pip output above)")
    return still


# ---------------------------------------------------------------------------------------------- vLLM engine
def model_gpu_gb(model, tp=1):
    """rough GPU memory (GB per GPU) a model needs to start: weights (from the size in its name; AWQ/GPTQ ~0.55 B/param)
    + CUDA graphs / activations / a minimal KV cache. None if the size cannot be read from the name."""
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model.split("/")[-1].replace("-", " ").replace("_", " "))
    if not m:
        return None
    weights = float(m.group(1)) * (0.55 if re.search(r"awq|gptq|int4", model, re.I) else 2.0)
    return weights / max(1, tp) * 1.08 + 2.5


def wait_for_gpu_memory(gpus, need_gb, margin, log=print):
    """block until every GPU in `gpus` has `need_gb` free (+ margin) or WWW_GPU_MEM_WAIT_MIN (default 120) minutes
    passed (other users' processes on shared GPUs come and go). Returns (min free fraction, min total GB) now, or
    (None, None) if nvidia-smi is unavailable."""
    wait_s = 60 * float(os.environ.get("WWW_GPU_MEM_WAIT_MIN", "120"))
    t0, warned = time.time(), False
    while True:
        mem = gpu_memory()
        st = [(mem[g][1] - mem[g][0], mem[g][1]) for g in gpus if g in mem and mem[g][1] > 0]
        if not st:
            return None, None
        free = min(f for f, _ in st) / 1024
        tot = min(t for _, t in st) / 1024
        ok = need_gb is None or free - margin * tot >= need_gb
        if ok or time.time() - t0 > wait_s:
            if warned:
                log(f"[llm] GPU memory {'available' if ok else 'still short (wait limit reached, trying anyway)'}: "
                    f"{free:.1f} GB free after {(time.time() - t0) / 60:.0f} min")
            return min(f / t for f, t in st), tot
        if not warned:
            log(f"[llm] GPUs {gpus}: {free:.1f} GB free, ~{need_gb:.1f} GB needed (+headroom) -> other processes use "
                f"this GPU; waiting up to {wait_s / 60:.0f} min for memory (WWW_GPU_MEM_WAIT_MIN)")
            warned = True
        time.sleep(60)


def build_vllm(model, tp=1, max_model_len=8192, gpu_mem=0.85, log_file=None, **kw):
    """vLLM engine with: TP clamped to the visible GPUs; before EVERY attempt the free GPU memory is re-measured (other
    users' processes on shared GPUs change it), the job waits until the model fits, and gpu_memory_utilization is set
    to min(target, free - headroom); retry ladder: eager mode, then half context (>= 4096), each with a lower target.
    Returns (engine, info). Raises if every attempt fails."""
    setup_env()
    import gc
    import torch
    from vllm import LLM as VLLM

    n_vis = torch.cuda.device_count()
    tp_eff = max(1, min(int(tp), n_vis or 1))
    vis = visible_physical_gpus()[:tp_eff]
    # headroom below the currently free memory: another process on the same GPU (other users, a growing job) must not
    # push vLLM into OOM at or right after start-up (we observed 3.8-7 GB used by other processes on the same GPU)
    margin = float(os.environ.get("WWW_VLLM_HEADROOM", "0.08"))
    need = model_gpu_gb(model, tp_eff)
    target = float(gpu_mem) if tp_eff >= int(tp) else max(float(gpu_mem), 0.92)  # planned for more GPUs: all we can
    ladder = [dict(target=target, max_model_len=max_model_len),
              dict(target=target, max_model_len=max_model_len, enforce_eager=True)]
    if max_model_len > 4096:
        ladder.append(dict(target=target, max_model_len=max(4096, max_model_len // 2), enforce_eager=True))
    errors = []
    for i, step in enumerate(ladder):
        ff, tot = wait_for_gpu_memory(vis, need, margin)
        util = step["target"] if ff is None else max(0.3, min(step["target"], round(ff - margin, 3)))
        extra = dict(gpu_memory_utilization=util, max_model_len=step["max_model_len"])
        if step.get("enforce_eager"):
            extra["enforce_eager"] = True
        try:
            print(f"[llm] vLLM init attempt {i + 1}/{len(ladder)}: {model} tp={tp_eff} {extra} "
                  f"(free {'?' if ff is None else f'{ff * tot:.1f}/{tot:.1f} GB'}, need ~{need or '?'} GB)", flush=True)
            eng = VLLM(model=model, tensor_parallel_size=tp_eff, **{**kw, **extra})
            info = dict(model=model, tp_requested=int(tp), tp=tp_eff, attempt=i + 1,
                        max_model_len=extra["max_model_len"], gpu_mem=extra["gpu_memory_utilization"],
                        enforce_eager=bool(extra.get("enforce_eager")), errors=errors,
                        fallback=bool(i > 0 or tp_eff != int(tp) or extra["max_model_len"] != max_model_len),
                        argv=" ".join(sys.argv), time=time.ctime())
            print(f"[llm] engine ready: {json.dumps(info)}", flush=True)
            if log_file is not None:
                try:
                    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(info) + "\n")
                except OSError:
                    pass
            return eng, info
        except Exception as e:  # noqa: BLE001  (vLLM raises many different types)
            msg = f"{type(e).__name__}: {str(e)[:500]}"
            errors.append(msg)
            print(f"[llm] vLLM init attempt {i + 1} failed: {msg}", flush=True)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if i + 1 < len(ladder):
                time.sleep(60)  # let the failed engine process release its memory before re-measuring
    raise RuntimeError(f"vLLM could not start {model} (tp={tp_eff}) after {len(ladder)} attempts: {errors}")
