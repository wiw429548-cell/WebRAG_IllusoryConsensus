"""Dependency-aware multi-GPU job scheduler (resumable via done-files).

Job(n_gpus=k) occupies k GPUs (CUDA_VISIBLE_DEVICES is set to exactly those); n_gpus=0 -> CPU job.
(Legacy: Job(gpu=False) == n_gpus=0, Job(gpu=True) == n_gpus=1.)
* A failed job is retried once (WWW_JOB_RETRIES, default 1; transient errors: OOM caused by other users, NFS
  hiccups, ...). If it still fails, its dependents are skipped and everything else continues.
* GPUs are locked per physical GPU (see runtime.py), so several run_all.py can be started at the same time on the
  same GPUs: jobs just wait for a free GPU.
* A GPU that is busy with a process outside this scheduler (> WWW_GPU_BUSY_MIB MiB used) is not used for up to
  WWW_GPU_WAIT_MIN minutes; after that jobs start anyway (vLLM then adapts to the free memory).
* logs/_status.json is rewritten every poll (what is running / done / failed).
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import runtime


class Job:
    def __init__(self, name, cmd, done_file, deps=(), n_gpus=1, env=None, always=False, gpu=None):
        """always=True: cheap analysis job that is re-run on every invocation (so that results added later, e.g. by a
        full run after a --quick run, are always included)"""
        if gpu is not None:
            n_gpus = 1 if gpu else 0
        self.name, self.cmd, self.done_file = name, [str(c) for c in cmd], Path(done_file)
        self.deps, self.n_gpus, self.env = list(deps), n_gpus, env or {}
        self.status = "done" if (self.done_file.exists() and not always) else "pending"
        self.proc = None
        self.slots = []
        self.locks = []
        self.attempts = 0
        self.t0 = None


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    if env is not None:  # CUDA_VISIBLE_DEVICES="" -> no GPU
        return []
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _write_status(log_dir, jobs):
    st = {s: [j.name for j in jobs if j.status == s] for s in ("running", "pending", "done", "failed", "skipped")}
    st["running"] = [f"{j.name} gpus={j.slots} {(time.time() - j.t0) / 60:.0f}min" for j in jobs if j.status == "running"]
    st["time"] = time.ctime()
    try:
        (Path(log_dir) / "_status.json").write_text(json.dumps(st, indent=1))
    except OSError:
        pass


def run_jobs(jobs, log_dir, cwd, gpus=None, slots=None, max_cpu_jobs=2, poll=10):
    runtime.setup_env()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = gpus if gpus is not None else (slots if slots is not None else visible_gpus())
    gpus = [g for g in gpus if g != "cpu"]
    free = list(gpus)
    max_retries = int(os.environ.get("WWW_JOB_RETRIES", "1"))
    busy_mib = float(os.environ.get("WWW_GPU_BUSY_MIB", "2048"))
    wait_s = 60 * float(os.environ.get("WWW_GPU_WAIT_MIN", "30"))
    by_name = {j.name: j for j in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("duplicate job names")
    for j in jobs:
        for d in j.deps:
            if d not in by_name:
                raise KeyError(f"{j.name} depends on unknown job {d}")
        if j.n_gpus > len(gpus):
            print(f"[scheduler] {j.name} wants {j.n_gpus} GPUs but only {len(gpus)} visible -> using {len(gpus)} "
                  f"(the LLM code clamps tensor parallelism accordingly)", flush=True)
            j.n_gpus = len(gpus)
    print(f"[scheduler] {len(jobs)} jobs ({sum(j.status == 'done' for j in jobs)} already done); gpus={gpus}; "
          f"gpu locks in {runtime.LOCK_DIR}", flush=True)
    cpu_running = 0
    foreign_since = {}   # gpu -> first time we saw it busy with a process outside this scheduler
    waiting_lock_msg = set()

    def on_term(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, AttributeError):
        pass

    def usable_gpus(n, mem):
        """pick n GPUs from `free` that are not locked by another WWW job and not busy with a foreign process."""
        cand = []
        now = time.time()
        for g in free:
            if g in mem and mem[g][0] > busy_mib:
                foreign_since.setdefault(g, now)
                if now - foreign_since[g] < wait_s:
                    continue
            else:
                foreign_since.pop(g, None)
            cand.append(g)
        cand.sort(key=lambda g: mem.get(g, (0, 1))[0])
        return cand[:n] if len(cand) >= n else None

    try:
        while True:
            for j in jobs:
                if j.status == "pending" and any(by_name[d].status in ("failed", "skipped") for d in j.deps):
                    j.status = "skipped"
                    print(f"[scheduler] SKIP  {j.name} (dependency failed)", flush=True)
            mem = None
            for j in jobs:
                if j.status != "pending" or not all(by_name[d].status == "done" for d in j.deps):
                    continue
                locks = []
                if j.n_gpus == 0:
                    if cpu_running >= max_cpu_jobs:
                        continue
                    slots = []
                else:
                    if len(free) < j.n_gpus:
                        continue
                    if mem is None:
                        mem = runtime.gpu_memory()
                    slots = usable_gpus(j.n_gpus, mem)
                    if slots is None:
                        continue
                    locks = runtime.try_lock_gpus(slots, j.name)
                    if locks is None:
                        if j.name not in waiting_lock_msg:
                            print(f"[scheduler] WAIT  {j.name}: GPUs {slots} in use by another job "
                                  f"{runtime.lock_holders(slots)}", flush=True)
                            waiting_lock_msg.add(j.name)
                        continue
                    for g in slots:
                        free.remove(g)
                if j.n_gpus == 0:
                    cpu_running += 1
                j.slots, j.locks = slots, locks
                j.attempts += 1
                env = dict(os.environ, **j.env)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(slots)
                env["PYTHONUNBUFFERED"] = "1"
                f = open(log_dir / f"{j.name}.log", "a", encoding="utf-8")
                f.write(f"\n===== {time.ctime()} | attempt {j.attempts} | gpus={slots} | {' '.join(j.cmd)}\n")
                f.flush()
                kw = {"pass_fds": tuple(locks)} if locks and os.name == "posix" else {}
                j.proc = subprocess.Popen(j.cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, **kw)
                j._f, j.t0, j.status = f, time.time(), "running"
                print(f"[scheduler] START {j.name} gpus={slots or 'cpu'}"
                      f"{'' if j.attempts == 1 else f' (retry {j.attempts - 1})'}", flush=True)
            for j in [x for x in jobs if x.status == "running"]:
                rc = j.proc.poll()
                if rc is None:
                    continue
                j._f.close()
                runtime.release_gpus(j.locks)
                free.extend(j.slots)
                j.locks = []
                if j.n_gpus == 0:
                    cpu_running -= 1
                mins = (time.time() - j.t0) / 60
                if rc == 0 and j.done_file.exists():
                    j.status = "done"
                    print(f"[scheduler] DONE  {j.name} ({mins:.1f} min)", flush=True)
                elif j.attempts <= max_retries:
                    j.status = "pending"
                    why = f"rc={rc}" if rc != 0 else f"rc=0 but {j.done_file} missing"
                    print(f"[scheduler] RETRY {j.name} {why} ({mins:.1f} min) -> {log_dir / (j.name + '.log')}",
                          flush=True)
                else:
                    j.status = "failed"
                    why = f"rc={rc}" if rc != 0 else f"rc=0 but {j.done_file} missing"
                    print(f"[scheduler] FAIL  {j.name} {why} ({mins:.1f} min) -> {log_dir / (j.name + '.log')}",
                          flush=True)
            _write_status(log_dir, jobs)
            if not any(j.status in ("pending", "running") for j in jobs):
                break
            if not any(j.status == "running" for j in jobs):
                launchable = [j for j in jobs if j.status == "pending" and all(by_name[d].status == "done" for d in j.deps)]
                if not launchable:
                    for j in jobs:
                        if j.status == "pending":
                            j.status = "skipped"
                            print(f"[scheduler] SKIP  {j.name} (unsatisfiable dependencies)", flush=True)
                    break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("[scheduler] interrupted -> terminating running jobs", flush=True)
        for j in jobs:
            if j.status == "running" and j.proc is not None:
                j.proc.terminate()
        for j in jobs:
            if j.status == "running" and j.proc is not None:
                try:
                    j.proc.wait(60)
                except subprocess.TimeoutExpired:
                    j.proc.kill()
                runtime.release_gpus(j.locks)
                j.status = "failed"
        _write_status(log_dir, jobs)
        raise
    summary = {s: [j.name for j in jobs if j.status == s] for s in ("done", "failed", "skipped")}
    print(f"[scheduler] finished: done={len(summary['done'])} failed={summary['failed']} skipped={summary['skipped']}",
          flush=True)
    return summary


def py(module, *args):
    return [sys.executable, "-m", module, *[str(a) for a in args]]
