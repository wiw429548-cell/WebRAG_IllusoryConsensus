"""Batched chat LLM with a persistent on-disk cache (resumable runs).

backend 'vllm' (default on the server) or 'hf' (transformers; used for local smoke tests / when vLLM is absent).
Every (model, messages, sampling params, sample index) is cached in data/cache/<model>.sqlite, so re-running a job
after a crash only generates what is missing.
"""
import hashlib
import json
import os
import sqlite3
import threading

from .runtime import build_vllm, setup_env
from .utils import DATA, slug

setup_env()  # must run before vLLM is imported anywhere in this process


class Cache:
    def __init__(self, model):
        p = DATA / "cache" / f"{slug(model)}.sqlite"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(p), check_same_thread=False, timeout=600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get_many(self, keys):
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = "SELECT k, v FROM c WHERE k IN (%s)" % ",".join("?" * len(chunk))
            for k, v in self.db.execute(q, chunk):
                out[k] = v
        return out

    def put_many(self, items):
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO c VALUES (?, ?)", items)
            self.db.commit()


class LLM:
    def __init__(self, model, backend=None, max_model_len=8192, gpu_mem=0.85, tp=1, dtype="auto",
                 enable_thinking=False, seed=0):
        self.model = model
        self.backend = backend or os.environ.get("LLM_BACKEND") or self._default_backend()
        self.cache = Cache(model)
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.engine_info = {"backend": self.backend, "model": model, "max_model_len": max_model_len}
        if self.backend == "vllm":
            # TP clamp + free-memory-aware gpu_mem + retry ladder; sets VLLM_USE_FLASHINFER_SAMPLER=0 (no nvcc needed)
            self.engine, self.engine_info = build_vllm(
                model, tp=tp, max_model_len=max_model_len, gpu_mem=gpu_mem, dtype=dtype, trust_remote_code=True,
                seed=seed, enable_prefix_caching=True, log_file=DATA.parent / "results" / "engine_log.jsonl")
            self.engine_info["backend"] = "vllm"
            max_model_len = self.engine_info["max_model_len"]
            self.tok = self.engine.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"
            dt = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available()                 else torch.float32
            self.hf = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=dt,
                trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.max_model_len = max_model_len
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}  # cost as if run uncached

    @staticmethod
    def _default_backend():
        import importlib.util
        return "vllm" if importlib.util.find_spec("vllm") else "hf"

    # ------------------------------------------------------------------ prompt rendering
    def render(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if getattr(self.tok, "chat_template", None):
            kw = {"enable_thinking": self.enable_thinking} if "qwen3" in self.model.lower() else {}
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        return "\n\n".join(m["content"] for m in messages) + "\n\n"

    def _truncate(self, prompt, max_new):
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        budget = self.max_model_len - max_new - 8
        if len(ids) > budget:  # keep the head (instructions) and the tail (question)
            head = budget // 3
            prompt = self.tok.decode(ids[:head]) + "\n...\n" + self.tok.decode(ids[-(budget - head):])
        return prompt

    # ------------------------------------------------------------------ generation
    def generate(self, prompts, temperature=0.0, max_tokens=256, n=1, stop=None, raw=False, tag=""):
        """prompts: list of (messages | str). raw=True -> str prompts are used verbatim (no chat template).
        returns list (len(prompts)) of list (n) of strings."""
        rendered = [p if (raw and isinstance(p, str)) else self.render(p) for p in prompts]
        params = json.dumps([temperature, max_tokens, stop, self.seed, tag])
        keys = [[hashlib.sha1((r + params + f"#{j}").encode()).hexdigest() for j in range(n)] for r in rendered]
        flat = [k for ks in keys for k in ks]
        hit = self.cache.get_many(flat)
        todo = [i for i, ks in enumerate(keys) if any(k not in hit for k in ks)]
        # generate in chunks and commit each chunk to the cache -> a crashed / killed job loses at most one chunk
        chunk = int(os.environ.get("LLM_CACHE_CHUNK", "2048"))
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            outs = self._run([self._truncate(rendered[i], max_tokens) for i in part], temperature, max_tokens, n, stop)
            items = []
            for i, o in zip(part, outs):
                for j, txt in enumerate(o):
                    hit[keys[i][j]] = txt
                    items.append((keys[i][j], txt))
            self.cache.put_many(items)
        out = [[hit[k] for k in ks] for ks in keys]
        self._account(rendered, out, max_tokens)
        return out

    def _account(self, rendered, outs, max_new):
        try:
            lp = [len(x) for x in self.tok(rendered, add_special_tokens=False)["input_ids"]]
            lc = [len(x) for o in outs for x in self.tok(o, add_special_tokens=False)["input_ids"]] if outs else []
        except Exception:  # tokenizer without batch support
            lp, lc = [len(r) // 4 for r in rendered], [len(t) // 4 for o in outs for t in o]
        n = len(outs[0]) if outs else 1
        self.usage["calls"] += len(rendered) * n
        self.usage["prompt_tokens"] += sum(min(x, self.max_model_len - max_new) for x in lp) * n
        self.usage["completion_tokens"] += sum(lc)

    def usage_snapshot(self):
        return dict(self.usage)

    def chat(self, prompts, **kw):
        return [o[0] for o in self.generate(prompts, **kw)]

    def _run(self, prompts, temperature, max_tokens, n, stop):
        if self.backend == "vllm":
            from vllm import SamplingParams
            n_req = 1 if temperature == 0 else n  # some vLLM versions reject n > 1 with greedy decoding
            sp = SamplingParams(temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                                max_tokens=max_tokens, n=n_req, stop=stop, seed=self.seed)
            res = self.engine.generate(prompts, sp, use_tqdm=len(prompts) > 200)
            return [[c.text for c in r.outputs] * (n // n_req) for r in res]
        import torch
        out = []
        bs = 8
        for s in range(0, len(prompts), bs):
            batch = prompts[s:s + bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.hf.device)
            with torch.no_grad():
                g = self.hf.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                     temperature=max(temperature, 1e-5), num_return_sequences=n,
                                     pad_token_id=self.tok.pad_token_id)
            g = g[:, enc["input_ids"].shape[1]:]
            txt = self.tok.batch_decode(g, skip_special_tokens=True)
            for b in range(len(batch)):
                outs = txt[b * n:(b + 1) * n]
                if stop:
                    outs = [min((o.split(st)[0] for st in stop), key=len) for o in outs]
                out.append(outs)
        return out
