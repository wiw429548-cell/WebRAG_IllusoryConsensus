"""All RAG methods (baselines + IndieRAG variants), batched over instances.

An instance: {"question", "docs": [ {text, origin?, domain?, date?} ], ...}
Each method returns (raw_output, short_prediction) per instance.
"""
import numpy as np

from . import prompts as PR
from .claims import group_answers
from .dependence import bayes_truth, clusters, ess
from .utils import extract_short_answer

MULTI_SUFFIX = " If several different answers are valid (e.g. the question is ambiguous), list all of them separated by ';'."


def fmt_docs(docs, header="Document"):
    return "\n\n".join(f"{header} [{i + 1}]: {d['text']}" for i, d in enumerate(docs))


class Ctx:
    """shared resources for a run: reader llm, doc store, dependence models, per-doc claims"""

    def __init__(self, llm, store, dep_models, claims, cfg, multi=False):
        self.llm, self.store, self.dep, self.claims, self.cfg, self.multi = llm, store, dep_models, claims, cfg, multi
        self._P = {}

    def claim(self, inst, d):
        return self.claims.get((inst["question"], d["text"]), "")

    def P(self, inst, var):
        key = (inst["iid"], var)
        if key not in self._P:
            self._P[key] = self.dep[var].matrix(inst["docs"], inst["question"])
        return self._P[key]

    def q(self, inst):
        return inst["question"] + (MULTI_SUFFIX if self.multi else "")


def _answer_from(raw):
    return extract_short_answer(raw)


# ------------------------------------------------------------------------------------------------- baselines
def vanilla(insts, ctx, docs_fn=None):
    prompts = []
    for it in insts:
        docs = docs_fn(it) if docs_fn else it["docs"]
        prompts.append([{"role": "system", "content": PR.QA_SYSTEM},
                        {"role": "user", "content": PR.VANILLA.format(docs=fmt_docs(docs), q=ctx.q(it))}])
    raws = ctx.llm.chat(prompts, max_tokens=200, tag="vanilla")
    return [(r, _answer_from(r)) for r in raws]


def dedup(insts, ctx):
    def keep(it):
        kept, sh = [], []
        for d in it["docs"]:
            s = ctx.store.doc(d["text"], it["question"])["s5"]
            if any(len(s & t) / max(1, min(len(s), len(t))) >= 0.5 for t in sh):
                continue
            kept.append(d)
            sh.append(s)
        return kept
    return vanilla(insts, ctx, keep)


def semdedup(insts, ctx):
    """SemDedup-style (Abbas et al. 2023) baseline: greedy removal of documents whose embedding cosine to an
    already-kept document is >= dependence.semdedup_cos, then vanilla RAG."""
    thr = float(ctx.cfg["dependence"].get("semdedup_cos", 0.95))

    def keep(it):
        kept, es = [], []
        for d in it["docs"]:
            e = ctx.store.doc(d["text"], it["question"])["e"]
            if any(float(e @ x) >= thr for x in es):
                continue
            kept.append(d)
            es.append(e)
        return kept
    return vanilla(insts, ctx, keep)


def majority(insts, ctx, fallback):
    out = []
    for it, fb in zip(insts, fallback):
        ans = [ctx.claim(it, d) for d in it["docs"]]
        gid, labels = group_answers(ans)
        if not labels:
            out.append(fb)
            continue
        cnt = np.bincount([g for g in gid if g >= 0], minlength=len(labels))
        if ctx.multi:
            pred = "; ".join(labels[g] for g in np.argsort(-cnt) if cnt[g] >= max(1, 0.5 * cnt.max()))
        else:
            pred = labels[int(np.argmax(cnt))]
        out.append(("majority-vote", pred))
    return out


def astute(insts, ctx):
    internal = ctx.llm.chat([PR.ASTUTE_INTERNAL.format(q=it["question"]) for it in insts], max_tokens=120, tag="astute_int")
    prompts = []
    for it, mem in zip(insts, internal):
        docs = f"[Internal] {mem}\n\n" + fmt_docs(it["docs"], "[External] Document")
        prompts.append(PR.ASTUTE_FINAL.format(docs=docs, q=ctx.q(it)))
    raws = ctx.llm.chat(prompts, max_tokens=400, tag="astute")
    return [(r, _answer_from(r)) for r in raws]


def instructrag(insts, ctx):
    raws = ctx.llm.chat([PR.INSTRUCTRAG.format(demo=PR.INSTRUCTRAG_DEMO, docs=fmt_docs(it["docs"]), q=ctx.q(it))
                         for it in insts], max_tokens=400, tag="instructrag")
    return [(r, _answer_from(r)) for r in raws]


def trustrag(insts, ctx):
    """TrustRAG stage-1 (simplified): 2-means on embeddings; a tight, multi-doc cluster is treated as injected and removed."""
    def keep(it):
        docs = it["docs"]
        if len(docs) < 3:
            return docs
        E = np.stack([ctx.store.doc(d["text"], it["question"])["e"] for d in docs])
        rng = np.random.RandomState(0)
        c = E[rng.choice(len(E), 2, replace=False)]
        for _ in range(20):
            a = np.argmax(E @ c.T, 1)
            c = np.stack([E[a == k].mean(0) if (a == k).any() else c[k] for k in range(2)])
        drop = set()
        for k in range(2):
            m = np.where(a == k)[0]
            if len(m) >= 2:
                S = E[m] @ E[m].T
                if (S.sum() - len(m)) / (len(m) * (len(m) - 1)) > ctx.cfg.get("trustrag_cos", 0.88):
                    drop |= set(m.tolist())
        kept = [d for i, d in enumerate(docs) if i not in drop]
        return kept or docs
    return vanilla(insts, ctx, keep)


def cag(insts, ctx):
    """CAG-style (Pan et al. 2024) credibility-aware generation, training-free: per-document credibility from the
    reader LLM (cached per question/document), documents annotated with it in the prompt."""
    import re as _re
    pairs = sorted({(it["question"], d["text"]) for it in insts for d in it["docs"]})
    u0 = ctx.llm.usage_snapshot()
    outs = ctx.llm.chat([PR.CRED.format(q=q, doc=t[:3000]) for q, t in pairs], max_tokens=4, tag="cred")
    u1 = ctx.llm.usage_snapshot()
    # ratings are generated once per unique (question, document); the reported cost is stand-alone per instance
    n_inst_docs = sum(len(it["docs"]) for it in insts)
    for k in ctx.llm.usage:
        ctx.llm.usage[k] += (u1[k] - u0[k]) / max(1, len(pairs)) * max(0, n_inst_docs - len(pairs))
    cred = {}
    for p, o in zip(pairs, outs):
        m = _re.search(r"\d+", o)
        cred[p] = min(10, max(1, int(m.group()))) if m else 5
    prompts = []
    for it in insts:
        docs = "\n\n".join(f"Document [{i + 1}] (credibility {cred[(it['question'], d['text'])]}/10): {d['text']}"
                            for i, d in enumerate(it["docs"]))
        prompts.append([{"role": "system", "content": PR.QA_SYSTEM},
                        {"role": "user", "content": PR.CAG.format(docs=docs, q=ctx.q(it))}])
    raws = ctx.llm.chat(prompts, max_tokens=200, tag="cag")
    return [(r, _answer_from(r)) for r in raws]


def madam(insts, ctx, rounds=2):
    """MADAM-RAG (Wang et al. 2025): one agent per document, multi-round debate, aggregator."""
    prev = [[None] * len(it["docs"]) for it in insts]
    for r in range(rounds):
        prompts, where = [], []
        for a, it in enumerate(insts):
            others = ""
            if r > 0:
                summ = "\n".join(f"- Agent {j + 1}: {x}" for j, x in enumerate(prev[a]))
                others = PR.MADAM_OTHERS.format(prev=summ)
            for b, d in enumerate(it["docs"]):
                prompts.append(PR.MADAM_AGENT.format(others=others, doc=d["text"][:3000], q=ctx.q(it)))
                where.append((a, b))
        outs = ctx.llm.chat(prompts, max_tokens=80, tag=f"madam{r}")
        for (a, b), o in zip(where, outs):
            prev[a][b] = " ".join(o.split())[:300]
    agg = ctx.llm.chat([PR.MADAM_AGG.format(answers="\n".join(f"- Agent {j + 1}: {x}" for j, x in enumerate(p)),
                                            q=ctx.q(it)) for it, p in zip(insts, prev)], max_tokens=200, tag="madam_agg")
    return [(r, _answer_from(r)) for r in agg]


# ------------------------------------------------------------------------------------------------- IndieRAG
def _groups(it, ctx):
    ans = [ctx.claim(it, d) for d in it["docs"]]
    return group_answers(ans)


def indievote(insts, ctx, var, fallback):
    out = []
    for it, fb in zip(insts, fallback):
        gid, labels = _groups(it, ctx)
        if not labels:
            out.append(fb)
            continue
        P = ctx.P(it, var)
        E = np.array([ess([i for i, g in enumerate(gid) if g == k], P) for k in range(len(labels))])
        cnt = np.bincount([g for g in gid if g >= 0], minlength=len(labels))
        order = sorted(range(len(labels)), key=lambda k: (-E[k], -cnt[k]))
        if ctx.multi:
            pred = "; ".join(labels[k] for k in order if E[k] >= max(1.0, 0.5 * E[order[0]]))
        else:
            pred = labels[order[0]]
        out.append((f"ess={E.round(2).tolist()} labels={labels}", pred))
    return out


def indiebayes(insts, ctx, var, fallback):
    out = []
    for it, fb in zip(insts, fallback):
        gid, labels = _groups(it, ctx)
        if not labels:
            out.append(fb)
            continue
        post = bayes_truth(gid, ctx.P(it, var))
        best = max(post, key=post.get)
        pred = "; ".join(labels[k] for k in post if post[k] >= 0.3) if ctx.multi else labels[best]
        out.append((f"post={post}", pred))
    return out


def indierag(insts, ctx, var, thr=None, mode="full"):
    """mode: full = evidence table + origin-compressed documents (IndieRAG); table = table + all documents;
    compress = origin-compressed documents without the table (component ablations)"""
    thr = ctx.cfg["dependence"]["hard_threshold"] if thr is None else thr
    prompts = []
    for it in insts:
        docs = it["docs"]
        P = ctx.P(it, var)
        cl = clusters(P, thr)
        gid, labels = _groups(it, ctx)
        rows = []
        for k, lab in enumerate(labels):
            idx = [i for i, g in enumerate(gid) if g == k]
            rows.append((ess(idx, P), len(idx), lab))
        rows.sort(key=lambda x: -x[0])
        table = "\n".join(f"- \"{lab}\": stated by {n} page(s), coming from about {e:.1f} independent origin(s)"
                          for e, n, lab in rows) or "- (no document states an explicit answer)"
        reps = []
        for c in sorted(set(cl)):
            mem = [i for i in range(len(docs)) if cl[i] == c]
            rep = max(mem, key=lambda i: len(docs[i]["text"]))
            reps.append((len(mem), docs[rep]))
        blocks = "\n\n".join(
            f"Origin group [{g + 1}] ({n} page(s) {'that copy/rewrite each other' if n > 1 else 'independent'}): {d['text']}"
            for g, (n, d) in enumerate(reps))
        if mode == "table":
            user = PR.INDIE_TABLE.format(table=table, docs=fmt_docs(docs), q=ctx.q(it))
        elif mode == "compress":
            user = PR.INDIE_COMPRESS.format(docs=blocks, q=ctx.q(it))
        else:
            user = PR.INDIE.format(table=table, docs=blocks, q=ctx.q(it))
        prompts.append([{"role": "system", "content": PR.QA_SYSTEM}, {"role": "user", "content": user}])
    tag = f"indierag_{var}" + ("" if thr == ctx.cfg["dependence"]["hard_threshold"] else f"_{thr}")
    raws = ctx.llm.chat(prompts, max_tokens=200, tag=tag + ("" if mode == "full" else f"_{mode}"))
    return [(r, _answer_from(r)) for r in raws]


def run_method(name, insts, ctx, cache):
    """cache: dict of already-computed method outputs (vanilla is needed as fallback by vote methods)"""
    if name == "vanilla":
        return vanilla(insts, ctx)
    # vanilla answers are the fallback of the vote methods (instances where no document states an answer);
    # aligned by instance id so that methods run on a SUBSET of instances get the right fallback
    vmap = cache.get("vanilla_by_iid") or {}
    missing = [it for it in insts if it["iid"] not in vmap]
    if missing:
        vmap.update({it["iid"]: r for it, r in zip(missing, vanilla(missing, ctx))})
        cache["vanilla_by_iid"] = vmap
    fb = [vmap[it["iid"]] for it in insts]
    if name == "dedup":
        return dedup(insts, ctx)
    if name == "semdedup":
        return semdedup(insts, ctx)
    if name == "majority":
        return majority(insts, ctx, fb)
    if name == "astute":
        return astute(insts, ctx)
    if name == "instructrag":
        return instructrag(insts, ctx)
    if name == "trustrag":
        return trustrag(insts, ctx)
    if name == "madam":
        return madam(insts, ctx)
    if name == "cag":
        return cag(insts, ctx)
    kind, var = name.split(":")
    thr = None
    if "#" in var:
        var, thr = var.split("#")
        thr = float(thr)
    if kind == "indievote":
        return indievote(insts, ctx, var, fb)
    if kind == "indiebayes":
        return indiebayes(insts, ctx, var, fb)
    if kind == "indierag":
        return indierag(insts, ctx, var, thr)
    if kind in ("indierag_table", "indierag_compress"):
        return indierag(insts, ctx, var, thr, mode=kind.split("_")[1])
    raise ValueError(name)
