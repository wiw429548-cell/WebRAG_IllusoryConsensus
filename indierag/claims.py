"""Per-document claim extraction (the answer each document supports) and answer-string grouping."""
from collections import Counter

from . import prompts as PR
from .utils import f1, normalize_answer


def extract_claims(llm, question_doc_pairs):
    """[(question, doc_text)] -> list of short answers ('' = no answer)"""
    outs = llm.chat([PR.CLAIM.format(doc=d[:4000], q=q) for q, d in question_doc_pairs], max_tokens=24, tag="claim")
    res = []
    for o in outs:
        a = o.strip().split("\n")[0].strip().strip('."')
        if not a or "none" in a.lower()[:6] or len(a.split()) > 12:
            a = ""
        res.append(a)
    return res


def group_answers(answers):
    """cluster answer strings (exact-normalised, containment or token-F1 >= 0.6) -> (group id per answer (-1 for
    empty), label per group)"""
    norm = [normalize_answer(a) for a in answers]
    parent = list(range(len(answers)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    idx = [i for i, n in enumerate(norm) if n]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            ni, nj = norm[i], norm[j]
            if ni == nj or f" {ni} " in f" {nj} " or f" {nj} " in f" {ni} " or f1(ni, [nj]) >= 0.6:
                parent[find(i)] = find(j)
    gid, roots = [-1] * len(answers), {}
    for i in idx:
        gid[i] = roots.setdefault(find(i), len(roots))
    labels = []
    for g in range(len(roots)):
        members = [answers[i] for i in idx if gid[i] == g]
        labels.append(Counter(members).most_common(1)[0][0])
    return gid, labels
