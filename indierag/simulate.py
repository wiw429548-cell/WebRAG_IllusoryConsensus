"""Theory check (Proposition: copy-corrected Condorcet jury theorem).

n independent origins, each correct w.p. p; one origin (correct w.p. p as well) is duplicated m times.
naive page-majority error grows with m and -> P(that origin is wrong) = 1-p, while origin-level voting is
independent of m. With an imperfect dependence detector (recall r), the error interpolates between the two."""
import numpy as np

from .utils import ROOT, save_json


def simulate(p=0.65, n=5, ms=(0, 1, 2, 4, 8, 16, 32), recalls=(0.0, 0.5, 0.8, 0.95, 1.0), trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for m in ms:
        correct = rng.random((trials, n)) < p          # origin 0 is the copied one
        res[m] = {}
        for r in recalls:
            # each copy is detected (merged into origin 0) with prob r; undetected copies count as extra votes
            undetected = rng.binomial(m, 1 - r, trials) if m else np.zeros(trials, int)
            votes_correct = correct.sum(1) + undetected * correct[:, 0]
            total = n + undetected
            acc = (votes_correct > total / 2).mean() + 0.5 * (votes_correct == total / 2).mean()
            res[m][r] = float(acc)
    return res


def main():
    out = ROOT / "results" / "theory"
    res = {f"p={p}": simulate(p=p) for p in (0.55, 0.65, 0.75)}
    save_json(res, out / "simulation.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
        for ax, (k, d) in zip(axes, res.items()):
            ms = sorted(d, key=int)
            for r in sorted(next(iter(d.values())), key=float):
                ax.plot([int(m) for m in ms], [d[m][r] for m in ms], marker="o", label=f"detector recall={r}")
            ax.set_xscale("symlog")
            ax.set_title(k)
            ax.set_xlabel("#copies m of one origin")
        axes[0].set_ylabel("majority-vote accuracy")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "simulation.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
