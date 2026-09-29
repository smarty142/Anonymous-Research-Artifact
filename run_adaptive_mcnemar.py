"""Adaptive-bench McNemar b/c counts at matched operating points.

Paper Sec. VI-I quotes, for the GLM-scored adaptive run, the McNemar
discordant-pair counts b = 0 / c = 37 (fused vs text), b = 0 / c = 33
(fused vs text+provenance) and b = 0 / c = 4 (text+provenance vs text).
The shipped run_adaptive_fusion_eval_glm.json stores the comparison
p-values but not the b/c counts themselves; this script recomputes them
from the same per-trajectory caches with the same threshold convention
(each system's own 7%-FPR threshold on the shared 680 rows), zero API
calls:

  - 440 attacks: AdaptiveBench(seed=0, n_per_pattern=20), rebuilt and
    asserted against the shipped ids before any cache read
  - 240 benign control: the tool-call bench's benign rows, rescored on
    the run's provider (caches glm_guard/camel_glm __adaptive_benign)
  - pattern channel recomputed locally as always

b counts rows the smaller system catches and the fused one misses;
c the reverse (evaluation.stats.mcnemar semantics, first argument is
the larger system).

Outputs results/run_adaptive_mcnemar.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from adaptive.generator import AdaptiveBench
from evaluation import stats
from pipeline.conformal import detect_patterns, nonconformity_score
from run_ensemble_ablation import _d01, _load_hard, _load_scores
from run_toolcall_bench import build_bench

FPR_BUDGET = 0.07
N_ATTACKS = 440
N_BENIGN = 240


def main() -> None:
    atk = list(AdaptiveBench(seed=0, n_per_pattern=20))
    assert len(atk) == N_ATTACKS, f"expected {N_ATTACKS} attacks, got {len(atk)}"

    tc, _, _ = build_bench(n_per_pattern=20, benign_reps=30, seed=0)
    ben_idx = [i for i, t in enumerate(tc) if t.ground_truth != "malicious"]
    assert len(ben_idx) == N_BENIGN

    text = ([float(s) for s in _load_scores("glm_guard", "adaptive")] +
            [float(s) for s in _load_scores("glm_guard", "adaptive_benign")])
    prov = ([float(m) for m in _load_hard("camel_glm", "adaptive")] +
            [float(m) for m in _load_hard("camel_glm", "adaptive_benign")])
    pat = ([_d01(nonconformity_score(detect_patterns(t))) for t in atk] +
           [_d01(nonconformity_score(detect_patterns(tc[i]))) for i in ben_idx])
    labels = [True] * N_ATTACKS + [False] * N_BENIGN
    n = len(labels)
    for name, col in (("text", text), ("prov", prov), ("pattern", pat)):
        assert len(col) == n, f"{name} column has {len(col)} rows, expected {n}"

    def mx(*cols):
        return [max(vals) for vals in zip(*cols)]

    systems = {
        "text": text,
        "text+provenance": mx(text, prov),
        "text+provenance+pattern": mx(text, prov, pat),
    }

    out = {"config": {"bench": "adaptive (440 attacks + 240 benign control)",
                      "fpr_budget": FPR_BUDGET,
                      "threshold": "each system's own 7%-FPR threshold",
                      "semantics": "b = smaller-only catches, c = larger-only"},
           "systems": {k: {"auc": round(stats.auc(labels, v), 4)}
                       for k, v in systems.items()},
           "comparisons": []}

    for big, small in (("text+provenance+pattern", "text+provenance"),
                       ("text+provenance+pattern", "text"),
                       ("text+provenance", "text")):
        th_big = stats.threshold_at_fpr(labels, systems[big], FPR_BUDGET)
        th_small = stats.threshold_at_fpr(labels, systems[small], FPR_BUDGET)
        mc = stats.mcnemar(labels,
                           stats.apply_threshold(systems[big], th_big),
                           stats.apply_threshold(systems[small], th_small))
        out["comparisons"].append({
            "larger": big, "smaller": small,
            "b": mc["b"], "c": mc["c"],
            "p_value": round(mc["p_value"], 6),
            "threshold_larger": round(float(th_big), 4),
            "threshold_smaller": round(float(th_small), 4),
        })

    Path("results/run_adaptive_mcnemar.json").write_text(json.dumps(out, indent=2))

    print("=== adaptive McNemar at matched 7%-FPR operating points ===")
    for cmp_ in out["comparisons"]:
        print(f"{cmp_['larger']:>26} vs {cmp_['smaller']:<18} "
              f"b={cmp_['b']:>2} c={cmp_['c']:>2} p={cmp_['p_value']}")
    print("[written] results/run_adaptive_mcnemar.json")


if __name__ == "__main__":
    main()
