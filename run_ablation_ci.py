"""Complete statistical treatment for the 7-subset ablation grid
(review #4/#10 and section-3 statistical-treatment critique; zero API).

Adds to what run_fusion_rule_ablation.py already computes (per-cell AUC
bootstrap CIs):

  1. TPR@7%-FPR bootstrap CIs for every computable grid cell (the
     operating-point metric the headline claims use), with the threshold
     re-derived inside every bootstrap replicate.
  2. Exact McNemar tests for every pairwise subset comparison at each
     subset's own 7%-FPR operating point, with Holm–Bonferroni family
     correction per benchmark.
  3. A sign-stability check for the AgentDojo pattern-channel
     anti-correlation (AUC 0.152): the fraction of bootstrap replicates
     whose AUC stays below 0.5, i.e. whether the inversion itself is
     stable rather than a point-estimate fluke.

Output: results/run_ablation_ci.json
"""
from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, ".")

from evaluation import stats
from run_fusion_rule_ablation import (build_benchmark_matrices, CHANNELS,
                                      bootstrap_auc_ci)

FPR_BUDGET = 0.07
N_BOOT = 2000


def _r4(x):
    return None if x is None else round(float(x), 4)


def bootstrap_tpr_ci(labels, scores, n_boot=N_BOOT, seed=0):
    """Percentile CI for TPR at the benchmark's own 7%-FPR operating point,
    re-deriving the threshold inside each replicate."""
    def _tpr(lb, sb):
        return stats.tpr_at_fpr(lb, sb, FPR_BUDGET)
    _, lo, hi = stats.bootstrap_ci(labels, scores, _tpr, n_boot=n_boot, seed=seed)
    return [round(lo, 4), round(hi, 4)]


def sign_stability(labels, scores, n_boot=N_BOOT, seed=0):
    """Fraction of bootstrap replicates with AUC < 0.5, plus the replicates'
    2.5/97.5 percentiles — is the below-chance inversion stable?"""
    rng = random.Random(seed)
    n = len(labels)
    idx_all = list(range(n))
    aucs = []
    for _ in range(n_boot):
        idx = [rng.choice(idx_all) for _ in range(n)]
        la = [labels[i] for i in idx]
        if all(la) or not any(la):
            continue
        aucs.append(stats.auc(la, [scores[i] for i in idx]))
    aucs.sort()
    return {"frac_auc_below_half": round(sum(1 for a in aucs if a < 0.5) / len(aucs), 4),
            "auc_ci95": [round(aucs[int(0.025 * len(aucs))], 4),
                         round(aucs[int(0.975 * len(aucs)) - 1], 4)],
            "auc_median": round(aucs[len(aucs) // 2], 4)}


def main():
    benches = build_benchmark_matrices()
    out = {"fpr_budget": FPR_BUDGET, "n_boot": N_BOOT, "per_bench": {}}

    for bname, B in benches.items():
        labels = B["labels"]
        subsets = []
        fused_scores, preds = {}, {}
        for k in (1, 2, 3):
            for subset in combinations(CHANNELS, k):
                if any(B["mat"][c][i] is None for c in subset for i in range(len(labels))):
                    continue
                name = "+".join(subset)
                fused = [max(B["mat"][c][i] for c in subset) for i in range(len(labels))]
                fused_scores[name] = fused
                th = stats.threshold_at_fpr(labels, fused, FPR_BUDGET)
                preds[name] = stats.apply_threshold(fused, th)
                subsets.append(name)

        rows = {}
        for name, fused in fused_scores.items():
            rows[name] = {
                "auc": _r4(stats.auc(labels, fused)),
                "auc_ci95": bootstrap_auc_ci(labels, fused, seed=0),
                "tpr_at_7pct_fpr": _r4(stats.tpr_at_fpr(labels, fused, FPR_BUDGET)),
                "tpr_ci95": bootstrap_tpr_ci(labels, fused, seed=0),
            }

        # pairwise McNemar at each subset's own operating point + Holm
        pairs, raw_ps = [], []
        for a, b in combinations(subsets, 2):
            mc = stats.mcnemar(labels, preds[a], preds[b])
            pairs.append({"a": a, "b": b, "b_only_catches": mc["b"],
                          "c_only_catches": mc["c"], "p_value": _r4(mc["p_value"])})
            raw_ps.append(mc["p_value"])
        adj = stats.holm_bonferroni(raw_ps)
        for p, a in zip(pairs, adj):
            p["p_holm"] = _r4(a)
            p["significant_holm_0.05"] = bool(a is not None and a < 0.05)
        pairs.sort(key=lambda p: p["p_holm"] if p["p_holm"] is not None else 1.0)

        out["per_bench"][bname] = {"n": len(labels), "n_malicious": sum(labels),
                                   "rows": rows, "pairwise_mcnemar_holm": pairs}

        print(f"\n### {bname} (n={len(labels)}, mal={sum(labels)})")
        print("| subset | AUC [CI] | TPR@7%FPR [CI] |")
        print("|---|---|---|")
        for name, r in rows.items():
            print(f"| {name} | {r['auc']:.4f} {r['auc_ci95']} "
                  f"| {r['tpr_at_7pct_fpr']:.3f} {r['tpr_ci95']} |")
        sig = sum(1 for p in pairs if p["significant_holm_0.05"])
        print(f"\nMcNemar pairwise: {len(pairs)} tests, {sig} significant after Holm (alpha=0.05)")
        for p in pairs[:6]:
            print(f"  {p['a']} vs {p['b']}: p={p['p_value']} holm={p['p_holm']} "
                  f"(b_only={p['b_only_catches']}, c_only={p['c_only_catches']})")

    # AgentDojo pattern-channel sign stability (the 0.152 anti-correlation)
    ad = benches["agentdojo"]
    pat = ad["mat"]["pattern"]
    if not any(s is None for s in pat):
        ss = sign_stability(ad["labels"], pat, seed=0)
        out["agentdojo_pattern_sign_stability"] = ss
        print(f"\n### AgentDojo pattern-channel sign stability: {ss}")

    Path("results/run_ablation_ci.json").write_text(json.dumps(out, indent=2))
    print("\n[written] results/run_ablation_ci.json")


if __name__ == "__main__":
    raise SystemExit(main())
