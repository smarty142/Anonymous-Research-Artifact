"""TPR at 1% FPR for the tool-call ablation (reviewer question).

The paper's matched operating point is 7% FPR (the native FPR of the
signature layer on this bench). Production guards run tighter, so this
script re-evaluates the same seven-subset ablation grid at a 1% FPR
budget, zero API calls: text/provenance scores come from the shipped
caches, the pattern channel is recomputed locally as always.

Outputs results/run_tpr_at_1pct.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

from evaluation import stats
from pipeline.conformal import detect_patterns, nonconformity_score
from run_ensemble_ablation import _d01, _load_hard, _load_scores
from run_toolcall_bench import build_bench

N_BOOT = 2000
FPR_BUDGETS = [0.01, 0.07]


def _paired(labels, a, b, stat):
    d, lo, hi = stats.bootstrap_paired_diff(labels, a, b, stat_fn=stat,
                                            n_boot=N_BOOT, seed=0)
    return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    trajs = build_bench(n_per_pattern=20, benign_reps=30, seed=0)[0]
    n = len(trajs)
    labels = [t.ground_truth == "malicious" for t in trajs]
    text = _load_scores("or_claude-haiku-4-5", "toolcall")
    prov = _load_hard("camel_full_or", "toolcall")
    pattern = [_d01(nonconformity_score(detect_patterns(t))) for t in trajs]
    for name, col in (("text", text), ("prov", prov), ("pattern", pattern)):
        assert len(col) == n, \
            f"{name} cache has {len(col)} rows, expected {n}"

    def mx(*cols):
        return [max(vals) for vals in zip(*cols)]

    tpr = {b: (lambda l, s, _b=b: stats.tpr_at_fpr(l, s, _b))
           for b in FPR_BUDGETS}

    subsets = {
        "text": [text], "prov": [prov], "pattern": [pattern],
        "text+prov": [text, prov], "text+pattern": [text, pattern],
        "prov+pattern": [prov, pattern], "triple": [text, prov, pattern],
    }
    out = {"config": {"bench": "toolcall", "n": n, "n_boot": N_BOOT,
                      "seed": 0, "fpr_budgets": FPR_BUDGETS},
           "ablation": {}, "paired_diffs": {}}
    fused_scores = {}
    for name, cols in subsets.items():
        f = mx(*cols)
        fused_scores[name] = f
        row = {"auc": round(stats.auc(labels, f), 4)}
        for b in FPR_BUDGETS:
            row[f"tpr@{int(b * 100)}pct"] = round(100 * tpr[b](labels, f), 1)
        out["ablation"][name] = row

    # the two headline paired comparisons, at both operating points
    for b in FPR_BUDGETS:
        tag = int(b * 100)
        out["paired_diffs"][f"triple_vs_text@{tag}pct"] = {
            "auc": _paired(labels, fused_scores["text"],
                           fused_scores["triple"], stats.auc),
            "tpr": _paired(labels, fused_scores["text"],
                           fused_scores["triple"], tpr[b])}
        out["paired_diffs"][f"textprov_vs_text@{tag}pct"] = {
            "auc": _paired(labels, fused_scores["text"],
                           fused_scores["text+prov"], stats.auc),
            "tpr": _paired(labels, fused_scores["text"],
                           fused_scores["text+prov"], tpr[b])}

    # score-granularity diagnostics (paper Sec. VI-C, "Tighter operating
    # points"): the text guard emits two-decimal probabilities, so a
    # benign tie plateau can make the 1% and 7% budgets select the same
    # threshold -- and therefore the same TPR -- for every subset.
    benign_text = [s for s, l in zip(text, labels) if not l]
    n_ben = len(benign_text)
    gran = {"n_distinct_text_scores": len(set(text)), "n_benign": n_ben}
    for tag, budget in ((1, 0.01), (7, 0.07)):
        thr = stats.threshold_at_fpr(labels, text, budget)
        preds = stats.apply_threshold(text, thr)
        fp = sum(1 for p, l in zip(preds, labels) if p and not l)
        below = [s for s in benign_text if s < thr]
        blocker = max(below) if below else None
        gran[f"text@{tag}pct"] = {
            "threshold": None if thr == math.inf else round(float(thr), 4),
            "actual_fpr_pct": round(100.0 * fp / n_ben, 2),
            "blocking_tie_value": blocker,
            "benign_tied_at_blocker": sum(1 for s in benign_text if s == blocker),
        }
    gran["tpr_identical_across_budgets"] = {
        name: row["tpr@1pct"] == row["tpr@7pct"]
        for name, row in out["ablation"].items()}
    out["granularity"] = gran

    Path("results/run_tpr_at_1pct.json").write_text(json.dumps(out, indent=2))

    print("=== tool-call ablation at 1% and 7% FPR ===")
    hdr = f"{'system':<14}{'AUC':>7}{'TPR@1%':>8}{'TPR@7%':>8}"
    print(hdr)
    for name, row in out["ablation"].items():
        print(f"{name:<14}{row['auc']:>7}{row['tpr@1pct']:>8}"
              f"{row['tpr@7pct']:>8}")
    for k, v in out["paired_diffs"].items():
        print(f"{k}: auc {v['auc']}  tpr {v['tpr']}")
    print("[written] results/run_tpr_at_1pct.json")


if __name__ == "__main__":
    main()
