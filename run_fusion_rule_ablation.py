"""Fusion-rule ablation + bootstrap CIs for the three-detector ensemble.

Addresses review section 7 / roadmap #8 (fusion-rule ablation for the
actual TriChannel ensemble, previously only run for the
composition-ceiling filter+classifier experiment) and roadmap #4 (CIs
for every cell of the ablation grid, especially the AgentDojo
pattern-channel anti-correlation AUC 0.152).

Cache-only: text = cached Claude Haiku 4.5 continuous scores
(or_claude-haiku-4-5__<bench>.jsonl), provenance = cached full-CaMeL
hard labels (camel_full_or__<bench>.jsonl), pattern = local signature
non-conformity score normalized d/(d+1). No API calls.

Channel construction and benchmark matrices are copied verbatim from
run_ensemble_ablation.py (the script behind the paper's Section 4.3
ensemble table) so numbers are directly comparable.

Fusion rules: max/mean/product/noisy-or are the paper's
composition-ceiling rules (run_review_response.py FUSIONS) extended
from two to three channels; lr is the learned rule (logistic
regression over the three channel scores, out-of-fold stratified
5-fold CV decision_function so every trajectory gets a prediction
from a model that did not see it — same estimator family as
run_ensemble_ablation.lr_cv_auc, but per-trajectory scores so the
bootstrap CI is well-defined).

Output: results/run_fusion_rule_ablation.json
"""
from __future__ import annotations
import json
import math
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from benchmarks.injecagent import InjecAgentBench
from benchmarks.agentdojo import AgentDojoBench
from run_toolcall_bench import build_bench
from run_agentdojo_fpr import build_benign
from pipeline.conformal import nonconformity_score, detect_patterns
from evaluation import stats

CHANNELS = ("text", "provenance", "pattern")
N_BOOT = 2000


def _load_scores(name: str, bench: str):
    c = {}
    for line in Path(f"results/cache/{name}__{bench}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = r["score"]
    return [c[i] for i in range(len(c))]


def _load_hard(name: str, bench: str):
    c = {}
    for line in Path(f"results/cache/{name}__{bench}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = float(r["mal"])
    return [c[i] for i in range(len(c))]


def _d01(d: float) -> float:
    return d / (d + 1.0) if d > 0 else 0.0


FUSIONS = {
    "max":      lambda xs: [max(v) for v in zip(*xs)],
    "mean":     lambda xs: [sum(v) / len(v) for v in zip(*xs)],
    "noisy-or": lambda xs: [1.0 - math.prod(1.0 - min(x, 1.0) for x in v)
                            for v in zip(*xs)],
    "product":  lambda xs: [math.prod(v) for v in zip(*xs)],
}


def build_benchmark_matrices():
    """{bench: {"labels": [...], "mat": {channel: [scores]}}}.

    Identical to run_ensemble_ablation.build_benchmark_matrices, except
    provenance scores may be None on AgentDojo constructed benign (no
    labels), which is carried through rather than dropped.
    """
    out = {}

    trajs = list(InjecAgentBench(max_trajs=510))
    out["injecagent"] = {
        "labels": [t.ground_truth == "malicious" for t in trajs],
        "mat": {
            "text": _load_scores("or_claude-haiku-4-5", "injecagent"),
            "provenance": _load_hard("camel_full_or", "injecagent"),
            "pattern": [_d01(nonconformity_score(detect_patterns(t))) for t in trajs],
        },
    }

    tc, _, _ = build_bench(n_per_pattern=20, benign_reps=30, seed=0)
    out["toolcall"] = {
        "labels": [t.ground_truth == "malicious" for t in tc],
        "mat": {
            "text": _load_scores("or_claude-haiku-4-5", "toolcall"),
            "provenance": _load_hard("camel_full_or", "toolcall"),
            "pattern": [_d01(nonconformity_score(detect_patterns(t))) for t in tc],
        },
    }

    ad = list(AgentDojoBench(max_trajs=280)) + list(build_benign(n_per_suite=40))
    prov = _load_hard("camel_full_or", "agentdojo")  # attacks only (280)
    out["agentdojo"] = {
        "labels": [t.ground_truth == "malicious" for t in ad],
        "mat": {
            "text": _load_scores("or_claude-haiku-4-5", "agentdojo")
            + _load_scores("or_claude-haiku-4-5", "agentdojo_benign"),
            "provenance": prov + [None] * 120,  # no labels for constructed benign
            "pattern": [_d01(nonconformity_score(detect_patterns(t))) for t in ad],
        },
    }
    return out


def bootstrap_auc_ci(labels, scores, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap CI for a single score vector's AUC."""
    rng = random.Random(seed)
    n = len(labels)
    idx_all = list(range(n))
    aucs = []
    for _ in range(n_boot):
        idx = [rng.choice(idx_all) for _ in range(n)]
        la = [labels[i] for i in idx]
        if all(la) or not any(la):
            continue
        try:
            aucs.append(stats.auc(la, [scores[i] for i in idx]))
        except Exception:
            continue
    if not aucs:
        return None
    aucs.sort()
    return [round(aucs[int(0.025 * len(aucs))], 4),
            round(aucs[int(0.975 * len(aucs)) - 1], 4)]


def paired_diff_ci(labels, scores_a, scores_b, n_boot=N_BOOT, seed=0):
    """Paired bootstrap CI for AUC(a) - AUC(b)."""
    rng = random.Random(seed)
    n = len(labels)
    idx_all = list(range(n))
    diffs = []
    for _ in range(n_boot):
        idx = [rng.choice(idx_all) for _ in range(n)]
        la = [labels[i] for i in idx]
        if all(la) or not any(la):
            continue
        try:
            diffs.append(stats.auc(la, [scores_a[i] for i in idx])
                         - stats.auc(la, [scores_b[i] for i in idx]))
        except Exception:
            continue
    if not diffs:
        return None
    diffs.sort()
    return [round(diffs[int(0.025 * len(diffs))], 4),
            round(diffs[int(0.975 * len(diffs)) - 1], 4)]


def lr_oof_scores(mat, labels, n_splits=5, seed=0):
    """Out-of-fold decision_function scores for the learned fusion rule.

    One score per trajectory, produced by a fold model that did not see
    it, so the resulting vector can be bootstrapped/AUC'd like any other
    fusion rule's output.
    """
    X = np.asarray([[mat[c][i] for c in CHANNELS] for i in range(len(labels))],
                   dtype=float)
    y = np.asarray(labels, dtype=int)
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.decision_function(sc.transform(X[te]))
    return oof.tolist()


def main():
    benches = build_benchmark_matrices()
    out = {"n_boot": N_BOOT, "single_channel_auc_ci": {},
           "ablation_grid_max_ci": {}, "fusion_rules": {}}

    # --- 1. single-channel AUC with bootstrap CI (roadmap #4) ---
    print("=== 1. single-channel AUC + bootstrap CI ===", flush=True)
    for bname, B in benches.items():
        labels = B["labels"]
        rows = {}
        for ch in CHANNELS:
            sc = B["mat"][ch]
            if any(s is None for s in sc):
                rows[ch] = {"auc": None, "note": "provenance channel has no labels for constructed benign"}
                continue
            auc = stats.auc(labels, sc)
            rows[ch] = {"auc": round(auc, 4), "auc_ci95": bootstrap_auc_ci(labels, sc, seed=0)}
            print(f"  {bname:12s} {ch:11s} AUC={auc:.4f} CI={rows[ch]['auc_ci95']}")
        out["single_channel_auc_ci"][bname] = rows

    # --- 2. CIs for every max-fusion cell of the ablation grid (roadmap #4) ---
    print("\n=== 2. max-fusion ablation grid + paired CI vs best single ===", flush=True)
    for bname, B in benches.items():
        labels = B["labels"]
        rows = {}
        single_aucs = {ch: (None if any(s is None for s in B["mat"][ch])
                            else stats.auc(labels, B["mat"][ch])) for ch in CHANNELS}
        computable = [c for c in single_aucs if single_aucs[c] is not None]
        best_single = max(computable, key=lambda c: single_aucs[c]) if computable else None
        for k in (1, 2, 3):
            for subset in combinations(CHANNELS, k):
                if any(B["mat"][ch2][i] is None for ch2 in subset for i in range(len(labels))):
                    continue  # subset needs a channel lacking labels
                fused = FUSIONS["max"]([B["mat"][c] for c in subset])
                auc = stats.auc(labels, fused)
                row = {"auc": round(auc, 4), "auc_ci95": bootstrap_auc_ci(labels, fused, seed=0)}
                if best_single and subset != (best_single,):
                    row["minus_best_single_ci95"] = paired_diff_ci(
                        labels, fused, B["mat"][best_single], seed=0)
                rows["+".join(subset)] = row
                print(f"  {bname:12s} {'+'.join(subset):28s} AUC={auc:.4f} "
                      f"CI={row['auc_ci95']} vs_best={row.get('minus_best_single_ci95')}")
        out["ablation_grid_max_ci"][bname] = {"rows": rows,
                                              "best_single": best_single}

    # --- 3. fusion-rule ablation for the actual ensemble (roadmap #8) ---
    print("\n=== 3. fusion rules on the full three-channel ensemble ===", flush=True)
    for bname, B in benches.items():
        labels = B["labels"]
        if any(B["mat"][ch2][i] is None for ch2 in CHANNELS for i in range(len(labels))):
            print(f"  {bname}: skipped (provenance incomplete)")
            out["fusion_rules"][bname] = {"note": "provenance channel has no labels for constructed benign"}
            continue
        best_single_auc = max(stats.auc(labels, B["mat"][c]) for c in CHANNELS)
        best_single_ch = max(CHANNELS, key=lambda c: stats.auc(labels, B["mat"][c]))
        rows = {}
        for rname, fused in [(r, fn([B["mat"][c] for c in CHANNELS]))
                             for r, fn in FUSIONS.items()] + \
                            [("lr", lr_oof_scores(B["mat"], labels))]:
            auc = stats.auc(labels, fused)
            row = {"auc": round(auc, 4), "auc_ci95": bootstrap_auc_ci(labels, fused, seed=0),
                   "minus_best_single_ci95": paired_diff_ci(labels, fused,
                                                            B["mat"][best_single_ch], seed=0)}
            row["tpr_at_7pct_fpr"] = round(stats.tpr_at_fpr(labels, fused, 0.07), 4)
            rows[rname] = row
            print(f"  {bname:12s} {rname:9s} AUC={auc:.4f} CI={row['auc_ci95']} "
                  f"vs_best({best_single_ch})={row['minus_best_single_ci95']} "
                  f"TPR@7%FPR={row['tpr_at_7pct_fpr']:.3f}")
        out["fusion_rules"][bname] = {"best_single": {"channel": best_single_ch,
                                                      "auc": round(best_single_auc, 4)},
                                      "rows": rows}

    Path("results/run_fusion_rule_ablation.json").write_text(json.dumps(out, indent=2))
    print("\n[written] results/run_fusion_rule_ablation.json")


if __name__ == "__main__":
    raise SystemExit(main())
