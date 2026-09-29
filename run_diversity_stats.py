"""Ensemble-diversity statistics for the TriChannel channels + backbone
score correlations (review weaknesses #4 and the R3 correlated-errors
confound; zero API, cache-only).

Two blocks:

1. Channel diversity. For each benchmark, each channel is thresholded at
   its own 7%-FPR operating point, then every pair of channels gets
   Kuncheva diversity measures over their correctness patterns
   (Q-statistic, disagreement rate, double-fault rate) plus Spearman
   rank correlation between raw channel scores. Q near 0 and high
   disagreement = genuinely complementary channels; Q >> 0 or high
   double-fault = correlated errors (fusion adds little; off-regime
   failure propagates). Provenance is skipped where it lacks labels
   (AgentDojo constructed benign).

2. Backbone correlation. Spearman correlation between the text channel's
   backbone variants (claude-haiku-4-5, glm-5.2, gpt-4o-mini, glm-4.6
   guard) on the same trajectories — a zero-cost preview of the shared-
   backbone confound: if backbones within one channel are near-perfectly
   correlated, per-channel error structure is about the *channel*, not
   the specific model.

Output: results/run_diversity_stats.json (+ printed markdown table)
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, ".")

from evaluation import stats
from run_fusion_rule_ablation import build_benchmark_matrices, CHANNELS

FPR_BUDGET = 0.07

BACKBONES = [
    ("claude-haiku-4-5", "or_claude-haiku-4-5"),
    ("glm-5.2", "or_glm-5.2"),
    ("gpt-4o-mini", "or_gpt-4o-mini"),
    ("glm-4.6-guard", "glm_guard"),
]


def _load_scores(name: str, bench: str):
    p = Path(f"results/cache/{name}__{bench}.jsonl")
    c = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = r["score"]
    return [c[i] for i in range(len(c))]


def _r4(x):
    return None if x is None else round(float(x), 4)


def channel_diversity(benches):
    out = {}
    for bname, B in benches.items():
        labels = B["labels"]
        preds, note = {}, None
        for ch in CHANNELS:
            sc = B["mat"][ch]
            if any(s is None for s in sc):
                note = "provenance channel has no labels for constructed benign"
                continue
            th = stats.threshold_at_fpr(labels, sc, FPR_BUDGET)
            preds[ch] = stats.apply_threshold(sc, th)
        rows = {}
        for a, b in combinations(sorted(preds), 2):
            rows[f"{a}|{b}"] = {
                "q_statistic": _r4(stats.q_statistic(labels, preds[a], preds[b])),
                "disagreement_rate": _r4(stats.disagreement_rate(preds[a], preds[b])),
                "double_fault_rate": _r4(stats.double_fault_rate(labels, preds[a], preds[b])),
                "spearman_raw_scores": _r4(stats.spearman(B["mat"][a], B["mat"][b])),
            }
        out[bname] = {"n": len(labels), "rows": rows}
        if note:
            out[bname]["note"] = note
        print(f"\n### {bname} (n={len(labels)})" + (f" — {note}" if note else ""))
        print("| pair | Q | disagreement | double-fault | spearman |")
        print("|---|---|---|---|---|")
        for k, v in rows.items():
            q = "n/a" if v["q_statistic"] is None else f"{v['q_statistic']:.3f}"
            print(f"| {k.replace('|', ' vs ')} | {q} | {v['disagreement_rate']:.3f} "
                  f"| {v['double_fault_rate']:.3f} | {v['spearman_raw_scores']:.3f} |")
    return out


def backbone_correlation():
    bench_files = {
        "injecagent": ["injecagent"],
        "toolcall": ["toolcall"],
        "agentdojo": ["agentdojo", "agentdojo_benign"],
    }
    out = {}
    for bname, parts in bench_files.items():
        scores = {}
        for label, cache in BACKBONES:
            try:
                scores[label] = sum((_load_scores(cache, p) for p in parts), [])
            except FileNotFoundError:
                pass
        rows = {}
        for a, b in combinations(sorted(scores), 2):
            rows[f"{a}|{b}"] = _r4(stats.spearman(scores[a], scores[b]))
        out[bname] = {"n_backbones": len(scores), "rows": rows}
        print(f"\n### backbone spearman ({bname}, {len(scores)} backbones)")
        for k, v in rows.items():
            print(f"  {k.replace('|', ' vs ')}: {v:.3f}")
    return out


def main():
    benches = build_benchmark_matrices()
    out = {
        "fpr_budget": FPR_BUDGET,
        "channel_diversity": channel_diversity(benches),
        "backbone_correlation": backbone_correlation(),
    }
    Path("results/run_diversity_stats.json").write_text(json.dumps(out, indent=2))
    print("\n[written] results/run_diversity_stats.json")


if __name__ == "__main__":
    raise SystemExit(main())
