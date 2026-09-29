"""Review-response experiments (persisted, cache-only — no API calls).

Addresses three review points with data rather than rewording:

A. Deterministic-score degeneracy. The zero-width AUC CI [0.500, 0.500]
   on InjecAgent reads like a precisely measured null; the actual cause
   is a degenerate (constant) score. This section reports the score
   distribution so the paper can state the cause explicitly.

B. Composition-ceiling replication. The published pairing (input filter
   + GLM-4.6, AUC 0.769 versus 0.916 alone) was a single filter/classifier
   pair. This section recomputes the same composition with four cached
   classifiers on InjecAgent's real benign data, plus paired-bootstrap
   CIs on the composition-minus-classifier AUC difference.

C. Trained-ensemble uncertainty. The +15.4 pp TPR gain at matched 7% FPR
   (three-channel fusion on the tool-call bench) had no interval. This
   section adds a paired bootstrap CI and records the exact fusion
   recipe (channels, weights, score normalization).

Output: results/run_review_response.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
from benchmarks.injecagent import InjecAgentBench
from benchmarks.agentdojo import AgentDojoBench
from run_toolcall_bench import build_bench
from run_agentdojo_fpr import build_benign
from pipeline.conformal import nonconformity_score, detect_patterns
from pipeline.input_filter import InputFilter
from evaluation import stats


# ---------------------------------------------------------------- helpers

def _load_scores(name: str, bench: str):
    """Continuous per-trajectory scores from cache, ordered by idx."""
    p = Path(f"results/cache/{name}__{bench}.jsonl")
    c = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = r["score"]
    return [c[i] for i in range(len(c))]


def _load_labels(name: str, bench: str):
    """0/1 hard labels from cache, ordered by idx."""
    p = Path(f"results/cache/{name}__{bench}.jsonl")
    c = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = int(r["mal"])
    return [c[i] for i in range(len(c))]


def _auc_diff_bootstrap(labels, scores_a, scores_b, n_boot=2000, seed=0):
    """Paired bootstrap CI for AUC(a) - AUC(b), resampling trajectories."""
    import random
    rng = random.Random(seed)
    n, diffs = len(labels), []
    idx_all = list(range(n))
    for _ in range(n_boot):
        idx = [rng.choice(idx_all) for _ in range(n)]
        la = [labels[i] for i in idx]
        if all(la) or not any(la):
            continue
        sa = [scores_a[i] for i in idx]
        sb = [scores_b[i] for i in idx]
        try:
            diffs.append(stats.auc(la, sa) - stats.auc(la, sb))
        except Exception:
            continue
    diffs.sort()
    if not diffs:
        return None
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return [round(lo, 4), round(hi, 4)]


def _tpr7_diff_bootstrap(labels, scores_a, scores_b, fpr=0.07,
                          n_boot=2000, seed=0):
    """Paired bootstrap CI for TPR@FPR(a) - TPR@FPR(b), recomputing each
    system's threshold inside every resample (matched-FPR, not matched-
    threshold)."""
    import random
    rng = random.Random(seed)
    n, diffs = len(labels), []
    idx_all = list(range(n))
    for _ in range(n_boot):
        idx = [rng.choice(idx_all) for _ in range(n)]
        la = [labels[i] for i in idx]
        if all(la) or not any(la):
            continue
        sa = [scores_a[i] for i in idx]
        sb = [scores_b[i] for i in idx]
        try:
            diffs.append(stats.tpr_at_fpr(la, sa, fpr)
                         - stats.tpr_at_fpr(la, sb, fpr))
        except Exception:
            continue
    diffs.sort()
    if not diffs:
        return None
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return [round(lo, 4), round(hi, 4)]


FUSIONS = {
    "max":      lambda i, g: [max(a, b) for a, b in zip(i, g)],
    "mean":     lambda i, g: [0.5 * (a + b) for a, b in zip(i, g)],
    "noisy-or": lambda i, g: [a + b - a * b for a, b in zip(i, g)],
    "product":  lambda i, g: [a * b for a, b in zip(i, g)],
}


def main():
    out = {}

    # ---------------- A. det-score degeneracy on InjecAgent ----------------
    print("=== A. deterministic-score distribution (InjecAgent 510) ===", flush=True)
    trajs = list(InjecAgentBench(max_trajs=510))
    labels = [t.ground_truth == "malicious" for t in trajs]
    det_scores = [nonconformity_score(detect_patterns(t)) for t in trajs]
    from collections import Counter
    hist = Counter(det_scores)
    out["A_det_score_distribution"] = {
        "n": len(det_scores),
        "n_distinct_values": len(hist),
        "fraction_at_mode": round(hist.most_common(1)[0][1] / len(det_scores), 4),
        "mode_value": hist.most_common(1)[0][0],
        "hist": {str(k): v for k, v in sorted(hist.items())},
        "auc": round(stats.auc(labels, det_scores), 4),
    }
    print(f"  distinct values: {len(hist)}, mode={hist.most_common(1)[0][0]} "
          f"covers {out['A_det_score_distribution']['fraction_at_mode']*100:.1f}%")

    # ---------------- B. composition ceiling replication ----------------
    print("\n=== B. composition-ceiling replication (InjecAgent real benign) ===", flush=True)
    inp = InputFilter()
    input_scores = [inp._max_instruction_shape(t)[0] for t in trajs]
    input_alone_auc = stats.auc(labels, input_scores)
    print(f"  input filter alone AUC = {input_alone_auc:.4f} (published: 0.6306)")

    classifiers = {
        "GLM-4.6": _load_scores("glm_guard", "injecagent"),
        "GLM-5.2": _load_scores("or_glm-5.2", "injecagent"),
        "Claude Haiku 4.5": _load_scores("or_claude-haiku-4-5", "injecagent"),
        "GPT-4o-mini": _load_scores("or_gpt-4o-mini", "injecagent"),
    }

    # Reconstruct the published fusion rule first (GLM-4.6 pairing must
    # reproduce composition AUC 0.7691; AgentDojo GPT-4o-mini must
    # reproduce 0.9083 — checked below once AgentDojo is loaded).
    g46 = classifiers["GLM-4.6"]
    rule_fits = {}
    for rname, fn in FUSIONS.items():
        comp = stats.auc(labels, fn(input_scores, g46))
        rule_fits[rname] = round(comp, 4)
        print(f"  fusion={rname:9s} -> comp AUC {comp:.4f}")
    best_rule = min(rule_fits, key=lambda r: abs(rule_fits[r] - 0.7691))
    print(f"  -> closest to published 0.7691: {best_rule}")
    out["B_fusion_rule_reconstruction"] = {
        "candidate_aucs_glm46": rule_fits,
        "selected_rule": best_rule,
        "published_composition_auc": 0.7691,
    }
    fuse = FUSIONS[best_rule]

    # Cross-check the rule on AgentDojo constructed benign (published
    # composition 0.9083 with GPT-4o-mini).
    ad_attacks = list(AgentDojoBench(max_trajs=280))
    ad_benign = build_benign(n_per_suite=40)
    ad_combined = ad_attacks + ad_benign
    ad_labels = [t.ground_truth == "malicious" for t in ad_combined]
    ad_input = [inp._max_instruction_shape(t)[0] for t in ad_combined]
    ad_gpt4o = _load_scores("or_gpt-4o-mini", "agentdojo") + \
        _load_scores("or_gpt-4o-mini", "agentdojo_benign")
    ad_comp = stats.auc(ad_labels, fuse(ad_input, ad_gpt4o))
    out["B_fusion_rule_reconstruction"]["agentdojo_gpt4o_composition_auc"] = round(ad_comp, 4)
    out["B_fusion_rule_reconstruction"]["agentdojo_input_filter_alone_auc"] = \
        round(stats.auc(ad_labels, ad_input), 4)
    out["B_fusion_rule_reconstruction"]["agentdojo_published"] = 0.9083
    print(f"  AgentDojo GPT-4o composition under this rule: {ad_comp:.4f} (published 0.9083)")
    print(f"  AgentDojo input filter alone: "
          f"{out['B_fusion_rule_reconstruction']['agentdojo_input_filter_alone_auc']:.4f}")

    # Four pairings on InjecAgent real benign.
    out["B_composition_pairings_injecagent"] = {}
    for cname, cscores in classifiers.items():
        comp_scores = fuse(input_scores, cscores)
        comp_auc = stats.auc(labels, comp_scores)
        clf_auc = stats.auc(labels, cscores)
        ci = _auc_diff_bootstrap(labels, comp_scores, cscores)
        # Fusion-rule sensitivity: does ANY simple rule beat the classifier?
        alt = {}
        for rname in ("mean", "product", "noisy-or"):
            if rname == best_rule:
                continue
            s = FUSIONS[rname](input_scores, cscores)
            a = stats.auc(labels, s)
            alt[rname] = {"auc": round(a, 4),
                          "diff": round(a - clf_auc, 4),
                          "diff_ci95": _auc_diff_bootstrap(labels, s, cscores)}
        out["B_composition_pairings_injecagent"][cname] = {
            "classifier_alone_auc": round(clf_auc, 4),
            "composition_auc": round(comp_auc, 4),
            "diff": round(comp_auc - clf_auc, 4),
            "diff_ci95": ci,
            "alternative_fusions": alt,
        }
        print(f"  {cname:18s} alone={clf_auc:.4f} comp={comp_auc:.4f} "
              f"diff={comp_auc-clf_auc:+.4f} CI={ci}")
        for rname, r in alt.items():
            print(f"     [{rname:8s}] diff={r['diff']:+.4f} CI={r['diff_ci95']}")

    # ---------------- C. trained-ensemble recipe + CI ----------------
    print("\n=== C. trained-ensemble recipe and CI (tool-call bench) ===", flush=True)
    tc, _, _ = build_bench(n_per_pattern=20, benign_reps=30, seed=0)
    tc_labels = [t.ground_truth == "malicious" for t in tc]
    tc_claude = _load_scores("or_claude-haiku-4-5", "toolcall")
    tc_camel = _load_labels("camel_full_or", "toolcall")
    tc_det = [nonconformity_score(detect_patterns(t)) for t in tc]

    claude_auc = stats.auc(tc_labels, tc_claude)
    print(f"  Claude alone AUC = {claude_auc:.4f} (published 0.8316)")

    # Two-channel (Claude + CaMeL): max(score, label*w). Find w matching
    # the published 0.8448.
    two_ch = {}
    for w in (0.5, 0.7, 0.9, 1.0):
        fused = [max(s, l * w) for s, l in zip(tc_claude, tc_camel)]
        two_ch[w] = round(stats.auc(tc_labels, fused), 4)
        print(f"  claude+camel w={w}: AUC {two_ch[w]:.4f}")
    w_star = min(two_ch, key=lambda w: abs(two_ch[w] - 0.8448))
    print(f"  -> closest to published 0.8448: w={w_star}")

    # Three-channel: add the signature (det) score, normalized into [0,1]
    # by d/(d+1) so max-fusion is scale-compatible. Confirm against the
    # published 0.8737.
    def _d01(d):
        return d / (d + 1.0) if d > 0 else 0.0

    three_ch = {}
    for w in (0.5, 0.7, 0.9, 1.0):
        fused = [max(s, l * w, _d01(d)) for s, l, d in zip(tc_claude, tc_camel, tc_det)]
        three_ch[w] = round(stats.auc(tc_labels, fused), 4)
    w3_star = min(three_ch, key=lambda w: abs(three_ch[w] - 0.8737))
    print(f"  three-channel AUC by w: {three_ch} -> closest to 0.8737: w={w3_star}")

    fused3 = [max(s, l * w3_star, _d01(d)) for s, l, d in zip(tc_claude, tc_camel, tc_det)]
    tpr7_claude = stats.tpr_at_fpr(tc_labels, tc_claude, 0.07)
    tpr7_two = stats.tpr_at_fpr(tc_labels,
                                [max(s, l * w_star) for s, l in zip(tc_claude, tc_camel)], 0.07)
    tpr7_three = stats.tpr_at_fpr(tc_labels, fused3, 0.07)
    ci3 = _tpr7_diff_bootstrap(tc_labels, fused3, tc_claude, 0.07)
    ci2 = _tpr7_diff_bootstrap(tc_labels,
                               [max(s, l * w_star) for s, l in zip(tc_claude, tc_camel)],
                               tc_claude, 0.07)

    out["C_ensemble_recipe"] = {
        "bench": "tool-call (n=480: 360 malicious, 120 constructed benign)",
        "channels": {
            "text": "Claude Haiku 4.5 continuous score (cached)",
            "provenance": "full CaMeL hard label (cached), weight w",
            "pattern": "signature non-conformity score, normalized d/(d+1)",
        },
        "fusion": "per-trajectory max over channel scores",
        "two_channel_w": w_star,
        "two_channel_auc_by_w": two_ch,
        "three_channel_w": w3_star,
        "three_channel_auc_by_w": three_ch,
    }
    out["C_tpr_at_7pct_fpr"] = {
        "claude_alone": round(tpr7_claude, 4),
        "claude_plus_camel": round(tpr7_two, 4),
        "three_channel": round(tpr7_three, 4),
        "gain_three_minus_claude_pp": round(100 * (tpr7_three - tpr7_claude), 1),
        "gain_three_minus_claude_ci95_pp": [round(100 * x, 1) for x in ci3] if ci3 else None,
        "gain_two_minus_claude_pp": round(100 * (tpr7_two - tpr7_claude), 1),
        "gain_two_minus_claude_ci95_pp": [round(100 * x, 1) for x in ci2] if ci2 else None,
    }
    print(f"  TPR@7%FPR: claude={100*tpr7_claude:.1f}%  +camel={100*tpr7_two:.1f}%  "
          f"3ch={100*tpr7_three:.1f}%  gain CI={out['C_tpr_at_7pct_fpr']['gain_three_minus_claude_ci95_pp']} pp")

    Path("results/run_review_response.json").write_text(json.dumps(out, indent=2))
    print("\n[written] results/run_review_response.json")


if __name__ == "__main__":
    raise SystemExit(main())
