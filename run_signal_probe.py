"""Stage-1 signal-informativeness probe.

The engineered cascade can only beat a learned baseline (GLM-4.6) on a
benchmark if some *cheap* Stage-1 signal is informative on that
benchmark's attack distribution. On InjecAgent (natural-language
injection) the hard-label det layer fires on 0% of attacks — but its
*continuous* non-conformity score, and the classical-ML probabilities,
may still carry signal.

This probe measures, on the FULL InjecAgent 510 (the only benchmark with
benign controls), the discriminative power of each cheap signal:

  * det non-conformity score   (pipeline.conformal.nonconformity_score)
  * GBDT malicious probability (baselines.gbdt_features)
  * embedding-kNN raw score    (baselines.embedding_knn)

Reports AUC and TPR @ {1%, 5%, 10%} FPR for each. The honest-fork rule
in the plan: if the BEST cheap signal has AUC > ~0.6 on InjecAgent, a
cascade can plausibly beat GLM-alone there; otherwise the cascade's win
is on cost + signature-attack benches, and we say so honestly.

Output: results/run_signal_probe.json + a printed table.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.injecagent import InjecAgentBench
from evaluation import stats


def det_nonconformity_scores(trajs):
    """Raw det non-conformity score per trajectory (higher = more malicious)."""
    from pipeline.conformal import nonconformity_score, detect_patterns
    out = []
    for t in trajs:
        ctx = detect_patterns(t)
        out.append(nonconformity_score(ctx))
    return out


def gbdt_mal_scores(trajs):
    """GBDT P(malicious) reconstructed from the verdict confidence.

    gbdt sets confidence = proba_mal (if MALICIOUS) or 1 - proba_mal (if
    BENIGN), so proba_mal is confidence if malicious else 1 - confidence.
    """
    from baselines.gbdt_features import GbdtFeaturesVerifier
    from schemas import Decision
    v = GbdtFeaturesVerifier()
    out = []
    for t in trajs:
        verdict = v.verify(t)
        if verdict.decision is Decision.MALICIOUS:
            out.append(float(verdict.confidence))
        else:
            out.append(1.0 - float(verdict.confidence))
    return out


def embedding_knn_scores(trajs):
    """embedding-kNN raw (mal_mean - ben_mean) score, parsed from notes."""
    from baselines.embedding_knn import EmbeddingKnnVerifier
    v = EmbeddingKnnVerifier()
    if not getattr(v, "_available", False):
        return None
    pat = re.compile(r"score=(-?[\d.]+)")
    out = []
    for t in trajs:
        verdict = v.verify(t)
        note = verdict.notes[0] if verdict.notes else ""
        m = pat.search(note)
        out.append(float(m.group(1)) if m else 0.0)
    return out


def combined_max(det_s, gbdt_s, knn_s):
    """A trivial late-fusion baseline: min-max normalize each signal to
    [0,1], then take the max. A weak but parameter-free combination."""
    def norm(xs):
        if not xs: return xs
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-12: return [0.5] * len(xs)
        return [(x - lo) / (hi - lo) for x in xs]
    d, g, k = norm(det_s), norm(gbdt_s), norm(knn_s or [0.0] * len(det_s))
    return [max(a, b, c) for a, b, c in zip(d, g, k)]


def evaluate_signal(name, labels, scores):
    if scores is None:
        return {"signal": name, "status": "unavailable"}
    try:
        a = stats.auc(labels, scores)
    except ValueError as e:
        return {"signal": name, "status": f"auc_undefined ({e})"}
    return {
        "signal": name,
        "n": len(labels),
        "n_mal": int(sum(labels)),
        "n_ben": int(sum(1 for l in labels if not l)),
        "auc": round(a, 4),
        "tpr_at_1pct_fpr": round(stats.tpr_at_fpr(labels, scores, 0.01), 4),
        "tpr_at_5pct_fpr": round(stats.tpr_at_fpr(labels, scores, 0.05), 4),
        "tpr_at_10pct_fpr": round(stats.tpr_at_fpr(labels, scores, 0.10), 4),
    }


def main():
    print("[loader] InjecAgent full 510 ...", flush=True)
    t0 = time.perf_counter()
    trajs = list(InjecAgentBench(max_trajs=510))
    print(f"[info] loaded {len(trajs)} trajectories in {time.perf_counter()-t0:.1f}s", flush=True)
    labels = [t.ground_truth == "malicious" for t in trajs]
    n_mal = sum(labels)
    print(f"[info] {n_mal} malicious / {len(labels)-n_mal} benign", flush=True)

    print("[probe] det non-conformity scores ...", flush=True)
    det_s = det_nonconformity_scores(trajs)
    print("[probe] GBDT P(mal) scores ...", flush=True)
    gbdt_s = gbdt_mal_scores(trajs)
    print("[probe] embedding-kNN scores ...", flush=True)
    knn_s = embedding_knn_scores(trajs)
    print("[probe] combined (max-fusion) scores ...", flush=True)
    comb_s = combined_max(det_s, gbdt_s, knn_s)

    results = [
        evaluate_signal("det-nonconformity", labels, det_s),
        evaluate_signal("gbdt-proba", labels, gbdt_s),
        evaluate_signal("embedding-knn", labels, knn_s),
        evaluate_signal("combined-max", labels, comb_s),
    ]

    # AUC bootstrap CIs for the two strongest (to gauge significance).
    for r in results:
        if "auc" not in r: continue
        sig_name = r["signal"]
        sc = {"det-nonconformity": det_s, "gbdt-proba": gbdt_s,
              "embedding-knn": knn_s, "combined-max": comb_s}[sig_name]
        if sc is None: continue
        pt, lo, hi = stats.bootstrap_ci(labels, sc, lambda lb, sb: stats.auc(lb, sb),
                                        n_boot=2000, seed=0)
        r["auc_ci95"] = [round(lo, 4), round(hi, 4)]

    out = {
        "bench": "InjecAgent full 510",
        "n_malicious": n_mal,
        "n_benign": len(labels) - n_mal,
        "signals": results,
        "honest_fork_rule": "cascade can beat GLM-alone on InjecAgent iff best cheap signal AUC > ~0.6",
    }
    out_path = Path("results/run_signal_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[written] {out_path}\n")

    print("=" * 78)
    print(f"{'signal':22s} {'AUC':>7s} {'95% CI':>16s} "
          f"{'TPR@1%':>7s} {'TPR@5%':>7s} {'TPR@10%':>8s}")
    print("-" * 78)
    for r in results:
        if "auc" not in r:
            print(f"{r['signal']:22s}   {r.get('status','?')}")
            continue
        ci = r.get("auc_ci95", [None, None])
        cistr = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci[0] is not None else ""
        print(f"{r['signal']:22s} {r['auc']:7.3f} {cistr:>16s} "
              f"{r['tpr_at_1pct_fpr']*100:6.1f}% {r['tpr_at_5pct_fpr']*100:6.1f}% "
              f"{r['tpr_at_10pct_fpr']*100:7.1f}%")
    print("=" * 78)
    best = max((r for r in results if "auc" in r), key=lambda r: r["auc"], default=None)
    if best:
        verdict = "GO — a cheap signal is informative on InjecAgent" if best["auc"] > 0.6 \
                  else "NO-GO — cheap signals are uninformative on NL injection; cascade win is on cost + signature benches"
        print(f"\nBest cheap signal: {best['signal']} AUC={best['auc']:.3f}")
        print(f"Honest-fork verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
