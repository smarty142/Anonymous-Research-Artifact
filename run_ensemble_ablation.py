"""TriChannel ensemble: ablation grid, learned-fusion baseline, transfer.

ICLR-restructure experiments (cache-only, no API calls):

A. Ablation grid. For every non-empty subset of {text, provenance,
   pattern} channels, evaluate max-fusion on each benchmark:
   InjecAgent (real benign), tool-call (action-shaped, constructed
   benign), AgentDojo (constructed benign; provenance channel has no
   labels for constructed benign, so provenance subsets are reported
   only where computable).

B. Learned fusion baseline. Logistic regression over the three channel
   scores, 5-fold stratified CV, compared against parameter-free
   max-fusion — tests whether the gain needs learning or just needs
   the right channels.

C. Cross-benchmark transfer. Train the LR fusion on one benchmark,
   evaluate zero-shot on the other two. Max-fusion needs no training,
   so it transfers by construction; the LR test asks whether learned
   weights transfer across attack distributions.

Channels per benchmark:
  text       = Claude Haiku 4.5 continuous score (cache)
  provenance = full-CaMeL hard label (cache; 0/1)
  pattern    = signature non-conformity score, normalized d/(d+1) (local)

Output: results/run_ensemble_ablation.json
"""
from __future__ import annotations
import json, sys
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


def _load_scores(name: str, bench: str):
    p = Path(f"results/cache/{name}__{bench}.jsonl")
    c = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = r["score"]
    return [c[i] for i in range(len(c))]


def _load_hard(name: str, bench: str):
    p = Path(f"results/cache/{name}__{bench}.jsonl")
    c = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c[int(r["idx"])] = float(r["mal"])
    return [c[i] for i in range(len(c))]


def _d01(d: float) -> float:
    return d / (d + 1.0) if d > 0 else 0.0


def _maxfuse(mat, subset):
    """Per-trajectory max over the selected channels (mat: channel -> scores)."""
    return [max(mat[c][i] for c in subset) for i in range(len(mat[subset[0]]))]


def _bootstrap_ci(labels, scores_a, scores_b, n_boot=2000, seed=0):
    """Paired bootstrap CI for AUC(a) - AUC(b)."""
    import random
    rng = random.Random(seed)
    n, diffs = len(labels), []
    idx_all = list(range(n))
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


def build_benchmark_matrices():
    """Return {bench: {"labels": [...], "mat": {channel: [scores]}, "n": int}}."""
    out = {}

    # --- InjecAgent (510, real benign) ---
    trajs = list(InjecAgentBench(max_trajs=510))
    out["injecagent"] = {
        "labels": [t.ground_truth == "malicious" for t in trajs],
        "mat": {
            "text": _load_scores("or_claude-haiku-4-5", "injecagent"),
            "provenance": _load_hard("camel_full_or", "injecagent"),
            "pattern": [_d01(nonconformity_score(detect_patterns(t))) for t in trajs],
        },
    }

    # --- tool-call (480: action-shaped attacks + constructed benign) ---
    tc, _, _ = build_bench(n_per_pattern=20, benign_reps=30, seed=0)
    out["toolcall"] = {
        "labels": [t.ground_truth == "malicious" for t in tc],
        "mat": {
            "text": _load_scores("or_claude-haiku-4-5", "toolcall"),
            "provenance": _load_hard("camel_full_or", "toolcall"),
            "pattern": [_d01(nonconformity_score(detect_patterns(t))) for t in tc],
        },
    }

    # --- AgentDojo (280 attacks + 120 constructed benign) ---
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


def lr_cv_auc(X, y, n_splits=5, seed=0):
    """Stratified 5-fold CV AUC of LR over channel scores."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        # decision_function = w.x + b, monotone in P(malicious)
        preds = lr.decision_function(sc.transform(X[te]))
        labels_te = y[te].tolist()
        if all(labels_te) or not any(labels_te):
            continue
        aucs.append(stats.auc(labels_te, preds.tolist()))
    if not aucs:
        return None
    return {"mean": round(float(np.mean(aucs)), 4),
            "std": round(float(np.std(aucs)), 4), "folds": aucs}


def main():
    benches = build_benchmark_matrices()
    out = {"ablation": {}, "lr_fusion": {}, "transfer": {}}

    # ---------------- A. ablation grid ----------------
    print("=== A. ablation grid (max-fusion over channel subsets) ===", flush=True)
    for bname, B in benches.items():
        labels = B["labels"]
        n_mal, n_ben = sum(labels), len(labels) - sum(labels)
        rows = {}
        for k in (1, 2, 3):
            for subset in combinations(CHANNELS, k):
                if any(B["mat"][c][i] is None for i in range(len(labels))
                       for c in subset):
                    rows["+".join(subset)] = {"auc": None,
                                              "note": "provenance channel has no "
                                                      "labels for constructed benign"}
                    continue
                fused = _maxfuse(B["mat"], subset)
                auc = round(stats.auc(labels, fused), 4)
                tpr7 = round(stats.tpr_at_fpr(labels, fused, 0.07), 4)
                rows["+".join(subset)] = {"auc": auc, "tpr_at_7pct_fpr": tpr7}
        # CI of the full ensemble vs best single channel
        best_single = max((v["auc"], k) for k, v in rows.items()
                          if v["auc"] is not None and "+" not in k)
        full = rows.get("text+provenance+pattern", {}).get("auc")
        ci = None
        if full is not None:
            ci = _bootstrap_ci(labels, _maxfuse(B["mat"], CHANNELS),
                               B["mat"][best_single[1]])
        out["ablation"][bname] = {"n_malicious": n_mal, "n_benign": n_ben,
                                  "rows": rows,
                                  "full_minus_best_single_ci95": ci,
                                  "best_single": {"channel": best_single[1],
                                                  "auc": best_single[0]}}
        print(f"\n  {bname} (mal={n_mal}, ben={n_ben})")
        for k, v in rows.items():
            a = f"{v['auc']:.4f}" if v["auc"] is not None else "  n/a"
            t = f"{100*v['tpr_at_7pct_fpr']:5.1f}%" if v.get("tpr_at_7pct_fpr") is not None else "  n/a"
            print(f"    {k:26s} AUC={a}  TPR@7%={t}")
        if ci:
            print(f"    full-vs-best-single CI: {ci}")

    # ---------------- B. learned fusion (LR) vs max-fusion ----------------
    print("\n=== B. learned fusion (logistic regression, 5-fold CV) ===", flush=True)
    models = {}
    for bname in ("toolcall", "injecagent"):
        B = benches[bname]
        X = [[B["mat"][c][i] for c in CHANNELS] for i in range(len(B["labels"]))]
        cv = lr_cv_auc(X, B["labels"])
        lr = LogisticRegression(max_iter=1000)
        sc = StandardScaler().fit(np.asarray(X, dtype=float))
        lr.fit(sc.transform(np.asarray(X, dtype=float)),
               np.asarray(B["labels"], dtype=int))
        models[bname] = (sc, lr)
        mx = out["ablation"][bname]["rows"]["text+provenance+pattern"]["auc"]
        out["lr_fusion"][bname] = {"cv_auc": cv,
                                   "max_fusion_auc": mx,
                                   "coefficients": dict(zip(CHANNELS,
                                                            [round(float(w), 4) for w in lr.coef_[0]]))}
        print(f"  {bname}: LR CV AUC = {cv['mean']:.4f} +/- {cv['std']:.4f}"
              f"   max-fusion AUC = {mx:.4f}   coefs={out['lr_fusion'][bname]['coefficients']}")

    # ---------------- C. cross-benchmark transfer ----------------
    print("\n=== C. cross-benchmark transfer (train here -> evaluate there) ===", flush=True)
    for src, dsts in [("toolcall", ["injecagent", "agentdojo"]),
                      ("injecagent", ["toolcall", "agentdojo"])]:
        for dst in dsts:
            B = benches[dst]
            if any(B["mat"][c][i] is None for i in range(len(B["labels"]))
                   for c in CHANNELS):
                # provenance missing on agentdojo benign: evaluate LR on the
                # subset with complete channels (attacks only is label-degenerate,
                # so report as not computable)
                out["transfer"][f"{src}->{dst}"] = {"lr_auc": None,
                                                    "note": "destination provenance incomplete"}
                print(f"  {src} -> {dst}: n/a (provenance incomplete at destination)")
                continue
            sc, lr = models[src]
            X = np.asarray([[B["mat"][c][i] for c in CHANNELS]
                            for i in range(len(B["labels"]))], dtype=float)
            preds = lr.decision_function(sc.transform(X)).tolist()
            auc = round(stats.auc(B["labels"], preds), 4)
            mx = out["ablation"][dst]["rows"]["text+provenance+pattern"]["auc"]
            best = out["ablation"][dst]["best_single"]["auc"]
            out["transfer"][f"{src}->{dst}"] = {"lr_auc": auc,
                                                "max_fusion_auc": mx,
                                                "best_single_auc": best}
            print(f"  {src} -> {dst}: LR AUC = {auc:.4f}  "
                  f"(max-fusion {mx:.4f}, best single {best:.4f})")

    Path("results/run_ensemble_ablation.json").write_text(json.dumps(out, indent=2))
    print("\n[written] results/run_ensemble_ablation.json")


if __name__ == "__main__":
    raise SystemExit(main())
