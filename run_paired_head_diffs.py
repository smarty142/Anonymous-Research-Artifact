"""Paired-difference CIs comparing the learned head and the
taxonomy-independent subset against the text channel alone.

Zero API calls: everything derives from the shipped caches and the
local pattern channel, exactly as in run_ensemble_ablation.py.

Compared systems (paired on identical trajectories, n_boot=2000):

  1. learned head (pooled out-of-fold) vs text, per benchmark
  2. zero-shot transfer head vs destination text, both directions
  3. text+provenance max-fusion vs text on the tool-call bench
     (the taxonomy-independent part of the headline gain)

Writes results/run_paired_head_diffs.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from evaluation import stats
from run_ensemble_ablation import build_benchmark_matrices, CHANNELS

N_BOOT = 2000


def _Xy(B):
    X = np.asarray([[B["mat"][c][i] for c in CHANNELS]
                    for i in range(len(B["labels"]))], dtype=float)
    return X, np.asarray(B["labels"], dtype=int)


def _fit_full(B):
    X, y = _Xy(B)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)
    return sc, lr


def _oof(B, seed=0):
    X, y = _Xy(B)
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.decision_function(sc.transform(X[te]))
    return oof


def _paired(labels, base, sys_b, stat):
    d, lo, hi = stats.bootstrap_paired_diff(labels, base, sys_b,
                                            stat_fn=stat,
                                            n_boot=N_BOOT, seed=0)
    return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main():
    B = build_benchmark_matrices()
    tpr7 = lambda l, s: stats.tpr_at_fpr(l, s, 0.07)
    out = {"n_boot": N_BOOT, "comparisons": {}}

    # 1. in-domain learned head (pooled OOF) vs text
    for bench in ("injecagent", "toolcall"):
        y = B[bench]["labels"]
        text = B[bench]["mat"]["text"]
        oof = _oof(B[bench]).tolist()
        out["comparisons"][f"{bench}_lr_oof_vs_text"] = {
            "lr_oof_auc": round(stats.auc(y, oof), 4),
            "text_auc": round(stats.auc(y, text), 4),
            "auc_diff": _paired(y, text, oof, stats.auc),
            "tpr7_diff": _paired(y, text, oof, tpr7),
        }
        print(f"{bench}: lr_oof {out['comparisons'][f'{bench}_lr_oof_vs_text']['lr_oof_auc']:.4f}"
              f" vs text {out['comparisons'][f'{bench}_lr_oof_vs_text']['text_auc']:.4f}"
              f"  auc_diff {out['comparisons'][f'{bench}_lr_oof_vs_text']['auc_diff']}")

    # 2. zero-shot transfer heads vs destination text
    for src, dst in (("toolcall", "injecagent"),
                     ("injecagent", "toolcall")):
        sc, lr = _fit_full(B[src])
        Xd, yd = _Xy(B[dst])
        preds = lr.decision_function(sc.transform(Xd)).tolist()
        text = B[dst]["mat"]["text"]
        out["comparisons"][f"transfer_{src}_to_{dst}_vs_text"] = {
            "lr_auc": round(stats.auc(yd, preds), 4),
            "text_auc": round(stats.auc(yd, text), 4),
            "auc_diff": _paired(yd, text, preds, stats.auc),
        }
        k = f"transfer_{src}_to_{dst}_vs_text"
        print(f"{src}->{dst}: lr {out['comparisons'][k]['lr_auc']:.4f}"
              f" vs text {out['comparisons'][k]['text_auc']:.4f}"
              f"  {out['comparisons'][k]['auc_diff']}")

    # 3. taxonomy-independent subset vs text on the tool-call bench
    Bt = B["toolcall"]
    y = Bt["labels"]
    tp = np.maximum(np.asarray(Bt["mat"]["text"]),
                    np.asarray(Bt["mat"]["provenance"])).tolist()
    text = Bt["mat"]["text"]
    out["comparisons"]["toolcall_text_prov_vs_text"] = {
        "text_prov_auc": round(stats.auc(y, tp), 4),
        "text_auc": round(stats.auc(y, text), 4),
        "auc_diff": _paired(y, text, tp, stats.auc),
        "tpr7_diff": _paired(y, text, tp, tpr7),
    }
    k = "toolcall_text_prov_vs_text"
    print(f"toolcall text+prov {out['comparisons'][k]['text_prov_auc']:.4f}"
          f" vs text {out['comparisons'][k]['text_auc']:.4f}"
          f"  auc {out['comparisons'][k]['auc_diff']}"
          f"  tpr7 {out['comparisons'][k]['tpr7_diff']}")

    Path("results/run_paired_head_diffs.json").write_text(
        json.dumps(out, indent=2))
    print("\n[written] results/run_paired_head_diffs.json")


if __name__ == "__main__":
    main()
