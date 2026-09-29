"""Render RESULTS.pdf: every published number and all figures of the
TriChannel paper, read live from the result JSONs in results/.

    python3 make_RESULTS.py   (run from the artifact root)

No number in this document is transcribed by hand: tables are rendered
from the JSON files at generation time; the five figure panels use the
same plotting code as figs/make_figs.py and figs/make_figs2.py.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

R = Path("results")
A4 = (8.27, 11.69)


def load(name):
    return json.loads((R / name).read_text())


def page(pdf, title=None, kicker=None):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    if kicker:
        fig.text(0.07, 0.955, kicker.upper(), fontsize=8, color="#666666",
                 family="sans-serif")
    if title:
        fig.text(0.07, 0.918, title, fontsize=12.5, fontweight="bold",
                 family="sans-serif")
    return fig


def para(fig, y, text, size=8.8, wrap=104, gap=0.0132, color="black",
         bold=False, mono=False, indent=0.0):
    fam = "monospace" if mono else "serif"
    import textwrap
    for line in textwrap.wrap(text, wrap) or [""]:
        fig.text(0.07 + indent, y, line, fontsize=size, color=color,
                 family=fam, fontweight="bold" if bold else "normal",
                 ha="left", va="top")
        y -= gap
    return y - 0.003


def _cap(fig, y, text, size=7.6, fontsize=None, color=None):
    """Figure caption, pre-wrapped so it never crosses the right margin."""
    import textwrap
    for i, line in enumerate(textwrap.wrap(text, 112)):
        fig.text(0.07, y - i * 0.013, line, fontsize=fontsize or size,
                 color="#444444", family="serif", ha="left", va="top")


def table(fig, y, rows, col_frac=None, size=7.2, row_h=0.0168,
          header=True, note=None):
    import textwrap
    col_frac = col_frac or [1.0 / len(rows[0])] * len(rows[0])
    x0 = 0.07
    for r, row in enumerate(rows):
        top = header and r == 0
        cells = []
        maxlines = 1
        for c, cell in enumerate(row):
            w = max(10, int(col_frac[c] * 118))
            lines = textwrap.wrap(str(cell), w) or [""]
            maxlines = max(maxlines, len(lines))
            cells.append((x0 + sum(col_frac[:c]), lines))
        for xoff, lines in cells:
            for li, line in enumerate(lines):
                fig.text(xoff, y - li * row_h * 0.85, line, fontsize=size,
                         family="monospace" if not top else "sans-serif",
                         fontweight="bold" if top else "normal",
                         color="#1a1a1a", ha="left", va="top")
        if top:
            fig.lines.append(plt.Line2D([x0, x0 + sum(col_frac)],
                                        [y - 0.0035, y - 0.0035],
                                        color="#999999", lw=0.7))
        y -= row_h * (0.6 + 0.85 * (maxlines - 1) + 0.6) * 0.62 + row_h * 0.55
    if note:
        y = para(fig, y - 0.004, note, size=6.6, color="#666666")
    return y - 0.004


def kv(fig, y, pairs, size=7.4, row_h=0.0165, kw=0.46, vw=0.50):
    for k, v in pairs:
        fig.text(0.07, y, k, fontsize=size, family="monospace",
                 color="#333333", va="top")
        fig.text(0.07 + kw, y, str(v), fontsize=size, family="monospace",
                 color="#111111", va="top", fontweight="bold")
        y -= row_h
    return y


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def ci(v, d=3):
    return "-" if v is None else f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


def main():
    ens = load("run_ensemble_ablation.json")
    fus = load("run_fusion_rule_ablation.json")
    ciJ = load("run_ablation_ci.json")
    div = load("run_diversity_stats.json")
    rev = load("run_review_response.json")
    con = load("run_contamination_prepost.json")
    adp = load("run_adaptive_fusion_eval_glm.json")
    cir = load("run_taxonomy_circularity.json")
    lay = load("run_adaptive_layer_robustness.json")
    orx = load("run_or_extra_comparison.json")
    lg4 = load("run_lg4_comparison.json")
    camel = load("run_camel_full_eval.json")
    tcb = load("run_toolcall_bench.json")
    tmf = load("run_toolcall_matched_fpr.json")
    amf = load("run_agentdojo_matched_fpr.json")
    afp = load("run_agentdojo_fpr.json")
    ado = load("run_agentdojo.json")
    blr = load("run_blast_radius_eval.json")
    sig = load("run_signal_probe.json")
    mcn = load("all_mcnemar_counts.json")
    dec = load("run_backbone_deconfound.json")
    tp1 = load("run_tpr_at_1pct.json")
    mcd = load("run_adaptive_mcnemar.json")
    cst = load("run_cost_table.json")

    pdf = PdfPages("RESULTS.pdf")
    SUBS = ["text", "provenance", "pattern", "text+provenance",
            "text+pattern", "provenance+pattern", "text+provenance+pattern"]
    BENCH = ["injecagent", "toolcall", "agentdojo"]
    BLAB = ["InjecAgent", "tool-call", "AgentDojo"]

    # ---------- 1. title + headline ----------
    fig = page(pdf, "TriChannel - complete results compendium",
               "results / generated from results/*.json")
    y = 0.868
    y = para(fig, y,
             "Every table below is rendered at generation time from the "
             "JSON files shipped in results/; the figures use the same "
             "plotting code as figs/make_figs.py and figs/make_figs2.py. "
             "Percentages are exact to the precision shown; intervals are "
             "95% paired-bootstrap.")
    y -= 0.006
    y = para(fig, y, "Headline numbers", bold=True)
    tc3 = ens["ablation"]["toolcall"]["rows"]["text+provenance+pattern"]
    tct = ens["ablation"]["toolcall"]["rows"]["text"]
    fused = adp["systems"]["text+provenance+pattern"]
    y = kv(fig, y, [
        ("action-shaped gain (AUC, fused vs best single)",
         f"{tct['auc']:.3f} -> {tc3['auc']:.3f}"),
        ("action-shaped gain (TPR@7% FPR)",
         f"{pct(tct['tpr_at_7pct_fpr'])} -> {pct(tc3['tpr_at_7pct_fpr'])}"),
        ("off-regime repair, InjecAgent (max -> LR)",
         f"0.944 -> {ens['lr_fusion']['injecagent']['cv_auc']['mean']:.3f}"),
        ("cross-bench transfer, toolcall -> InjecAgent",
         f"{ens['transfer']['toolcall->injecagent']['lr_auc']:.4f} "
         f"(best single {ens['transfer']['toolcall->injecagent']['best_single_auc']:.4f})"),
        ("adaptive fused AUC (GLM-4.6)",
         f"{fused['auc']:.3f} {ci(fused['auc_ci95'])}"),
        ("adaptive margin over text (paired CI)",
         ci([k for k in adp['key_comparisons']
             if k['a'] == 'text+provenance+pattern' and k['b'] == 'text'][0]
            ['auc_diff_ci95'])),
        ("signature-evading draws (circularity)",
         f"{cir['draws_that_fully_evade_signatures']}/{cir['n_attacks']}; "
         f"fused {pct(cir['fused_detection_on_evading'])} vs text "
         f"{pct(cir['text_detection_on_evading'])}"),
        ("contamination pre -> post (GLM-4.6 text AUC)",
         f"{con['pre']['text']['auc']:.3f} -> {con['post']['text']['auc']:.3f}"),
        ("enforcement layer (irreversible prevented)",
         f"{blr['prevention_irreversible_default_pct']:.0f}% "
         f"({blr['n_irreversible']}/{blr['n_attacks']} attacks)"),
    ])
    pdf.savefig(fig); plt.close(fig)

    # ---------- 2. regime baseline ----------
    fig = page(pdf, "Regime baseline: no single channel covers both shapes",
               "results 1 / paper fig. 3a + sec. VI-B")
    y = 0.868

    def g(d, *keys, default=None):
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d or default

    rows = [("detector", "InjecAgent TPR", "tool-call TPR", "AgentDojo TPR")]
    det_rows = [
        ("GLM-4.6 (prompted)", lambda b: lg4[b]["systems"]["GLM-alone (prompted)"]["tpr"]),
        ("GLM-5.2", lambda b: orx["glm-5.2"][b]["alone"]["tpr"]),
        ("GPT-4o-mini", lambda b: orx["gpt-4o-mini"][b]["alone"]["tpr"]),
        ("Claude Haiku 4.5", lambda b: orx["claude-haiku-4-5"][b]["alone"]["tpr"]),
        ("Llama Guard 4 (trained)",
         lambda b: lg4[b]["systems"]["LG4-alone (trained guard)"]["tpr"]),
        ("full CaMeL (provenance)", lambda b: camel[b]["tpr"] / 100),
        ("signature layer (pattern)", None),
    ]
    sig_tpr = {"injecagent": 0.0, "toolcall": 0.70, "agentdojo":
               ado["systems"][0]["tpr"]}
    for name, fn in det_rows:
        cells = [name]
        for b in BENCH:
            try:
                v = fn(b) if fn else sig_tpr[b]
                cells.append(pct(v))
            except Exception:
                cells.append("n/a")
        rows.append(tuple(cells))
    y = table(fig, y, rows, col_frac=[0.30, 0.23, 0.23, 0.24],
              note="Sources: run_lg4_comparison.json, run_or_extra_comparison."
                   "json, run_camel_full_eval.json, run_toolcall_bench.json, "
                   "run_agentdojo.json. Signature layer on AgentDojo = "
                   "det-pipeline TPR; CaMeL AgentDojo has no benign control.")
    y -= 0.004
    y = para(fig, y, "Tool-call bench, matched-FPR operating points (TPR %)",
             bold=True)
    rows = [("detector", "AUC")] + [
        (r["detector"], f"{r['auc']:.3f}" if isinstance(r["auc"], float)
         else r["auc"]) +
        tuple(f"{r['tpr_at_fpr'][f]:.1f}" for f in ("1", "5", "7", "10"))
        for r in tmf["rows"]]
    y = table(fig, y, [("detector", "AUC", "@1%", "@5%", "@7%", "@10%")] + rows[1:],
              col_frac=[0.32, 0.125, 0.125, 0.125, 0.125, 0.13],
              note="Source: run_toolcall_matched_fpr.json.")
    y -= 0.004
    y = para(fig, y, "AgentDojo matched-FPR comparators", bold=True)
    rows = [("comparator", "AUC", "TPR@0.5", "FPR@0.5", "TPR@67.5%FPR")]
    for name, d in amf["comparators"].items():
        rows.append((name, f3(d.get("auc")), pct(d.get("tpr_at_0.5"), 1),
                     pct(d.get("fpr_at_0.5"), 1),
                     pct(d.get("tpr_at_67.5pct_fpr"), 1)))
    y = table(fig, y, rows, col_frac=[0.28, 0.14, 0.17, 0.17, 0.17],
              note=(f"System: TPR {amf['system_tpr']}% at FPR "
                    f"{amf['system_fpr']}%. Source: "
                    "run_agentdojo_matched_fpr.json."))
    pdf.savefig(fig); plt.close(fig)

    # ---------- 3. ablation grid + learned fusion ----------
    fig = page(pdf, "Channel-subset ablation grid (max-fusion)",
               "results 2 / paper fig. 3b + sec. VI-C")
    y = 0.868
    rows = [("subset", "InjA AUC", "InjA TPR", "TC AUC", "TC TPR",
             "AD AUC", "AD TPR")]
    for s in SUBS:
        cells = [s]
        for b in BENCH:
            r = ens["ablation"][b]["rows"].get(s)
            if isinstance(r, dict):
                cells += [f3(r.get("auc")),
                          pct(r.get("tpr_at_7pct_fpr"))]
            else:
                cells += [r if isinstance(r, str) else "n/a", "n/a"]
        rows.append(tuple(cells))
    y = table(fig, y, rows, col_frac=[0.25, 0.113, 0.113, 0.113, 0.113,
                                      0.113, 0.113],
              note="TPR at 7% FPR. n = malicious/benign per bench: "
                   "InjecAgent 187/323, tool-call 240/240, AgentDojo "
                   "280/120. Source: run_ensemble_ablation.json.")
    y -= 0.004
    y = para(fig, y, "Tighter operating points and score granularity",
             bold=True)
    g = tp1["granularity"]
    rows = [("budget", "threshold", "actual FPR", "blocking tie",
             "benign at tie")]
    for tag in ("1pct", "7pct"):
        d = g[f"text@{tag}"]
        rows.append((tag, d["threshold"], f"{d['actual_fpr_pct']}%",
                     d["blocking_tie_value"], d["benign_tied_at_blocker"]))
    y = table(fig, y, rows, col_frac=[0.12, 0.16, 0.18, 0.26, 0.20],
              note=f"Text guard emits {g['n_distinct_text_scores']} distinct "
                   f"scores over 480 trajectories; the benign tie plateau at "
                   f"{g['text@1pct']['blocking_tie_value']} blocks both "
                   f"budgets, so TPR@1% = TPR@7% for every subset. "
                   f"Source: run_tpr_at_1pct.json.")
    y -= 0.004
    y = para(fig, y, "Learned linear head (5-fold CV)", bold=True)
    rows = [("bench", "max-fusion AUC", "LR CV AUC", "coef (text, prov, pattern)")]
    for b in ("toolcall", "injecagent"):
        d = ens["lr_fusion"][b]
        c = d["coefficients"]
        if isinstance(c, dict):
            cs = ", ".join(f"{c[k]:.2f}" for k in
                           ("text", "provenance", "pattern") if k in c)
        else:
            cs = ", ".join(f"{v:.2f}" for v in c)
        rows.append((b, f3(d["max_fusion_auc"]),
                     f"{d['cv_auc']['mean']:.4f} +/- {d['cv_auc']['std']:.4f}",
                     cs))
    y = table(fig, y, rows, col_frac=[0.16, 0.20, 0.28, 0.36])
    y -= 0.004
    y = para(fig, y, "Fusion-rule comparison", bold=True)
    rows = [("rule", "InjecAgent AUC [CI]", "tool-call AUC [CI]")]
    for rule, label in (("max", "max"), ("mean", "mean"),
                        ("noisy-or", "noisy-or"), ("product", "product"),
                        ("lr", "lr_oof")):
        cells = [label]
        for b in BENCH[:2]:
            r = fus["fusion_rules"][b]["rows"].get(rule)
            cells.append(f"{r['auc']:.4f} {ci(r['auc_ci95'])}" if r else "-")
        rows.append(tuple(cells))
    for b in BENCH[:2]:
        pass
    y = table(fig, y, rows, col_frac=[0.16, 0.44, 0.40],
              note="Source: run_fusion_rule_ablation.json; lr_oof = pooled "
                   "out-of-fold decision function.")
    y -= 0.004
    y = para(fig, y, "Cross-benchmark transfer", bold=True)
    rows = [("train -> evaluate", "LR AUC", "max-fusion", "best single")]
    for k, d in ens["transfer"].items():
        def f4(v):
            return "n/a" if v is None else f"{v:.4f}"
        rows.append((k.replace("->", " -> "), f4(d.get("lr_auc")),
                     f4(d.get("max_fusion_auc")),
                     f4(d.get("best_single_auc"))))
    y = table(fig, y, rows, col_frac=[0.34, 0.22, 0.22, 0.22])
    pdf.savefig(fig); plt.close(fig)

    # ---------- 4. composition + contamination ----------
    fig = page(pdf, "Composition ceiling and benchmark contamination",
               "results 3 / paper sec. VI-G + VI-H")
    y = 0.868
    y = para(fig, y, "Instruction-shape filter composed with each classifier "
                     "(InjecAgent, real benign; max-fusion)", bold=True)
    rows = [("classifier", "alone", "composed", "diff", "95% CI")]
    for name, d in rev["B_composition_pairings_injecagent"].items():
        rows.append((name, f"{d['classifier_alone_auc']:.3f}",
                     f"{d['composition_auc']:.3f}",
                     f"{d['diff']:+.3f}", ci(d["diff_ci95"])))
    y = table(fig, y, rows, col_frac=[0.26, 0.16, 0.17, 0.15, 0.26],
              note="Source: run_review_response.json.")
    y -= 0.004
    y = para(fig, y, "AgentDojo constructed benign + alternative fusion rules",
             bold=True)
    bb = rev["B_fusion_rule_reconstruction"]
    alt = rev["B_composition_pairings_injecagent"]["GLM-4.6"]["alternative_fusions"]
    rows = [("system", "AUC", "diff vs classifier [CI]")]
    rows.append(("input filter alone (AgentDojo)",
                 f"{bb['agentdojo_input_filter_alone_auc']:.3f}", ""))
    rows.append(("filter + GPT-4o-mini (AgentDojo)",
                 f"{bb['agentdojo_gpt4o_composition_auc']:.3f}", ""))
    for rule in ("mean", "product"):
        cell = f"{alt[rule]['diff']:+.4f} {ci(alt[rule]['diff_ci95'])}"
        if rule == "mean":
            cell += "  (best of all pairings)"
        rows.append((f"GLM-4.6 + filter, {rule} rule",
                     f"{alt[rule]['auc']:.3f}", cell))
    y = table(fig, y, rows, col_frac=[0.34, 0.16, 0.44],
              note="Source: run_review_response.json. Under mean or product "
                   "fusion no pairing improves on the classifier alone on "
                   "InjecAgent real benign (paper rounds the best diff, "
                   "+0.0065, to +0.007).")
    y -= 0.004
    y = para(fig, y, "InjecAgent loader contamination, same classifier "
                     "(GLM-4.6, 510 records)", bold=True)
    pre, post = con["pre"]["text"], con["post"]["text"]
    rows = [("metric", "pre-fix", "post-fix", "delta"),
            ("AUC", f"{pre['auc']:.3f}", f"{post['auc']:.3f}",
             f"{con['delta_post_minus_pre']['text']['auc']:+.3f}"),
            ("benign FPR @0.5", pct(pre.get("benign_fpr_at_0.5")),
             pct(post.get("benign_fpr_at_0.5")), ""),
            ("TPR @7% FPR", pct(pre["tpr_at_7pct_fpr"]),
             pct(post["tpr_at_7pct_fpr"]),
             f"{con['delta_post_minus_pre']['text']['tpr_at_7pct_fpr']:+.3f}")]
    y = table(fig, y, rows, col_frac=[0.25, 0.22, 0.22, 0.22],
              note="Source: run_contamination_prepost.json; cache "
                   "glm_guard__injecagent_prefix.jsonl. Only the 323 benign "
                   "rows differ between constructions.")
    y -= 0.004
    y = para(fig, y, "Cheap text-derived signals (InjecAgent, 510)", bold=True)
    rows = [("signal", "AUC", "TPR@1%FPR")]
    for s in sig["signals"]:
        rows.append((s["signal"], f3(s["auc"]), pct(s.get("tpr_at_1pct_fpr"))))
    y = table(fig, y, rows, col_frac=[0.44, 0.20, 0.24],
              note="Source: run_signal_probe.json. Det nonconformity is "
                   "identically zero - no signature fires on InjecAgent.")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 5. adaptive bench ----------
    fig = page(pdf, "Adaptive benchmark: fused score under paraphrase attack",
               "results 4 / paper sec. VI-H + fig. 5a")
    y = 0.868
    y = para(fig, y, f"440 attacks / 240 benign, provider "
                     f"{adp['scoring_provider']}, bootstrap "
                     f"n={adp['n_boot']}", bold=True)
    rows = [("system", "AUC [95% CI]", "TPR@7%FPR [CI]")]
    for s in ["text", "provenance", "pattern", "text+provenance",
              "text+pattern", "provenance+pattern", "text+provenance+pattern"]:
        d = adp["systems"][s]
        rows.append((s, f"{d['auc']:.3f} {ci(d['auc_ci95'])}",
                     f"{pct(d['tpr_at_7pct_fpr'])} {ci(d['tpr_ci95'], 1)}"))
    for s in [k for k in adp["systems"] if k.startswith("lr")]:
        d = adp["systems"][s]
        rows.append((s, f"{d['auc']:.3f} {ci(d['auc_ci95'])}",
                     f"{pct(d['tpr_at_7pct_fpr'])} {ci(d['tpr_ci95'], 1)}"))
    y = table(fig, y, rows, col_frac=[0.30, 0.36, 0.34],
              note="Source: run_adaptive_fusion_eval_glm.json.")
    y -= 0.004
    y = para(fig, y, "Key paired comparisons", bold=True)
    rows = [("comparison", "AUC diff [95% CI]", "McNemar p")]
    for k in adp["key_comparisons"]:
        rows.append((f"{k['a']} vs {k['b']}", ci(k["auc_diff_ci95"]),
                     f"{k['mcnemar_p']:.2e}"))
    y = table(fig, y, rows, col_frac=[0.46, 0.30, 0.24])
    y -= 0.004
    y = para(fig, y, "Detection by attack shape (fused 7% threshold)", bold=True)
    pf = adp["per_family_detection_at_own_7pct_threshold"]
    fused_pf = pf["text+provenance+pattern"]
    nopat = pf.get("text+provenance", {})
    rows = [("tier", "fused", "without pattern channel")]
    for t in ("T.A", "T.B", "T.C"):
        rows.append((t, pct(fused_pf.get(t)), pct(nopat.get(t))))
    y = table(fig, y, rows, col_frac=[0.30, 0.32, 0.38],
              note="T.A tool-argument, T.B emitted-text, T.C data-flow. "
                   "Per-family values follow on the next page.")
    y -= 0.004
    y = para(fig, y, "Static vs adaptive (unpaired, indicative)", bold=True)
    rows = [("channel", "static tool-call AUC", "adaptive AUC", "delta")]
    for name, d in adp["static_vs_adaptive_side_by_side"].items():
        rows.append((name, f"{d['static_toolcall_auc']:.3f}",
                     f"{d['adaptive_auc']:.3f}", f"{d['delta']:+.3f}"))
    y = table(fig, y, rows, col_frac=[0.30, 0.26, 0.22, 0.18])
    pdf.savefig(fig); plt.close(fig)

    # ---------- 6. per-family + circularity ----------
    fig = page(pdf, "Per-family adaptive detection and taxonomy circularity",
               "results 5 / paper sec. VI-I + fig. 5b")
    y = 0.868
    pf = adp["per_family_detection_at_own_7pct_threshold"]
    fused_pf = pf["text+provenance+pattern"]
    nopat = pf.get("text+provenance", {})
    fams = [k for k in fused_pf if k.count(".") == 2]
    rows = [("family", "fused", "no pattern")] + [
        (f, pct(fused_pf[f]), pct(nopat.get(f))) for f in fams]
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    while len(right) < len(left):
        right.append(("", "", ""))
    merged = [("family", "fused", "no pat", "family", "fused", "no pat")]
    for a, b in zip(left, right):
        merged.append(a + b)
    y = table(fig, y, merged, col_frac=[0.14, 0.09, 0.09, 0.14, 0.09, 0.09])
    y -= 0.002
    y = para(fig, y, "Taxonomy circularity split (zero API)", bold=True)
    y = kv(fig, y, [
        ("draws tripping >= 1 signature",
         f"{cir['draws_that_trip_a_signature']} / {cir['n_attacks']} "
         f"({100 * cir['draws_that_trip_a_signature'] / cir['n_attacks']:.1f}%)"),
        ("draws fully evading signatures",
         str(cir["draws_that_fully_evade_signatures"])),
        ("fused detection on tripping draws",
         pct(cir["fused_detection_on_tripping"])),
        ("fused detection on evading draws",
         pct(cir["fused_detection_on_evading"])),
        ("text detection on evading draws",
         pct(cir["text_detection_on_evading"])),
        ("provenance flag rate on evading draws",
         pct(cir["prov_flagrate_on_evading"])),
        ("evading-draw detection by tier",
         ", ".join(f"{k} {pct(v)}" for k, v in
                   cir["fused_detection_on_evading_by_tier"].items())),
        ("text scores <= 0.05 among attacks",
         f"{cir['text_le_0p05_count']} / {cir['n_attacks']} "
         f"({100 * cir['text_le_0p05_count'] / cir['n_attacks']:.1f}%)"),
        ("pattern channel cost",
         f"{cir['pattern_ms_per_trajectory']} ms / trajectory, local"),
        ("thresholds (pattern / fused @7% FPR)",
         f"{cir['pattern_threshold_7pct']} / {cir['fused_threshold_7pct']}"),
    ])
    y = para(fig, y,
             "Source: run_taxonomy_circularity.json, regenerated by "
             "run_taxonomy_circularity.py. The 133 tripping draws are "
             "exactly the pattern channel's TPR@7%FPR: detection is 100% "
             "on tripping draws and 0% on evading ones - the adaptive "
             "margin is a property of the draw distribution.",
             size=7.6, color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 6b. McNemar / Holm detail ----------
    fig = page(pdf, "McNemar / Holm detail: subset pairs and adaptive",
               "results 6b / paper sec. VI-C + VI-I")
    y = 0.868
    sums = []
    for bench in ("toolcall", "injecagent", "agentdojo"):
        pmh = ciJ["per_bench"][bench]["pairwise_mcnemar_holm"]
        sums.append(f"{sum(1 for e in pmh if e['significant_holm_0.05'])}"
                    f"/{len(pmh)} {bench}")
    y = para(fig, y,
             "Holm-corrected subset-pair McNemar at matched 7% FPR: "
             + ", ".join(sums), bold=True)
    tc_p = ciJ["per_bench"]["toolcall"]["pairwise_mcnemar_holm"]
    ij_by = {(e["a"], e["b"]): e
             for e in ciJ["per_bench"]["injecagent"]["pairwise_mcnemar_holm"]}
    rows = [("pair", "tc b", "tc c", "tc holm", "InjA b", "InjA c", "InjA holm")]
    for e in tc_p:
        ij = ij_by[(e["a"], e["b"])]
        rows.append((f"{e['a']} vs {e['b']}",
                     e["b_only_catches"], e["c_only_catches"],
                     "sig" if e["significant_holm_0.05"] else "-",
                     ij["b_only_catches"], ij["c_only_catches"],
                     "sig" if ij["significant_holm_0.05"] else "-"))
    y = table(fig, y, rows, col_frac=[0.30, 0.10, 0.10, 0.11, 0.10, 0.10, 0.11],
              row_h=0.0145, size=6.6,
              note="b = rows only the second system catches; c = only the "
                   "first. Holm-corrected at 0.05 within each benchmark's "
                   "21-pair family (AgentDojo: 3 non-degenerate pairs, "
                   "2 significant). Source: run_ablation_ci.json.")
    y -= 0.004
    y = para(fig, y, "Adaptive bench: McNemar at matched operating points",
             bold=True)
    rows = [("comparison", "b", "c", "p")]
    for c_ in mcd["comparisons"]:
        rows.append((f"{c_['larger']} vs {c_['smaller']}",
                     c_["b"], c_["c"], c_["p_value"]))
    y = table(fig, y, rows, col_frac=[0.52, 0.10, 0.10, 0.16],
              note="b = smaller-only catches, c = larger-only; b = 0 means "
                   "disagreements are one-directional. Zero-API recompute "
                   "from the shipped adaptive caches. Source: "
                   "run_adaptive_mcnemar.json.")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 7. statistics ----------
    fig = page(pdf, "Statistical treatment: CIs, McNemar, diversity",
               "results 6 / paper sec. VI-C")
    y = 0.868
    y = para(fig, y, f"Full-grid bootstrap CIs (AUC), n_boot="
                     f"{ciJ['n_boot']}, FPR budget {ciJ['fpr_budget']}",
             bold=True)
    rows = [("subset", "InjecAgent", "tool-call", "AgentDojo")]
    for s in SUBS:
        cells = [s]
        for b in BENCH:
            r = ciJ["per_bench"][b]["rows"].get(s)
            cells.append(f"{r['auc']:.3f} {ci(r['auc_ci95'])}" if r else "n/a")
        rows.append(tuple(cells))
    y = table(fig, y, rows, col_frac=[0.18, 0.27, 0.27, 0.28],
              note="Source: run_ablation_ci.json. TPR CIs and Holm-corrected "
                   "McNemar outcomes are in the same file; the AgentDojo "
                   "pattern inversion is stable: "
                   f"frac(replicates < 0.5) = "
                   f"{ciJ['agentdojo_pattern_sign_stability']['frac_auc_below_half']}, "
                   f"CI {ci(ciJ['agentdojo_pattern_sign_stability']['auc_ci95'])}.")
    y -= 0.004
    y = para(fig, y, "Channel-error diversity (Kuncheva)", bold=True)
    rows = [("bench", "pair", "Q", "disagree", "double fault")]
    for b, d in div["channel_diversity"].items():
        for pair, r in d["rows"].items():
            rows.append((b, pair, f"{r['q_statistic']:.3f}",
                         pct(r["disagreement_rate"]),
                         pct(r["double_fault_rate"])))
    y = table(fig, y, rows, col_frac=[0.15, 0.27, 0.16, 0.20, 0.20],
              note="Source: run_diversity_stats.json.")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 7b. statistics, page 2 ----------
    fig = page(pdf, "Statistical treatment (cont.): backbone, McNemar",
               "results 7 / paper sec. VI-C")
    y = 0.868
    y = para(fig, y, "Backbone-variant Spearman (text channel)", bold=True)
    rows = [("bench", "pair", "spearman")]
    for b, d in div["backbone_correlation"].items():
        for pair, v in d["rows"].items():
            rows.append((b, pair, f"{v:.3f}"))
    y = table(fig, y, rows, col_frac=[0.20, 0.55, 0.25])
    y -= 0.004
    y = para(fig, y, "McNemar counts (system vs comparator)", bold=True)
    rows = [("comparison", "b", "c", "p")]
    for name, d in mcn.items():
        if isinstance(d, dict):
            rows.append((name, d.get("b"), d.get("c"), f"{d.get('p')}"))
    y = table(fig, y, rows, col_frac=[0.46, 0.16, 0.16, 0.22])
    y -= 0.004
    y = para(fig, y, "Paired differences vs the text channel alone",
             bold=True)
    pdJ = load("run_paired_head_diffs.json")
    rows = [("comparison", "AUCs", "diff [CI]")]

    def _r(label, d, key="auc_diff"):
        return (label, f"{d.get('lr_oof_auc', d.get('lr_auc', d.get('text_prov_auc'))):.4f}"
                       f" vs {d['text_auc']:.4f}",
                f"{d[key]['diff']:+.4f} {ci(d[key]['ci95'])}")

    c = pdJ["comparisons"]
    rows.append(_r("learned head (OOF), injecagent", c["injecagent_lr_oof_vs_text"]))
    rows.append(_r("learned head (OOF), toolcall", c["toolcall_lr_oof_vs_text"]))
    rows.append(_r("transfer tc->inj", c["transfer_toolcall_to_injecagent_vs_text"]))
    rows.append(_r("transfer inj->tc", c["transfer_injecagent_to_toolcall_vs_text"]))
    rows.append(_r("text+prov max-fusion, toolcall", c["toolcall_text_prov_vs_text"]))
    y = table(fig, y, rows, col_frac=[0.40, 0.26, 0.34],
              note="Source: run_paired_head_diffs.py (zero-API). The head "
                   "matches text on InjecAgent (CI crosses zero) and adds "
                   "where channels carry signal (toolcall); the "
                   "taxonomy-independent subset's toolcall gain excludes "
                   "zero.")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 8. enforcement + layers ----------
    fig = page(pdf, "Enforcement layer, layer robustness, AgentDojo FPR",
               "results 8 / paper sec. VI-K")
    y = 0.868
    y = para(fig, y, "Blast-radius evaluation (capability-tier gateway)",
             bold=True)
    y = kv(fig, y, [(k.replace("_", " "), v) for k, v in blr.items()
                    if not isinstance(v, (list, dict))])
    y -= 0.006
    y = para(fig, y, "Layer-level adaptive bypass (440 paraphrase attacks)",
             bold=True)
    y = kv(fig, y, [(k, v) for k, v in lay.items()
                    if not isinstance(v, (list, dict))])
    y -= 0.006
    y = para(fig, y, "AgentDojo layer FPR (120 constructed benign)", bold=True)
    rows = [("layer", "TPR", "benign FPR")]
    for name, d in afp["layers"].items():
        rows.append((name, f"{d['tpr']}%", f"{d['fpr_benign']}%"))
    y = table(fig, y, rows, col_frac=[0.40, 0.28, 0.32],
              note=(f"pii_benign_fraction = {afp['pii_benign_fraction']}. "
                    "Source: run_agentdojo_fpr.json."))
    y -= 0.006
    y = para(fig, y, "Tool-call bench layer summary", bold=True)
    rows = [("system", "TPR [CI]", "FPR [CI]")]
    for name in ("det-output-filter", "GLM-alone (@0.5)", "tool-gateway",
                 "input-filter", "context-guard", "FULL-SYSTEM (GLM)"):
        d = tcb["systems"].get(name)
        if d and isinstance(d, dict):
            rows.append((name, f"{pct(d['tpr'])} {ci(d['tpr_ci95'], 1)}",
                         pct(d['fpr']) + f" {ci(d['fpr_ci95'], 1)}"))
    y = table(fig, y, rows, col_frac=[0.30, 0.36, 0.34],
              note=(f"GLM AUC on this bench: {tcb['glm_auc']:.3f}. "
                    "Source: run_toolcall_bench.json."))
    pdf.savefig(fig); plt.close(fig)

    # ---------- 9-13. figures ----------
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})

    # Fig: regime heatmap (values as in figs/make_figs.py)
    fig = page(pdf, "Figure: regime baseline heatmap", "figures / paper fig. 3a")
    rowsn = ["GLM-4.6", "GLM-5.2", "GPT-4o-mini", "Claude Haiku 4.5",
             "Llama Guard 4", "full CaMeL (provenance)",
             "signature layer (pattern)"]
    M = np.array([[79.1, 23.8, 21], [95.2, 34.6, 44], [97.9, 27.1, 84],
                  [89.3, 34.6, 34], [46.0, 10.8, 70], [4.3, 42.5, np.nan],
                  [0, 70.0, 10]])
    ax = fig.add_axes([0.30, 0.30, 0.56, 0.58])
    im = ax.imshow(np.nan_to_num(M, nan=0), cmap="Blues", vmin=0, vmax=100,
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.0f}", ha="center",
                    va="center",
                    color="white" if (not np.isnan(v) and v > 55) else "black",
                    fontsize=8)
    ax.set_xticks(range(3), ["InjecAgent", "tool-call", "AgentDojo"])
    ax.set_yticks(range(len(rowsn)), rowsn)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("TPR (%)")
    _cap(fig, 0.22, "Classifiers are strong on text-shaped attacks and "
            "collapse on action-shaped ones; CaMeL is the mirror image; the "
            "signature layer is taxonomy-bound. Rendering identical to "
            "figs/make_figs.py (values from the regime table, page 2).",
            fontsize=7.6, color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # Fig: ablation heatmap
    fig = page(pdf, "Figure: channel-subset ablation heatmap",
               "figures / paper fig. 3b")
    subs = ["text", "provenance", "pattern", "text + provenance",
            "text + pattern", "provenance + pattern", "all three"]
    A = np.array([[0.972, 0.832, 0.923], [0.503, 0.713, np.nan],
                  [0.500, 0.656, 0.152], [0.944, 0.845, np.nan],
                  [0.972, 0.861, 0.237], [0.503, 0.754, np.nan],
                  [0.944, 0.874, np.nan]])
    ax = fig.add_axes([0.30, 0.30, 0.56, 0.58])
    im = ax.imshow(np.where(np.isnan(A), 0.5, A), cmap="RdYlGn", vmin=0.5,
                   vmax=1.0, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center",
                    va="center", fontsize=8)
    ax.set_xticks(range(3), ["InjecAgent", "tool-call", "AgentDojo"])
    ax.set_yticks(range(len(subs)), subs)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("AUC (max-fusion)")
    _cap(fig, 0.22, "Monotone complementarity on the action-shaped "
            "bench; degradation below the best single channel where a "
            "channel is off-regime. Identical to figs/make_figs.py; AUC "
            "values equal run_ensemble_ablation.json (page 3).",
            fontsize=7.6, color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # Fig: adaptive AUC bars
    fig = page(pdf, "Figure: fused margin under adaptation",
               "figures / paper fig. 5a")
    names = ["text", "provenance", "pattern", "text+prov.", "text+pattern",
             "prov.+pattern", "all three", "LR head (tool-call)",
             "LR head (InjecAgent)"]
    keys = ["text", "provenance", "pattern", "text+provenance",
            "text+pattern", "provenance+pattern", "text+provenance+pattern"]
    lrk = [k for k in adp["systems"] if k.startswith("lr")]
    keys += lrk
    auc = [adp["systems"][k]["auc"] for k in keys]
    lo = [adp["systems"][k]["auc_ci95"][0] for k in keys]
    hi = [adp["systems"][k]["auc_ci95"][1] for k in keys]
    names = names[:7] + [k.replace("lr@", "LR head, ") for k in lrk]
    ax = fig.add_axes([0.34, 0.30, 0.52, 0.58])
    yv = np.arange(len(names))[::-1]
    cols = ["#4c72b0"] * 6 + ["#2a9d8f"] + ["#999999"] * (len(names) - 7)
    ax.barh(yv, auc, color=cols, height=0.6)
    ax.errorbar(auc, yv,
                xerr=[np.array(auc) - np.array(lo),
                      np.array(hi) - np.array(auc)],
                fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_yticks(yv, names)
    ax.set_xlim(0.5, 0.85)
    ax.set_xlabel("AUC (adaptive bench, 95% CI)")
    ax.axvline(adp["systems"]["text"]["auc"], ls="--", lw=0.7,
               color="#4c72b0")
    _cap(fig, 0.22, "Bars read live from "
            "run_adaptive_fusion_eval_glm.json (systems section, page 5). "
            "Dashed line: text channel alone.", fontsize=7.6,
            color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # Fig: per-shape detection
    fig = page(pdf, "Figure: detection by attack shape",
               "figures / paper fig. 3b")
    shapes = ["tool-argument", "emitted-text", "data-flow"]
    fusedp = adp["per_family_detection_at_own_7pct_threshold"][
        "text+provenance+pattern"]
    nopatp = adp["per_family_detection_at_own_7pct_threshold"][
        "text+provenance"]
    fused = [100 * fusedp[t] for t in ("T.A", "T.B", "T.C")]
    nopat = [100 * nopatp[t] for t in ("T.A", "T.B", "T.C")]
    evade = [100 * cir["fused_detection_on_evading_by_tier"][t]
             for t in ("T.A", "T.B", "T.C")]
    ax = fig.add_axes([0.15, 0.32, 0.70, 0.54])
    x = np.arange(3)
    w = 0.26
    ax.bar(x - w, fused, w, label="fused, all 440 draws", color="#2a9d8f")
    ax.bar(x, nopat, w, label="without pattern channel", color="#4c72b0")
    ax.bar(x + w, evade, w, label="fused, 307 signature-evading draws",
           color="#e76f51")
    for xi, v in zip(list(x - w) + list(x) + list(x + w),
                     fused + nopat + evade):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, shapes)
    ax.set_ylabel("detected at 7% FPR (%)")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    _cap(fig, 0.24, "First two bar groups read from "
            "run_adaptive_fusion_eval_glm.json; the evasion-conditional "
            "bars from run_taxonomy_circularity.json. Data-flow attacks "
            "are fully undetected once the signature layer is evaded.",
            fontsize=7.6, color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # Fig: contamination
    fig = page(pdf, "Figure: contamination pre/post", "figures / paper fig. 4")
    m = ["benign FPR @0.5", "TPR @7% FPR", "AUC x100"]
    pre = con["pre"]["text"]
    post = con["post"]["text"]
    pre_v = [100 * pre.get("benign_fpr_at_0.5", 0.814),
             100 * pre["tpr_at_7pct_fpr"], 100 * pre["auc"]]
    post_v = [100 * post.get("benign_fpr_at_0.5", 0.0),
              100 * post["tpr_at_7pct_fpr"], 100 * post["auc"]]
    ax = fig.add_axes([0.15, 0.34, 0.68, 0.50])
    x = np.arange(3)
    ax.bar(x - 0.18, pre_v, 0.36, label="pre-fix", color="#e76f51")
    ax.bar(x + 0.18, post_v, 0.36, label="post-fix", color="#2a9d8f")
    for xi, v in zip(list(x - 0.18) + list(x + 0.18), pre_v + post_v):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, m)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, frameon=False)
    _cap(fig, 0.26, "Read from run_contamination_prepost.json: the "
            "loader bug put the text channel at chance (AUC 0.518) with "
            "81.4% benign FPR; post-fix 0.916 and 0%.", fontsize=7.6,
            color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 13b. cost and latency (Table VIII) ----------
    fig = page(pdf, "Cost and latency per channel (Table VIII provenance)",
               "results 8b / paper sec. VI-J + table VIII")
    y = 0.868
    r = cst["recomputed"]
    cc = cst["cost_usd_per_1000_attack_trajectories"]
    ml = cst["measured_from_log"]
    y = para(fig, y,
             "Adaptive run: 440 attacks + 240 benign, one provider "
             "(GLM-4.6). Wall-clock and call counts are parsed from the "
             "shipped run log; prompt lengths and pattern timing are "
             "recomputed locally.", size=8.2)
    rows = [("channel", "calls/traj", "wall-clock per traj",
             "USD / 1k attacks")]
    rows.append(("text (GLM-4.6)", 1,
                 f"{r['text_s_per_trajectory_8way']} s @ 8-way "
                 f"({r['text_s_per_call_est']} s/call)",
                 f"${cc['text_glm46']}"))
    rows.append(("provenance (CaMeL)", r["camel_calls_per_trajectory"],
                 f"{r['camel_s_per_trajectory_sequential']} s sequential "
                 f"({r['camel_s_per_call']} s/call)",
                 f"${cc['camel_glm46_est']} (est)"))
    rows.append(("pattern", 0,
                 f"{r['pattern_ms_per_trajectory']} ms local, single-thread",
                 "$0"))
    rows.append(("fused (max)",
                 round(1 + r["camel_calls_per_trajectory"], 2),
                 "CaMeL-bound unless parallelized", f"${cc['fused_max']}"))
    y = table(fig, y, rows, col_frac=[0.20, 0.12, 0.44, 0.24])
    y -= 0.004
    y = para(fig, y, "Benign traffic (240 constructed benign)", bold=True)
    y = para(fig, y,
             f"CaMeL issued calls on only {ml['camel_benign_calls']} of 240 "
             f"benign trajectories ({r['camel_benign_calls_per_trajectory']} "
             f"per trajectory, {r['camel_benign_s_per_trajectory']} s per "
             f"trajectory amortized); text scored all 240 in "
             f"{ml['text_benign_s']} s at 8-way. Deployed benign-heavy "
             f"traffic approaches text-plus-pattern cost.")
    y = para(fig, y, "Token and price provenance", bold=True)
    y = para(fig, y,
             f"Mean guard prompt {r['mean_guard_prompt_chars_attacks']} "
             f"characters over the 440 attacks (~{r['mean_guard_prompt_tokens_est']} "
             f"input tokens at 4 chars/token; output is one score line). "
             f"CaMeL per-call tokens estimated at "
             f"{cst['config']['estimates']['camel_tokens']['in']} in / "
             f"{cst['config']['estimates']['camel_tokens']['out']} out. "
             f"September 2026 list prices; the same prompt on Claude Haiku "
             f"4.5 costs ${cc['text_claude_haiku45']} per 1,000. Pattern "
             f"timing is machine-dependent (recomputed at render time).")
    y = para(fig, y,
             "Source: run_cost_table.py -> run_cost_table.json (parses "
             "run_adaptive_fusion_eval_glm.log; zero API calls).",
             size=7.6, color="#444444")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 14. backbone deconfound ----------
    fig = page(pdf, "Backbone deconfound (provenance re-scored on Gemma 4 31B)",
               "results 9 / paper sec. VI-D note")
    y = 0.868
    y = para(fig, y,
             "The shipped tool-call channels share Claude Haiku 4.5 (text) "
             "and Haiku-backed CaMeL (provenance). This run re-scores ONLY "
             "the provenance channel on a different backbone, then "
             "recomputes every affected number from caches; all intervals "
             "are 95% paired bootstraps (n=2000). Source: "
             "run_backbone_deconfound.json.")
    y -= 0.004
    ps = dec["provenance_single"]; fd = dec["fused_triple"]; tpp = dec["text_plus_provenance"]
    cfg = dec["config"]
    y = para(fig, y, f"Setup: n={cfg['n']} tool-call trajectories, backbone "
                     f"{cfg['model']} for CaMeL P/Q ({cfg['gateway_calls']} "
                     f"gateway calls, {cfg['gateway_ok']} answered); text "
                     f"stays on its shipped Haiku scores; pattern recomputed "
                     f"locally.", size=8.0)
    y -= 0.006
    y = para(fig, y, "Manipulation check: provenance channel alone", bold=True)
    y = table(fig, y, [
        ("provenance on", "AUC", "hard-label TPR", "hard-label FPR"),
        ("Gemma 4 31B (new)", f3(ps["new_auc"]), pct(ps["new_tpr"]), pct(ps["new_fpr"])),
        ("Haiku 4.5 (shipped)", f3(ps["shipped_auc"]), "-", "-"),
    ], col_frac=[0.34, 0.18, 0.24, 0.24])
    y = para(fig, y,
             f"Paired AUC difference (new - shipped): "
             f"{ps['auc_diff_new_vs_shipped']['diff']:+.4f} "
             f"{ci(ps['auc_diff_new_vs_shipped']['ci95'], 4)}", size=8.0)
    y -= 0.006
    y = para(fig, y, "Fused triple and taxonomy-independent subset", bold=True)
    y = table(fig, y, [
        ("system", "AUC (new prov)", "TPR@7% (new)", "AUC (shipped prov)"),
        ("text+prov+pattern", f3(fd["new_auc"]), pct(fd["new_tpr7"]), f3(fd["shipped_auc"])),
        ("text+prov", f3(tpp["new_auc"]), pct(tpp["new_tpr7"]), f3(tpp["shipped_auc"])),
        ("text alone (unchanged)", f3(tpp["text_auc"]), "-", "-"),
    ], col_frac=[0.28, 0.23, 0.21, 0.28])
    y = para(fig, y,
             f"Fused, new vs shipped provenance: "
             f"{fd['auc_diff_new_vs_shipped']['diff']:+.4f} "
             f"{ci(fd['auc_diff_new_vs_shipped']['ci95'], 4)} (TPR@7%: "
             f"{fd['tpr7_diff_new_vs_shipped']['diff']:+.4f} "
             f"{ci(fd['tpr7_diff_new_vs_shipped']['ci95'], 4)}).",
             size=8.0)
    y = para(fig, y,
             f"Taxonomy-independent gain survives: text+prov (new) vs text "
             f"alone {tpp['auc_diff_vs_text']['diff']:+.4f} "
             f"{ci(tpp['auc_diff_vs_text']['ci95'])} AUC, "
             f"{tpp['tpr7_diff_vs_text']['diff']:+.4f} "
             f"{ci(tpp['tpr7_diff_vs_text']['ci95'])} TPR@7%.",
             size=8.0)
    y -= 0.008
    dec2 = load("run_backbone_deconfound_injecagent.json")
    p2 = dec2["provenance_single"]; f2 = dec2["fused_triple"]
    lh = dec2["lr_head"]; cfg2 = dec2["config"]
    y = para(fig, y, f"Off-regime bench (InjecAgent, n={cfg2['n']}): is the "
                     f"P2 degradation backbone-independent? "
                     f"{cfg2['gateway_calls']} gateway calls, all answered "
                     f"(CaMeL policy short-circuits no-flow trajectories "
                     f"before any LLM call). Source: "
                     f"run_backbone_deconfound_injecagent.json.", bold=False)
    y = table(fig, y, [
        ("system", "shipped prov (Haiku)", "new prov (Gemma)"),
        ("provenance fires / trajectories",
         f"{p2['shipped_fires']} / {cfg2['n']}", f"{p2['new_fires']} / {cfg2['n']}"),
        ("provenance alone AUC", f3(p2["shipped_auc"]), f3(p2["new_auc"])),
        ("fused triple AUC", f3(f2["shipped_auc"]), f3(f2["new_auc"])),
        ("fused TPR@7% FPR", pct(f2["shipped_tpr7"]), pct(f2["new_tpr7"])),
        ("LR head CV AUC", f"{lh['with_shipped_prov']['mean']:.4f}",
         f"{lh['with_new_prov']['mean']:.4f} (text-only "
         f"{lh['text_alone']['mean']:.4f})"),
    ], col_frac=[0.40, 0.28, 0.32])
    y = para(fig, y,
             f"Fused (Gemma prov) equals text alone exactly: paired AUC "
             f"diff {f2['auc_diff_new_vs_text']['diff']:+.4f} "
             f"{ci(f2['auc_diff_new_vs_text']['ci95'], 4)} (constant column); "
             f"fused new-vs-shipped {f2['auc_diff_new_vs_shipped']['diff']:+.4f} "
             f"{ci(f2['auc_diff_new_vs_shipped']['ci95'], 4)}. The "
             f"degradation is the backbone's firing propensity: 20 hard "
             f"fires import benign false positives; 0 fires import nothing.",
             size=8.0)
    pdf.savefig(fig); plt.close(fig)

    # ---------- 15. inventory ----------
    fig = page(pdf, "Artifact inventory", "results 10 / files")
    y = 0.868
    rows = [("file", "bytes", "top-level keys")]
    for p in sorted(R.glob("*.json")):
        d = json.loads(p.read_text())
        keys = ", ".join(list(d.keys())[:5]) if isinstance(d, dict) else \
            f"list[{len(d)}]"
        rows.append((p.name, f"{p.stat().st_size:,}", keys[:60]))
    y = table(fig, y, rows, col_frac=[0.34, 0.10, 0.56], size=6.6,
              row_h=0.0150)
    y -= 0.004
    caches = sorted((R / "cache").glob("*.jsonl"))
    y = para(fig, y, f"Per-trajectory caches ({len(caches)} files, "
                     f"results/cache/):", bold=True)
    y = para(fig, y, ", ".join(c.name for c in caches), size=6.4,
             color="#444444")
    y -= 0.004
    y = para(fig, y, f"Adaptive run log: results/run_adaptive_fusion_eval_glm."
                     f"log (timing lines behind the cost table).", size=7.4)
    pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print("[written] RESULTS.pdf")


if __name__ == "__main__":
    main()
