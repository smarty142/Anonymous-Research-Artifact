"""Render README.pdf for the TriChannel supplementary artifact.

Self-contained: uses only matplotlib. Run from the artifact root:

    python3 make_README.py

Writes README.pdf (replaces the markdown readme; this artifact ships
no markdown documentation by design).
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)
BODY = 9.2
HEAD = 13
MONO = 7.6


def page(pdf, title=None, kicker=None):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    if kicker:
        fig.text(0.07, 0.955, kicker.upper(), fontsize=8, color="#666666",
                 family="sans-serif")
    if title:
        fig.text(0.07, 0.915, title, fontsize=HEAD, fontweight="bold",
                 family="sans-serif")
    return fig


def para(fig, y, text, size=BODY, wrap=98, gap=0.014, color="black",
         bold=False, mono=False, indent=0.0):
    fam = "monospace" if mono else "serif"
    for line in textwrap.wrap(text, wrap) or [""]:
        fig.text(0.07 + indent, y, line, fontsize=size, color=color,
                 family=fam, fontweight="bold" if bold else "normal",
                 ha="left", va="top")
        y -= gap
    return y - 0.004


def table(fig, y, rows, col_frac=(0.30, 0.38, 0.32), size=7.3, header=True,
          row_h=0.019):
    import textwrap
    x0 = 0.07
    for r, row in enumerate(rows):
        x = x0
        top = header and r == 0
        maxlines = 1
        for c, cell in enumerate(row):
            w = max(10, int(col_frac[c] * 120))
            lines = textwrap.wrap(str(cell), w) or [""]
            maxlines = max(maxlines, len(lines))
            for li, line in enumerate(lines):
                fig.text(x, y - li * row_h * 0.8, line, fontsize=size,
                         family="monospace" if not top else "sans-serif",
                         fontweight="bold" if top else "normal",
                         color="#1a1a1a" if not top else "#000000",
                         ha="left", va="top")
            x += col_frac[c]
        if top:
            fig.lines.append(plt.Line2D([x0, x0 + sum(col_frac[:len(row)])],
                                        [y - 0.004, y - 0.004],
                                        color="#999999", lw=0.7))
        y -= row_h * (1 + 0.8 * (maxlines - 1))
    return y - 0.006


def main():
    pdf = PdfPages("README.pdf")

    # ---------------- page 1: what this is ----------------
    fig = page(pdf, "TriChannel - Supplementary Code and Data",
               "readme / anonymized for review")
    y = 0.875
    y = para(fig, y,
             "Anonymized code, per-trajectory score caches, benchmark data, "
             "and result files for the IEEE SaTML 2027 submission \"TriChannel: "
             "Signal-Support-Aware Fusion for Prompt-Injection Detection in "
             "LLM Agents\".")
    y = para(fig, y,
             "Every table and figure in the paper regenerates from this "
             "folder with ZERO API CALLS: the per-trajectory channel scores "
             "are shipped in results/cache/, the pattern channel is a local "
             "deterministic verifier, and each analysis script re-derives its "
             "statistics (bootstrap CIs, McNemar tests, fusion rules, learned "
             "heads) from those inputs.", bold=False)
    y -= 0.008
    y = para(fig, y, "Documents in this artifact", bold=True)
    y = table(fig, y, [
        ("document", "contents"),
        ("README.pdf", "this file: layout, table-to-file map, cache format, providers"),
        ("PLAYBOOK.pdf", "step-by-step runbook: order, commands, expected numbers,"),
        ("", "runtimes, verification gates, optional live re-scoring"),
        ("RESULTS.pdf", "every published number and all five figures, generated"),
        ("", "directly from the result JSONs by make_RESULTS.py"),
    ], col_frac=(0.18, 0.80), row_h=0.021)
    y = para(fig, y,
             "RESULTS.pdf is produced by make_RESULTS.py, which loads the "
             "JSONs in results/ at render time - no numbers are transcribed "
             "by hand anywhere in that document.")
    pdf.savefig(fig); plt.close(fig)

    # ---------------- page 2: layout + map ----------------
    fig = page(pdf, "Repository layout", "readme / 2")
    y = 0.875
    y = table(fig, y, [
        ("path", "contents"),
        ("run_*.py", "experiment drivers (one per paper section - see map)"),
        ("schemas.py", "trajectory / tool-call / decision dataclasses"),
        ("adaptive/", "adaptive paraphrase generator (seed 0, 440 draws)"),
        ("benchmarks/", "InjecAgent / AgentDojo / tool-call loaders"),
        ("baselines/", "channels: glm_guard (text), camel_full (provenance),"),
        ("", "deterministic (pattern, 48 signatures), comparators"),
        ("pipeline/", "conformal verifier, tool gateway, layered system"),
        ("evaluation/", "AUC, TPR@FPR, paired bootstrap, McNemar, Holm, Kuncheva"),
        ("taxonomy/", "capability + sensitivity + threat-taxonomy YAML"),
        ("osi_bench/", "base trajectories + mutations for the tool-call bench"),
        ("tests/", "unit tests (pytest)"),
        ("results/", "result JSONs backing every paper table"),
        ("results/cache/", "per-trajectory score caches (see format below)"),
        ("figs/", "figure scripts + rendered PDFs"),
        ("data/InjecAgent/", "public InjecAgent release (Zhan et al. 2024)"),
        ("data/bipia/", "public BIPIA corpus subset (Yi et al. 2023)"),
        ("agentdojo_clone/runs/", "AgentDojo shipped runs, gpt-3.5-turbo-0125 subset"),
        ("make_README.py / make_PLAYBOOK.py /", "generators for the three PDF documents"),
        ("make_RESULTS.py", ""),
    ], col_frac=(0.26, 0.72), row_h=0.0185, size=7.0)
    pdf.savefig(fig); plt.close(fig)

    # ---------------- page 3: table map ----------------
    fig = page(pdf, "Paper table / figure to artifact map", "readme / 3")
    y = 0.875
    y = table(fig, y, [
        ("paper item", "result file", "script"),
        ("Fig. 3a regime heatmap; VI-B table", "run_or_extra_comparison.json +", "run_or_extra_comparison.py,"),
        ("", "run_camel_full_eval.json, run_toolcall_bench.json,", "run_camel_full_eval.py,"),
        ("", "run_agentdojo_matched_fpr.json", "run_toolcall_bench.py"),
        ("Fig. 3b + Table III ablation; VI-C", "run_ensemble_ablation.json", "run_ensemble_ablation.py"),
        ("VI-E learned fusion + rules", "run_fusion_rule_ablation.json", "run_fusion_rule_ablation.py"),
        ("VI-F transfer + paired-diff CIs", "run_ensemble_ablation.json +", "run_ensemble_ablation.py,"),
        ("", "run_paired_head_diffs.json", "run_paired_head_diffs.py"),
        ("VI-D backbone deconfound (live only)", "run_backbone_deconfound.json (+_injecagent)", "run_backbone_deconfound.py"),
        ("VI-C tighter operating points (1% FPR)", "run_tpr_at_1pct.json", "run_tpr_at_1pct.py"),
        ("VI-G composition ceiling", "run_review_response.json", "run_review_response.py"),
        ("VI-H contamination pre/post", "run_contamination_prepost.json", "run_contamination_prepost.py"),
        ("VI-I adaptive fused", "run_adaptive_fusion_eval_glm.json (+.log)",
         "run_adaptive_fusion_eval.py"),
        ("VI-I adaptive McNemar detail", "run_adaptive_mcnemar.json",
         "run_adaptive_mcnemar.py"),
        ("VI-J cost/latency (Table VIII)", "run_cost_table.json",
         "run_cost_table.py"),
        ("VI-I circularity split (Fig. 5b)", "run_taxonomy_circularity.json", "run_taxonomy_circularity.py"),
        ("VI-I layer bypass rates", "run_adaptive_layer_robustness.json", "run_adaptive_layer_robustness.py"),
        ("statistical treatment", "run_ablation_ci.json, run_diversity_stats.json,", "run_ablation_ci.py,"),
        ("", "all_mcnemar_counts.json", "run_diversity_stats.py"),
        ("IV-C cheap-feature chance", "run_signal_probe.json", "run_signal_probe.py"),
        ("VI-K enforcement layer", "run_blast_radius_eval.json", "run_blast_radius_eval.py"),
        ("figures", "figs/*.pdf", "figs/make_figs.py,"),
        ("", "", "figs/make_figs2.py"),
    ], col_frac=(0.27, 0.38, 0.30), row_h=0.0185, size=7.0)
    pdf.savefig(fig); plt.close(fig)

    # ---------------- page 4: caches, providers, data ----------------
    fig = page(pdf, "Cache format, providers, third-party data", "readme / 4")
    y = 0.875
    import json as _json
    pattern_ms = _json.load(
        open("results/run_cost_table.json"))["recomputed"][
            "pattern_ms_per_trajectory"]
    y = para(fig, y, "Cache format", bold=True)
    y = para(fig, y,
             "Each results/cache/<channel>__<benchmark>.jsonl file holds one "
             "JSON object per trajectory with a positional idx key matching "
             "the benchmark iteration order, plus the channel output (score "
             "for text, mal for provenance). The pattern channel is always "
             f"recomputed locally (deterministic, ~{pattern_ms:.0f} "
             "ms/trajectory, see run_cost_table.json) rather "
             "than cached. Because keying is positional, analysis scripts "
             "assert row-count alignment against the rebuilt benchmark "
             "before reading.")
    y -= 0.010
    y = para(fig, y, "Providers (disclosure)", bold=True)
    y = para(fig, y,
             "Static benches: text and CaMeL P/Q on Claude Haiku 4.5 via "
             "OpenRouter; comparators GLM-4.6, GLM-5.2, GPT-4o-mini, Claude "
             "Haiku 4.5 under one guard prompt, Llama Guard 4 shipped prompt.")
    y = para(fig, y,
             "Adaptive bench and contamination (paper VI-H/VI-I): text and "
             "CaMeL on GLM-4.6 served by z.ai. The z.ai endpoint silently "
             "aliases claude-* model ids to GLM models; every run was "
             "probe-gated before spend and the configuration is labeled "
             "GLM-4.6 throughout.")
    y = para(fig, y,
             "Backbone deconfound (paper VI-B note): only the provenance "
             "channel was re-scored on Gemma 4 31B via an OpenAI-compatible "
             "endpoint, on both the tool-call and the InjecAgent bench; "
             "per-trajectory scores ship in results/cache/ "
             "(camel_gw_*__*.jsonl), so the analysis reruns with no "
             "credentials. No credentials are required anywhere in this "
             "folder; re-scoring against live endpoints is optional "
             "(see PLAYBOOK.pdf).")
    y -= 0.010
    y = para(fig, y, "Third-party data", bold=True)
    y = para(fig, y,
             "data/InjecAgent/ and agentdojo_clone/runs/ are redistributed "
             "subsets of the public InjecAgent and AgentDojo releases, "
             "included so benchmark rebuilds (and the pre-fix contamination "
             "construction) reproduce without external downloads. They "
             "belong to their original authors (Zhan et al. 2024; "
             "Debenedetti et al. 2024).")
    y -= 0.010
    y = para(fig, y, "Determinism", bold=True)
    y = para(fig, y,
             "Benchmarks are seeded (seed=0 everywhere). "
             "run_adaptive_fusion_eval.py asserts that a second benchmark "
             "rebuild reproduces every trajectory id before any scoring; "
             "the same assertion pattern guards the circularity split.")
    pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print("[written] README.pdf")


if __name__ == "__main__":
    main()
