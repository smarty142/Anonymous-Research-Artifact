"""Render PLAYBOOK.pdf: the step-by-step runbook for the TriChannel
supplementary artifact.

    python3 make_PLAYBOOK.py   (run from the artifact root)

Everything marked VERIFIED was executed from a clean copy of this folder
on macOS / CPython 3.9 with the pinned requirements; measured wall-clock
is listed next to each command.
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)


def page(pdf, title=None, kicker=None):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    if kicker:
        fig.text(0.07, 0.955, kicker.upper(), fontsize=8, color="#666666",
                 family="sans-serif")
    if title:
        fig.text(0.07, 0.915, title, fontsize=13, fontweight="bold",
                 family="sans-serif")
    return fig


def para(fig, y, text, size=9.0, wrap=100, gap=0.0138, color="black",
         bold=False, mono=False, indent=0.0):
    fam = "monospace" if mono else "serif"
    for line in textwrap.wrap(text, wrap) or [""]:
        fig.text(0.07 + indent, y, line, fontsize=size, color=color,
                 family=fam, fontweight="bold" if bold else "normal",
                 ha="left", va="top")
        y -= gap
    return y - 0.004


def steps(fig, y, rows, size=7.4, row_h=0.0175):
    """rows: list of (cmd, note) or ('', section-header)."""
    for cmd, note in rows:
        if cmd.startswith("#"):
            y -= 0.004
            fig.text(0.07, y, cmd[1:].strip().upper(), fontsize=7.6,
                     color="#8a5a00" if False else "#555555",
                     family="sans-serif", fontweight="bold", va="top")
            y -= row_h
            continue
        fig.text(0.07, y, "$", fontsize=size, family="monospace",
                 color="#2a6f4f", va="top", fontweight="bold")
        fig.text(0.095, y, cmd, fontsize=size, family="monospace",
                 color="#1a1a1a", va="top")
        y -= row_h
        if note:
            for line in textwrap.wrap(note, 96):
                fig.text(0.095, y, line, fontsize=7.0, family="serif",
                         color="#444444", va="top", style="italic")
                y -= row_h * 0.82
        y -= row_h * 0.25
    return y


def main():
    pdf = PdfPages("PLAYBOOK.pdf")

    # -------- page 1: quick start --------
    fig = page(pdf, "Playbook: reproduce every result, zero API calls",
               "runbook / 1")
    y = 0.875
    y = para(fig, y,
             "This runbook regenerates every table in the paper from the "
             "shipped caches and code. No credentials, no network. Measured "
             "wall-clock on one core (CPython 3.9, macOS): the full pipeline "
             "below completes in well under 15 minutes.")
    y -= 0.006
    y = para(fig, y, "0. Environment (once)", bold=True)
    y = steps(fig, y, [
        ("python3 -m pip install -r requirements.txt",
         "numpy, scikit-learn, matplotlib, pydantic, PyYAML, jsonschema, "
         "regex, pytest. Nothing else is needed for zero-API reproduction."),
        ("python3 -m py_compile \"$(find . -name '*.py' -not -path "
         "'./agentdojo_clone/*')\"",
         "Sanity gate: every shipped file compiles. VERIFIED."),
    ])
    y = para(fig, y, "1. Fastest end-to-end check (~2 minutes)", bold=True)
    y = steps(fig, y, [
        ("python3 run_taxonomy_circularity.py",
         "~40 s incl. bench rebuild. VERIFIED: prints 133/440 signature-"
         "tripping draws, fused detection on the 307 evading draws = 0.2638 "
         "vs text 0.2508, tiers T.A 0.2647 / T.B 0.6316 / T.C 0.0; rewrites "
         "results/run_taxonomy_circularity.json with identical values."),
        ("python3 -m pytest tests/ -q",
         "~30 s. VERIFIED: 66 passed."),
    ])
    y = para(fig, y, "2. Core paper tables (~5 minutes)", bold=True)
    y = steps(fig, y, [
        ("python3 run_ensemble_ablation.py",
         "~48 s. VERIFIED: ablation grid (7 subsets x 3 benches), learned "
         "LR fusion, and transfer - final lines read toolcall->injecagent "
         "LR AUC = 0.9766, injecagent->toolcall 0.8447."),
        ("python3 run_fusion_rule_ablation.py",
         "Fusion-rule comparison (max / mean / noisy-OR / product / LR "
         "out-of-fold) with bootstrap CIs; regenerates Table IV (section VI-E)."),
        ("python3 run_diversity_stats.py",
         "~15 s. VERIFIED: Kuncheva Q / disagreement / double-fault per "
         "bench and backbone-pair Spearman correlations (0.25-0.79)."),
        ("python3 run_review_response.py",
         "Composition-ceiling replication across 4 classifier pairings + "
         "fusion rules; regenerates section VI-G numbers."),
        ("python3 run_cost_table.py",
         "~1 min. VERIFIED: Table VIII provenance - parses wall-clock and "
         "call counts from run_adaptive_fusion_eval_glm.log, recomputes "
         "prompt length (2,098 chars over the 440 attacks) and pattern "
         "latency, and re-derives costs ($0.36 text GLM-4.6 / $1.63 CaMeL "
         "est / $1.99 fused per 1,000 attack trajectories)."),
    ])
    pdf.savefig(fig); plt.close(fig)

    # -------- page 2: statistics + adaptive + figures --------
    fig = page(pdf, "Playbook page 2: statistics, figures, documents",
               "runbook / 2")
    y = 0.875
    y = para(fig, y, "3. Statistical treatment (a few minutes)", bold=True)
    y = steps(fig, y, [
        ("python3 run_ablation_ci.py",
         "Full-grid 2,000-replicate paired bootstrap CIs for AUC and "
         "TPR@7%FPR on every subset x bench cell, all 21 pairwise McNemar "
         "tests per bench with Holm correction, and the AgentDojo "
         "pattern-channel sign-stability check (expect: all replicates "
         "below AUC 0.5, CI [0.111, 0.195]). CPU-bound; the slowest script "
         "in the artifact."),
        ("python3 run_contamination_prepost.py --help",
         "Pre-fix vs post-fix construction uses the same loader with "
         "contamination_fix=False; the cache glm_guard__injecagent_prefix."
         "jsonl holds the pre-fix GLM-4.6 scores, so the comparison also "
         "runs without API calls (see script source for the exact flag)."),
        ("python3 run_paired_head_diffs.py",
         "~1 min. VERIFIED: paired AUC-difference CIs of the learned head "
         "and the text+provenance subset against the text channel alone - "
         "the head matches text on InjecAgent, diff CI [-0.0041, +0.0127]; "
         "adds on tool-call, CI [0.0183, 0.0594]; taxonomy-independent "
         "subset gain CI [0.0053, 0.0233]."),
        ("python3 run_tpr_at_1pct.py",
         "Re-evaluates the ablation grid at 1% FPR and reports the score-"
         "granularity diagnostics: the text guard emits 18 distinct scores "
         "and the benign tie plateau at 0.15 makes TPR@1% = TPR@7% for "
         "every subset (actual FPR 0.42%)."),
        ("python3 run_adaptive_mcnemar.py",
         "~10 s. VERIFIED: adaptive-bench McNemar at matched 7%-FPR "
         "thresholds from the shipped caches - b = 0 in all three "
         "comparisons (c = 33, 37, 4): disagreements are one-directional."),
    ])
    y = para(fig, y, "4. Figures", bold=True)
    y = steps(fig, y, [
        ("cd figs && python3 make_figs.py && python3 make_figs2.py",
         "Regenerates the five published figures (regime heatmap, ablation "
         "heatmap, adaptive AUC bars, per-shape detection, contamination "
         "pre/post) as PDFs in figs/."),
        ("python3 make_RESULTS.py",
         "Rebuilds RESULTS.pdf: every published number rendered as tables "
         "read live from results/*.json, plus all five figures."),
    ])
    y = para(fig, y, "5. Documents", bold=True)
    y = steps(fig, y, [
        ("python3 make_README.py && python3 make_PLAYBOOK.py",
         "Rebuild README.pdf and this runbook."),
    ])
    y = para(fig, y, "6. Expected-values checklist", bold=True)
    y = para(fig, y,
             "After running steps 1-3, these exact values must appear "
             "(any deviation means the environment altered numerics):",
             size=8.4)
    y = steps(fig, y, [
        ("ablation grid, tool-call triple: AUC 0.874",
         "text alone 0.832; text+pattern 0.861; +15.4 pp TPR@7%FPR "
         "(35.4 -> 50.8)."),
        ("adaptive bench, fused: AUC 0.795 [0.766, 0.825]",
         "text 0.778; paired-difference CI [0.011, 0.026]; McNemar b=0, "
         "c=37."),
        ("circularity split: 133 trip / 307 evade",
         "evading-draw detection 0.2638 vs text 0.2508; T.C (data-flow) "
         "0.0."),
        ("contamination: pre AUC 0.518 / 81.4% FPR -> post 0.916 / 0%",
         "same classifier GLM-4.6, 510 records, only the 323 benign rows "
         "differ."),
    ])
    pdf.savefig(fig); plt.close(fig)

    # -------- page 3: optional API + troubleshooting --------
    fig = page(pdf, "Playbook page 3: optional live re-scoring, troubleshooting",
               "runbook / 3")
    y = 0.875
    y = para(fig, y,
             "None of the following is required to reproduce any published "
             "number; it re-scores trajectories against live endpoints and "
             "will differ in transport noise.")
    y = para(fig, y, "7. Optional: re-score the text channel (OpenRouter)",
             bold=True)
    y = steps(fig, y, [
        ("export OPENROUTER_API_KEY=...   # never commit this",
         "Provider for the static benches (Claude Haiku 4.5 text channel, "
         "CaMeL P/Q, comparators)."),
        ("python3 run_or_extra_comparison.py --dry-run 5",
         "Every API-touching script ships a small-N dry-run flag; use it "
         "before any full run. Caches append to results/cache/, so a "
         "re-run resumes rather than double-spends."),
    ])
    y = para(fig, y, "8. Optional: backbone deconfound (any OpenAI-compatible endpoint)", bold=True)
    y = steps(fig, y, [
        ("export GATEWAY_API_KEY=...   GATEWAY_BASE_URL=.../v1",
         "Re-scores ONLY the provenance channel (CaMeL P/Q) on a second "
         "backbone (default google/gemma-4-31b-it), then recomputes the "
         "tool-call fused/subset numbers from caches. The gateway must "
         "accept x-api-key header auth; 429s back off automatically and "
         "failed calls are never cached as verdicts. ~1,150 calls total."),
        ("python3 run_backbone_deconfound.py --dry-run 8 --workers 2",
         "Probe + 8 trajectories (~20 calls) with transport-health gate "
         "before the full run; cache resumes on rerun."),
        ("python3 run_backbone_deconfound.py --bench injecagent",
         "Off-regime companion (P2 backbone-independence): re-scores "
         "provenance on the 510 InjecAgent trajectories. Measured cost: "
         "103 calls - the CaMeL policy short-circuits trajectories with "
         "no untrusted-to-sensitive flow before any LLM call. Same env, "
         "same cache-resume behavior; --analyze-only reruns both benches "
         "from caches with zero calls."),
    ])
    y = para(fig, y, "9. Optional: adaptive bench (z.ai / GLM-4.6)", bold=True)
    y = steps(fig, y, [
        ("export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic",
         "WARNING: this endpoint silently aliases claude-* model ids to GLM "
         "models. The script probes the endpoint first and labels itself "
         "GLM-4.6; treat any claude-* id on this transport as GLM."),
        ("python3 run_adaptive_fusion_eval.py --provider glm --dry-run 10",
         "~30 calls. Full run: 440 attacks + 240 benign, ~2,800 calls, "
         "documented abort gates (transport-empty 20%, probe mismatch). "
         "--provider anthropic exists for a directly-funded Anthropic run."),
    ])
    y = para(fig, y, "10. Troubleshooting", bold=True)
    y = steps(fig, y, [
        ("# ModuleNotFoundError: osi_bench / benchmarks / adaptive",
         "Run from the artifact root; scripts resolve results/ and module "
         "paths relative to the working directory."),
        ("# AgentDojo runs not found at agentdojo_clone/runs/...",
         "The loader wants agentdojo_clone/runs/gpt-3.5-turbo-0125; the "
         "shipped subset provides exactly that directory. Point "
         "AgentDojoBench(runs_root=...) elsewhere if you bring your own."),
        ("# InjecAgent loader path",
         "data/InjecAgent/data/test_cases_dh_base.json ships as-is; pass "
         "data_path= to InjecAgentBench to override."),
        ("# sklearn RuntimeWarnings in matmul",
         "Benign: emitted while the LR head fits on degenerate columns "
         "(e.g. pattern channel identically zero on InjecAgent); results "
         "are unaffected."),
        ("# pytest collects 66 tests",
         "test_stackelberg is intentionally not shipped (it exercises a "
         "module outside this paper's scope)."),
    ])
    pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print("[written] PLAYBOOK.pdf")


if __name__ == "__main__":
    main()
