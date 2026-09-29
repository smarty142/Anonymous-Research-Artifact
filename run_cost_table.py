"""Cost and latency table for the adaptive run (paper Table VIII).

Every number in the paper's cost table is either measured in the shipped
run log (results/run_adaptive_fusion_eval_glm.log), recomputed here from
shipped data, or a documented estimate. This script makes the provenance
explicit and regenerates the table zero-API:

  measured (parsed from the log)
    - text: 440 attacks / 240 benign wall-clock at 8-way concurrency
    - CaMeL: call counts, empty-response counts, wall-clock (sequential)
  recomputed (zero API)
    - mean guard-prompt length over the same 680 trajectories
      (baselines.glm_guard._GUARD_PROMPT + constitutional._render_trajectory)
    - pattern-channel latency per trajectory, timed locally
  estimates (stated as such in the paper's table caption)
    - tokens: input = prompt chars / 4 (the paper's own "about 513");
      text output = one score line (~20 tokens); CaMeL 600 in / 150 out
    - prices: September 2026 list prices per million tokens

Outputs results/run_cost_table.json.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from adaptive.generator import AdaptiveBench
from baselines.constitutional import _render_trajectory
from baselines.glm_guard import _GUARD_PROMPT
from pipeline.conformal import detect_patterns, nonconformity_score
from run_toolcall_bench import build_bench

LOG = Path("results/run_adaptive_fusion_eval_glm.log")

# September 2026 list prices, USD per million tokens.
PRICES = {"glm-4.6": {"in": 0.60, "out": 2.20},
          "claude-haiku-4-5": {"in": 1.00, "out": 5.00}}
TEXT_WORKERS = 8          # text channel concurrency in the logged run
CAMEL_TOKENS = {"in": 600, "out": 150}   # estimate, see paper caption
TEXT_OUT_TOKENS = 20                     # one score line, estimate

DONE = {
    "text_atk": r"\[text:adaptive\] done 440/440 in (\d+)s",
    "text_ben": r"\[text:adaptive-benign\] done 240/240 in (\d+)s",
    "camel_atk": r"\[camel:adaptive\] done 440/440 \((\d+) calls, (\d+) empty\) in (\d+)s",
    "camel_ben": r"\[camel:adaptive-benign\] done 240/240 \((\d+) calls, (\d+) empty\) in (\d+)s",
}


def main() -> None:
    log = LOG.read_text()
    m = {k: re.search(p, log) for k, p in DONE.items()}
    missing = [k for k, v in m.items() if v is None]
    if missing:
        sys.exit(f"[abort] log is missing lines for: {missing}")

    text_atk_s = int(m["text_atk"].group(1))
    text_ben_s = int(m["text_ben"].group(1))
    camel_atk_calls, camel_atk_empty, camel_atk_s = map(int, m["camel_atk"].groups())
    camel_ben_calls, camel_ben_empty, camel_ben_s = map(int, m["camel_ben"].groups())

    # ---- rebuild the same 680 trajectories (deterministic) ----
    atk = list(AdaptiveBench(seed=0, n_per_pattern=20))
    assert len(atk) == 440
    tc, _, _ = build_bench(n_per_pattern=20, benign_reps=30, seed=0)
    ben = [t for t in tc if t.ground_truth != "malicious"]
    assert len(ben) == 240
    trajs = atk + ben

    prompts = [len(_GUARD_PROMPT.format(trajectory=_render_trajectory(t)))
               for t in trajs]
    mean_chars = statistics.mean(
        len(_GUARD_PROMPT.format(trajectory=_render_trajectory(t))) for t in atk)
    mean_chars_all = statistics.mean(prompts)
    est_in_tokens = mean_chars / 4.0        # chars/4 token estimate

    # ---- pattern-channel timing (local, single-threaded) ----
    for t in trajs[:20]:                    # warm-up
        nonconformity_score(detect_patterns(t))
    t0 = time.perf_counter()
    for t in trajs:
        nonconformity_score(detect_patterns(t))
    pattern_ms = 1000.0 * (time.perf_counter() - t0) / len(trajs)

    # ---- derived per-trajectory / per-call figures ----
    camel_calls_per_traj = camel_atk_calls / 440
    text_s_per_traj = text_atk_s / 440
    text_s_per_call = text_atk_s * TEXT_WORKERS / 440
    camel_s_per_traj = camel_atk_s / 440
    camel_s_per_call = camel_atk_s / camel_atk_calls
    ben_calls_per_traj = camel_ben_calls / 240
    ben_s_per_traj = camel_ben_s / 240

    def cost_1k(in_tok, out_tok, model):
        p = PRICES[model]
        return (in_tok * p["in"] + out_tok * p["out"]) / 1000.0

    text_tok_in = round(est_in_tokens)
    cost_text_glm = cost_1k(text_tok_in, TEXT_OUT_TOKENS, "glm-4.6")
    cost_text_haiku = cost_1k(text_tok_in, TEXT_OUT_TOKENS, "claude-haiku-4-5")
    cost_camel = camel_calls_per_traj * cost_1k(CAMEL_TOKENS["in"],
                                                CAMEL_TOKENS["out"], "glm-4.6")

    out = {
        "config": {
            "source_log": LOG.name, "n_attacks": 440, "n_benign": 240,
            "text_workers": TEXT_WORKERS,
            "prices_usd_per_mtok_sept2026": PRICES,
            "estimates": {
                "text_input_tokens": "prompt chars / 4 (paper: 'about 513')",
                "text_output_tokens": TEXT_OUT_TOKENS,
                "camel_tokens": CAMEL_TOKENS,
            },
        },
        "measured_from_log": {
            "text_attacks_s": text_atk_s, "text_benign_s": text_ben_s,
            "camel_attack_calls": camel_atk_calls,
            "camel_attack_empty": camel_atk_empty,
            "camel_attack_s": camel_atk_s,
            "camel_benign_calls": camel_ben_calls,
            "camel_benign_empty": camel_ben_empty,
            "camel_benign_s": camel_ben_s,
        },
        "recomputed": {
            "mean_guard_prompt_chars_attacks": round(mean_chars, 1),
            "mean_guard_prompt_chars_all_680": round(mean_chars_all, 1),
            "mean_guard_prompt_tokens_est": text_tok_in,
            "pattern_ms_per_trajectory": round(pattern_ms, 1),
            "text_s_per_trajectory_8way": round(text_s_per_traj, 2),
            "text_s_per_call_est": round(text_s_per_call, 2),
            "camel_calls_per_trajectory": round(camel_calls_per_traj, 2),
            "camel_s_per_trajectory_sequential": round(camel_s_per_traj, 2),
            "camel_s_per_call": round(camel_s_per_call, 2),
            "camel_benign_calls_per_trajectory": round(ben_calls_per_traj, 3),
            "camel_benign_s_per_trajectory": round(ben_s_per_traj, 2),
        },
        "cost_usd_per_1000_attack_trajectories": {
            "text_glm46": round(cost_text_glm, 2),
            "text_claude_haiku45": round(cost_text_haiku, 2),
            "camel_glm46_est": round(cost_camel, 2),
            "fused_max": round(cost_text_glm + cost_camel, 2),
            "pattern": 0.0,
        },
    }
    Path("results/run_cost_table.json").write_text(json.dumps(out, indent=2))

    r = out["recomputed"]; c = out["cost_usd_per_1000_attack_trajectories"]
    print("=== cost table (adaptive run, Table VIII provenance) ===")
    print(f"text:  {r['text_s_per_trajectory_8way']} s/traj @ {TEXT_WORKERS}-way "
          f"({r['text_s_per_call_est']} s/call est), "
          f"{r['mean_guard_prompt_chars_attacks']} chars mean prompt (attacks)")
    print(f"camel: {r['camel_calls_per_trajectory']} calls/traj, "
          f"{r['camel_s_per_trajectory_sequential']} s/traj sequential "
          f"({r['camel_s_per_call']} s/call)")
    print(f"benign camel: {r['camel_benign_calls_per_trajectory']} calls/traj, "
          f"{r['camel_benign_s_per_trajectory']} s/traj")
    print(f"pattern: {r['pattern_ms_per_trajectory']} ms/traj local")
    print(f"cost/1k: text glm {c['text_glm46']} | text haiku "
          f"{c['text_claude_haiku45']} | camel {c['camel_glm46_est']} | "
          f"fused {c['fused_max']}")
    print("[written] results/run_cost_table.json")


if __name__ == "__main__":
    main()
