"""Evaluation harness CLI.

Usage:
    python -m evaluation.harness --bench osi-bench --systems det,always-allow,always-block
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

from evaluation.metrics import evaluate_many, coverage_table
from benchmarks.osi_bench import OSIBench


def _get_system(name: str):
    from baselines import (
        DeterministicVerifier, AlwaysAllow, AlwaysBlock, RandomBaseline,
    )
    table = {
        "det": DeterministicVerifier,
        "always-allow": AlwaysAllow,
        "always-block": AlwaysBlock,
        "random": lambda: RandomBaseline(p_malicious=0.5, seed=0),
        "prompt-shields": lambda: _safe_import("PromptShieldsVerifier"),
        "llama-guard":   lambda: _safe_import("LlamaGuardVerifier"),
        "constitutional": lambda: _safe_import("ConstitutionalVerifier"),
    }
    if name not in table:
        raise SystemExit(f"unknown system: {name}; choose from {list(table)}")
    return table[name]()


def _safe_import(basename: str):
    """Lazy import of an optional baseline. Returns a 'no-credentials' stub
    if the underlying SDK isn't installed or credentials aren't set."""
    import importlib
    try:
        mod = importlib.import_module(f"baselines.{_module_for(basename)}")
        cls = getattr(mod, basename)
        return cls()
    except Exception as e:
        # Return a stub verifier that always says benign with a note
        from baselines.base import BaseVerifier
        from schemas import Action, Decision, Trajectory, Verdict
        class _Stub(BaseVerifier):
            name = basename
            version = "stub"
            def _verify(self, trajectory: Trajectory) -> Verdict:
                return Verdict(decision=Decision.BENIGN, confidence=0.0,
                               action=Action.ALLOW,
                               notes=[f"{basename} unavailable: {e}"])
        return _Stub()


def _module_for(basename: str) -> str:
    return {
        "PromptShieldsVerifier": "prompt_shields",
        "LlamaGuardVerifier": "llama_guard",
        "ConstitutionalVerifier": "constitutional",
    }[basename]


def _get_bench(name: str, attacks_per_pattern: int, benign_count: int,
               bench_path: str | None):
    if bench_path:
        from osi_bench.generator import OSIBenchGenerator
        gen = OSIBenchGenerator.from_jsonl(bench_path)
        # wrap in a tiny list-like object
        from benchmarks.base import BenchInfo
        class _ListBench:
            def __init__(self, trajs):
                self._trajs = trajs
                self.info = BenchInfo(
                    name=f"jsonl:{bench_path}",
                    description="loaded from JSONL",
                    n_trajectories=len(trajs),
                    ground_truth_field="ground_truth",
                    attack_pattern_field="traj_id_suffix",
                )
            def __iter__(self): return iter(self._trajs)
            def __len__(self): return len(self._trajs)
        return _ListBench(gen)
    if name == "osi-bench":
        return OSIBench(attacks_per_pattern=attacks_per_pattern,
                        benign_count=benign_count, seed=0)
    if name == "agentdojo":
        from benchmarks import AgentDojoBench
        return AgentDojoBench()
    if name == "injecagent":
        from benchmarks import InjecAgentBench
        return InjecAgentBench()
    if name == "adaptive":
        from adaptive import AdaptiveBench
        return AdaptiveBench(seed=0, n_per_pattern=20)
    if name == "mixed-threat":
        from mixed_threat import MixedThreatBench
        return MixedThreatBench(seed=0, n_per_category=10)
    raise SystemExit(f"unknown bench: {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run verifier(s) on a benchmark.")
    p.add_argument("--bench", default="osi-bench",
                   choices=["osi-bench", "agentdojo", "injecagent", "adaptive", "mixed-threat"])
    p.add_argument("--systems", default="det,always-allow,always-block,random",
                   help="comma-separated system names")
    p.add_argument("--out", default="results/run.json",
                   help="output JSON path")
    p.add_argument("--attacks-per-pattern", type=int, default=5,
                   help="for osi-bench: attacks per pattern (default 5)")
    p.add_argument("--benign-count", type=int, default=50,
                   help="for osi-bench: benign trajectory count (default 50)")
    p.add_argument("--bench-path", default=None,
                   help="load bench from a JSONL file (overrides --bench)")
    args = p.parse_args(argv)

    bench = _get_bench(args.bench, args.attacks_per_pattern,
                       args.benign_count, args.bench_path)
    sys_names = [s.strip() for s in args.systems.split(",") if s.strip()]
    systems = [_get_system(s) for s in sys_names]

    print(f"[harness] bench={args.bench} n_trajs={len(bench)} "
          f"systems={sys_names}", file=sys.stderr)

    metrics_list = evaluate_many(systems, bench)
    metrics_by_name = {m.name: m for m in metrics_list}

    # Build output
    out = {
        "bench": args.bench,
        "n_trajectories": len(bench),
        "systems": [m.as_row() for m in metrics_list],
        "coverage": coverage_table(metrics_by_name),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # Stdout summary (markdown table)
    print("\n## Headline metrics\n")
    print("| system | n | TPR | FPR | precision | F1 | coverage | mean_ms | p99_ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in metrics_list:
        print(f"| {m.name} | {m.n_total} | {m.tpr:.3f} | {m.fpr:.3f} | "
              f"{m.precision:.3f} | {m.f1:.3f} | {m.coverage:.3f} | "
              f"{m.mean_latency_ms:.2f} | {m.p99_latency_ms:.2f} |")

    print(f"\n[written] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())