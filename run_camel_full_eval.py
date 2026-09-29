"""Full CaMeL (P-LLM + Q-LLM + execution) via OpenRouter/Claude Haiku 4.5,
on InjecAgent 510 + AgentDojo 280 + tool-call 480, with per-trajectory
caching (resume-on-interrupt). Replaces the §5.10 execution-layer-only
comparison with a full-CaMeL data point.

Cost ~$2.7 total; CamelFullOR (OpenAI transport) is 100% reliable (probe).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from run_camel_full_probe import CamelFullOR
from benchmarks.injecagent import InjecAgentBench
from benchmarks.agentdojo import AgentDojoBench
from run_toolcall_bench import build_bench
from schemas import Decision

BENCHES = {
    "injecagent": lambda: list(InjecAgentBench(max_trajs=510)),
    "agentdojo":  lambda: list(AgentDojoBench(max_trajs=280)),
    "toolcall":   lambda: build_bench(n_per_pattern=20, benign_reps=30, seed=0)[0],
}


def _load(name):
    p = Path(f"results/cache/camel_full_or__{name}.jsonl")
    c = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line); c[int(r["idx"])] = int(r["mal"])
    return c, p


def main():
    cf = CamelFullOR(api_key=os.environ["OPENROUTER_API_KEY"])
    out = {}
    for name, loader in BENCHES.items():
        trajs = loader()
        cache, p = _load(name)
        todo = [(i, trajs[i]) for i in range(len(trajs)) if i not in cache]
        print(f"\n[{name}] n={len(trajs)} cached={len(cache)} todo={len(todo)}", flush=True)
        t0 = time.perf_counter(); done = 0; buf = []
        for i, t in todo:
            try:
                v = cf.verify(t)
                cache[i] = 1 if v.decision is not Decision.BENIGN else 0
            except Exception:
                cache[i] = 0
            buf.append({"idx": i, "mal": cache[i]}); done += 1
            if len(buf) >= 20:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    for r in buf: f.write(json.dumps(r) + "\n")
                buf = []
            if done % 50 == 0:
                print(f"  [{name}] {done}/{len(todo)} calls={cf._ncalls} "
                      f"({(time.perf_counter()-t0)/done:.1f}s/traj)", flush=True)
        if buf:
            with open(p, "a") as f:
                for r in buf: f.write(json.dumps(r) + "\n")

        labels = [t.ground_truth == "malicious" for t in trajs]
        verdicts = [bool(cache[i]) for i in range(len(trajs))]
        nm = sum(labels); nb = len(labels) - nm
        tp = sum(1 for v, l in zip(verdicts, labels) if l and v)
        fp = sum(1 for v, l in zip(verdicts, labels) if not l and v)
        tpr = round(100*tp/nm, 1) if nm else None
        fpr = round(100*fp/nb, 1) if nb else None
        out[name] = {"n": len(trajs), "n_mal": nm, "tpr": tpr, "fpr": fpr}
        print(f"[{name}] CaMeL-full TPR={tpr}% FPR={fpr}% (n_mal={nm})", flush=True)

    print("\n" + "=" * 70)
    print("Full CaMeL (Claude Haiku 4.5, P-LLM+Q-LLM+exec) vs §5.10 execution-layer:")
    for name, r in out.items():
        print(f"  {name:12s} TPR={r['tpr']}% FPR={r['fpr']}%")
    print("=" * 70)
    Path("results/run_camel_full_eval.json").write_text(json.dumps(out, indent=2))
    print("[written] results/run_camel_full_eval.json")

if __name__ == "__main__":
    raise SystemExit(main())
