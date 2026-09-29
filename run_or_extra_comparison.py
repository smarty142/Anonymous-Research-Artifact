"""Finish the comparator panel via OpenRouter.

Adds two prompted frontier baselines served by OpenRouter:
  * z-ai/glm-5.2  (frontier comparator, prompted)
  * openai/gpt-4o-mini  (a cross-family / OpenAI comparator)

Caches per-trajectory scores (results/cache/or_<model>__<bench>.jsonl,
resume-on-interrupt) and evaluates each alone vs the det layer and the
full layered system, on InjecAgent (FPR+TPR), tool-call (FPR+TPR), and
AgentDojo (TPR only). Gives the regime result a 3-frontier-model +
1-trained-guard comparator panel.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.injecagent import InjecAgentBench
from benchmarks.agentdojo import AgentDojoBench
from baselines.gateway_glm import GatewayGLMVerifier
from baselines.deterministic import DeterministicVerifier
from run_toolcall_bench import build_bench, _Cached, _eval
from pipeline.input_filter import InputFilter
from pipeline.context_guard import ContextGuard
from pipeline.tool_gateway import ToolGateway
from pipeline.layered_system import LayeredProtectionSystem
from evaluation import stats
from schemas import Decision

MODELS = [("z-ai/glm-5.2", "glm-5.2"), ("openai/gpt-4o-mini", "gpt-4o-mini"),
          ("anthropic/claude-haiku-4.5", "claude-haiku-4-5")]


def _run(model_id, slug, name, trajs, workers=8):
    p = Path(f"results/cache/or_{slug}__{name}.jsonl")
    cache = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line); cache[int(r["idx"])] = r["score"]
    todo = [(i, trajs[i]) for i in range(len(trajs)) if i not in cache]
    if todo:
        tf = GatewayGLMVerifier(model=model_id, base_url="https://openrouter.ai/api/v1",
                                 api_key=os.environ["OPENROUTER_API_KEY"], max_tokens=300)
        if not tf._available:
            print(f"[abort] {slug}:{name} no key"); return None
        t0 = time.perf_counter()
        scores = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(tf.score, t): i for i, t in todo}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]; scores[i] = round(float(fut.result()), 4); done += 1
                if done % 100 == 0:
                    print(f"  [{slug}:{name}] {done}/{len(todo)} ({done/(time.perf_counter()-t0):.1f}/s)", flush=True)
        cache.update(scores)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for i in sorted(cache): f.write(json.dumps({"idx": i, "score": cache[i]}) + "\n")
        print(f"[{slug}:{name}] done {len(cache)}/{len(trajs)} in {time.perf_counter()-t0:.0f}s", flush=True)
    return [cache[i] for i in range(len(trajs))]


def main():
    benches = {"injecagent": list(InjecAgentBench(max_trajs=510))}
    tc, _, _ = build_bench(n_per_pattern=20, benign_reps=30, seed=0)
    benches["toolcall"] = tc
    benches["agentdojo"] = list(AgentDojoBench(max_trajs=280))

    inp, ctx, gw = InputFilter(), ContextGuard(), ToolGateway()
    det = DeterministicVerifier()

    out = {}
    for model_id, slug in MODELS:
        print(f"\n##### {model_id} #####", flush=True)
        out[slug] = {}
        for name, trajs in benches.items():
            scores = _run(model_id, slug, name, trajs)
            if scores is None: continue
            labels = [t.ground_truth == "malicious" for t in trajs]
            has_benign = (len(labels) - sum(labels)) > 0
            block = {"n": len(labels), "n_mal": int(sum(labels))}
            block["alone"] = _eval(_Cached(scores), trajs, labels) if has_benign else \
                {"tpr": round(100*sum(1 for s,l in zip(scores,labels) if l and s>=0.5)/max(1,sum(labels)),1)}
            if has_benign:
                block["det"] = _eval(det, trajs, labels)
                full = LayeredProtectionSystem(input_filter=inp, context_guard=ctx,
                                               tool_gateway=gw, output_filter=_Cached(scores))
                block["full-system"] = _eval(full, trajs, labels)
            else:
                block["det_tpr"] = round(100*sum(1 for t in trajs if t.ground_truth=='malicious' and det.verify(t).decision is not Decision.BENIGN)/max(1,sum(labels)),1)
            out[slug][name] = block
            _print_block(slug, name, block, has_benign)

    Path("results/run_or_extra_comparison.json").write_text(json.dumps(out, indent=2))
    print("\n[written] results/run_or_extra_comparison.json")


def _print_block(slug, name, b, has_benign):
    print(f"\n--- {slug} / {name} (n={b['n']}, mal={b['n_mal']}, benign={'yes' if has_benign else 'NO'}) ---")
    a = b["alone"]
    if "fpr" in a:
        print(f"  {slug:14s} alone   TPR={a['tpr']*100:5.1f}%  FPR={a['fpr']*100:4.1f}%")
        d = b["det"]; print(f"  {'det':14s}         TPR={d['tpr']*100:5.1f}%  FPR={d['fpr']*100:4.1f}%")
        f = b["full-system"]; print(f"  {'full-system':14s} TPR={f['tpr']*100:5.1f}%  FPR={f['fpr']*100:4.1f}%")
    else:
        print(f"  {slug:14s} alone   TPR={a['tpr']}%   | det TPR={b['det_tpr']}%")


if __name__ == "__main__":
    raise SystemExit(main())
