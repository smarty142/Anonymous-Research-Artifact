"""McNemar: full system vs the two new OpenRouter comparators on
independent AgentDojo attacks (the tightest comparisons).

vs GPT-4o-mini the raw margin is only +8pp (92.5% vs 84.3%); this tests
whether it is nonetheless a significant paired difference.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from benchmarks.agentdojo import AgentDojoBench
from pipeline.input_filter import InputFilter
from pipeline.context_guard import ContextGuard
from pipeline.tool_gateway import ToolGateway
from pipeline.layered_system import LayeredProtectionSystem
from evaluation import stats
from schemas import Action, Decision, Verdict

def load_or(slug):
    c = {}
    for line in Path(f"results/cache/or_{slug}__agentdojo.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); c[int(r["idx"])] = r["score"]
    return [c[i] for i in range(len(c))]

class BoolV:
    name="x"; version="x"
    def __init__(s, b): s._b=list(b); s._i=0
    def verify(s, t):
        m = s._b[s._i] if s._i < len(s._b) else False; s._i += 1
        return Verdict(decision=Decision.MALICIOUS if m else Decision.BENIGN, confidence=0,
                       action=Action.BLOCK if m else Action.ALLOW)

def main():
    adj = list(AgentDojoBench(max_trajs=280))
    # GLM-4.6 scores drive the system's output filter (the 92.5% figure)
    gc = {}
    for line in Path("results/cache/glm_guard__agentdojo.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); gc[int(r["idx"])] = r["score"]
    glm_out = [gc[i] >= 0.5 for i in range(len(gc))]
    inp, ctx, gw = InputFilter(), ContextGuard(), ToolGateway()
    full = LayeredProtectionSystem(input_filter=inp, context_guard=ctx,
                                   tool_gateway=gw, output_filter=BoolV(glm_out))
    preds_full = [full.verify(t).decision is not Decision.BENIGN for t in adj]
    print(f"full-system AgentDojo TPR = {100*sum(p for p in preds_full)/len(preds_full):.1f}%")
    for slug in ["glm-5.2", "gpt-4o-mini", "claude-haiku-4-5"]:
        sc = load_or(slug)
        preds = [s >= 0.5 for s in sc]
        mc = stats.mcnemar([True]*len(adj), preds, preds_full)
        tpr = 100*sum(preds)/len(preds)
        print(f"\n{slug}: alone TPR={tpr:.1f}%  McNemar (system vs {slug}): "
              f"b={mc['b']} c={mc['c']} p={mc['p_value']:.4f}")

if __name__ == "__main__":
    raise SystemExit(main())
