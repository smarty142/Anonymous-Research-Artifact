"""Render a harness JSON output to a paper-ready markdown report.

Usage:
    python -m evaluation.report results/run.json --out results/run.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(report: dict) -> str:
    """Render the harness JSON output as a markdown report."""
    out: list[str] = []
    bench = report["bench"]
    n = report["n_trajectories"]
    out.append(f"# Verification Report — `{bench}`")
    out.append("")
    out.append(f"**Trajectories evaluated:** {n}")
    out.append("")

    # ---- Headline table ----
    out.append("## Headline metrics")
    out.append("")
    out.append("| system | TPR | FPR | precision | F1 | coverage | mean_ms | p99_ms |")
    out.append("|---|---|---|---|---|---|---|---|")
    for s in report["systems"]:
        out.append(
            f"| `{s['system']}` | {_format_pct(s['tpr'])} | "
            f"{_format_pct(s['fpr'])} | {_format_pct(s['precision'])} | "
            f"{s['f1']:.3f} | {_format_pct(s['coverage'])} | "
            f"{s['mean_latency_ms']:.1f} | {s['p99_latency_ms']:.1f} |"
        )
    out.append("")

    # ---- Confusion counts ----
    out.append("## Confusion-matrix counts")
    out.append("")
    out.append("| system | TP | FP | TN | FN |")
    out.append("|---|---|---|---|---|")
    for s in report["systems"]:
        out.append(
            f"| `{s['system']}` | {s['tp']} | {s['fp']} | {s['tn']} | {s['fn']} |"
        )
    out.append("")

    # ---- Coverage matrix (per system x per pattern) ----
    cov = report.get("coverage", [])
    if cov:
        systems = sorted({row["system"] for row in cov})
        # patterns preserve order from first appearance
        seen: list[str] = []
        for row in cov:
            if row["pattern"] not in seen:
                seen.append(row["pattern"])
        patterns = seen

        # Build system -> pattern -> result
        grid: dict[str, dict[str, str]] = defaultdict(dict)
        for row in cov:
            grid[row["system"]][row["pattern"]] = row["result"]

        out.append("## Coverage matrix (per pattern)")
        out.append("")
        out.append("`H` = hit, `M` = missed. Empty = system did not evaluate that pattern.")
        out.append("")
        out.append("| pattern | " + " | ".join(f"`{s}`" for s in systems) + " |")
        out.append("|---" + "|---" * len(systems) + "|")
        for pid in patterns:
            cells = []
            for s in systems:
                r = grid[s].get(pid, "")
                if r == "hit":
                    cells.append("**H**")
                elif r == "miss":
                    cells.append("M")
                else:
                    cells.append("·")
            out.append(f"| `{pid}` | " + " | ".join(cells) + " |")
        out.append("")

    # ---- Coverage summary per system ----
    out.append("## Coverage summary")
    out.append("")
    for s in report["systems"]:
        sys_name = s["system"]
        hits = sorted(r["pattern"] for r in cov if r["system"] == sys_name and r["result"] == "hit")
        misses = sorted(r["pattern"] for r in cov if r["system"] == sys_name and r["result"] == "miss")
        out.append(f"### `{sys_name}` — {_format_pct(s['coverage'])} coverage "
                   f"({len(hits)}/{len(hits) + len(misses)})")
        out.append("")
        if hits:
            out.append(f"- **Hits:** {', '.join(f'`{h}`' for h in hits)}")
        if misses:
            out.append(f"- **Misses:** {', '.join(f'`{m}`' for m in misses)}")
        out.append("")

    # ---- Footnotes ----
    out.append("---")
    out.append("")
    out.append("*Produced by the released evaluation scripts. TPR = attack recall; FPR = benign "
               "false-alarm rate; coverage = fraction of distinct attack patterns caught.*")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="harness JSON output (results/run.json)")
    p.add_argument("--out", required=True, help="markdown output path")
    args = p.parse_args(argv)

    report = json.loads(Path(args.input).read_text())
    md = render_markdown(report)
    Path(args.out).write_text(md)
    print(f"[report] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())