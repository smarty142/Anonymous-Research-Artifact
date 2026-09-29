"""Tests for the BIPIA loader (Yi et al., arXiv:2312.14197)."""
import json

import pytest

from benchmarks.bipia import BipiABench


@pytest.fixture(scope="module")
def bench():
    return BipiABench(n_per_attack=6, n_benign_per_task=50, seed=0)


def test_construction_counts(bench):
    trajs = list(bench)
    n_mal = sum(1 for t in trajs if t.ground_truth == "malicious")
    # 15 categories x 5 attack strings = 75 attacks, 6 contexts each
    assert n_mal == 75 * 6
    assert len(trajs) - n_mal == 150  # 3 tasks x 50 benign
    assert len(trajs) == len(bench)


def test_construction_is_deterministic():
    a = BipiABench(n_per_attack=2, n_benign_per_task=5, seed=0)
    b = BipiABench(n_per_attack=2, n_benign_per_task=5, seed=0)
    assert [t.task for t in a] == [t.task for t in b]
    assert [t.ground_truth for t in a] == [t.ground_truth for t in b]


def test_attack_text_is_embedded(bench):
    attacks = json.load(open("data/bipia/text_attack_test.json"))
    flat = {f"{cat}-{i}": s for cat, lst in attacks.items()
            for i, s in enumerate(lst)}
    checked = 0
    for t in bench:
        m = bench.meta(t)
        if m["attack_name"] is None:
            continue
        ctx = str(t.steps[0].tool_response.output)
        assert flat[m["attack_name"]] in ctx, m["attack_name"]
        checked += 1
    assert checked == 75 * 6


def test_benign_contexts_are_unpoisoned(bench):
    attacks = json.load(open("data/bipia/text_attack_test.json"))
    strs = [s for lst in attacks.values() for s in lst]
    for t in bench:
        if t.ground_truth != "benign":
            continue
        ctx = str(t.steps[0].tool_response.output)
        assert not any(s in ctx for s in strs)


def test_positions_rotate(bench):
    from collections import Counter
    pos = Counter(bench.meta(t)["position"] for t in bench
                  if bench.meta(t)["position"])
    assert set(pos) == {"start", "middle", "end"}
    assert max(pos.values()) - min(pos.values()) <= 75 * 6 // 3


def test_trajectory_shape(bench):
    for t in bench:
        assert t.source_bench == "bipia"
        assert len(t.steps) == 2
        call, resp = t.steps[0].tool_call, t.steps[0].tool_response
        assert call is not None and resp is not None
        assert call.call_id == resp.call_id
        assert resp.source in ("email", "file", "web")
        assert t.steps[0].tool_call.name == f"read_{bench.meta(t)['task']}"
        # final step reserved for the victim answer
        assert t.steps[-1].text == ""


def test_every_attack_category_represented(bench):
    from collections import Counter
    cats = Counter(bench.meta(t)["attack_name"].rsplit("-", 1)[0]
                   for t in bench if bench.meta(t)["attack_name"])
    assert len(cats) == 15
    assert set(cats.values()) == {6 * 5}  # each category: 5 strings x 6 contexts
