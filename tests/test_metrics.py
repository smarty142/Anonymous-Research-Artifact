"""Tests for evaluation metrics and sanity baselines."""
import pytest

from baselines import AlwaysAllow, AlwaysBlock, DeterministicVerifier
from benchmarks.osi_bench import OSIBench
from evaluation.metrics import evaluate, SystemMetrics


def test_always_allow_metrics():
    bench = OSIBench(attacks_per_pattern=2, benign_count=5, seed=0)
    m = evaluate(AlwaysAllow(), bench)
    assert m.tp == 0
    assert m.fn == m.n_malicious
    assert m.fp == 0
    assert m.tn == m.n_benign
    assert m.tpr == 0.0
    assert m.fpr == 0.0


def test_always_block_metrics():
    bench = OSIBench(attacks_per_pattern=2, benign_count=5, seed=0)
    m = evaluate(AlwaysBlock(), bench)
    assert m.tp == m.n_malicious
    assert m.fp == m.n_benign
    assert m.tpr == 1.0
    assert m.fpr == 1.0


def test_det_pipeline_runs_without_error():
    bench = OSIBench(attacks_per_pattern=2, benign_count=5, seed=0)
    m = evaluate(DeterministicVerifier(), bench)
    # We expect SOME attacks caught (this is the empirical contribution)
    assert 0 <= m.tpr <= 1
    assert 0 <= m.fpr <= 1
    # Determinism: mean latency should be small (< 500ms on tiny bench in CI)
    assert m.mean_latency_ms < 500


def test_coverage_metric():
    bench = OSIBench(attacks_per_pattern=2, benign_count=5, seed=0)
    m = evaluate(DeterministicVerifier(), bench)
    # Coverage is deduped by pattern: total = unique patterns evaluated
    assert len(m.coverage_hit) + len(m.coverage_miss) > 0
    # We expect at least *some* patterns hit
    # (this is a smoke check, not a strict threshold)


def test_as_row_dict():
    m = SystemMetrics(name="x")
    row = m.as_row()
    assert row["system"] == "x"
    assert "tpr" in row and "fpr" in row and "coverage" in row