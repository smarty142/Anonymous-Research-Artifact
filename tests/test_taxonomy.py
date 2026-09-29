"""Tests for the threat taxonomy loader."""
import pytest

from taxonomy.loader import (
    Taxonomy, ThreatClass, PipelineStage, Severity, DEFAULT_PATH,
)


def test_default_path_exists():
    assert DEFAULT_PATH.exists()


def test_load_default_parses_all_patterns():
    tax = Taxonomy.load_default()
    # Locked count from paper draft §1
    assert len(tax) >= 40
    # All classes present
    classes = {p.cls for p in tax}
    assert classes == {ThreatClass.TOOL_CALL, ThreatClass.TEXT_OUTPUT,
                       ThreatClass.DATA_FLOW}


def test_unique_pattern_ids():
    tax = Taxonomy.load_default()
    ids = tax.pattern_ids()
    assert len(ids) == len(set(ids)), "duplicate pattern ids"


def test_by_id_lookup():
    tax = Taxonomy.load_default()
    p = tax.by_id("T.A.4")
    assert p.name == "path_traversal"
    assert p.stage is PipelineStage.THREAT_MATCH


def test_by_stage_lookup():
    tax = Taxonomy.load_default()
    tokenize = tax.by_stage("tokenize")
    assert all(p.stage is PipelineStage.TOKENIZE for p in tokenize)
    assert len(tokenize) >= 3  # T.B.1, T.B.4, T.B.12, T.A.14


def test_by_class_lookup():
    tax = Taxonomy.load_default()
    a = tax.by_class("tool_call")
    assert len(a) >= 15
    assert all(p.cls is ThreatClass.TOOL_CALL for p in a)


def test_examples_have_kind():
    tax = Taxonomy.load_default()
    for p in tax:
        for ex in p.examples:
            assert ex.kind in ("positive", "negative")
            assert ex.input


def test_severity_values():
    tax = Taxonomy.load_default()
    valid = {s.value for s in Severity}
    for p in tax:
        assert p.severity.value in valid