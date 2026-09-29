"""Tests for the OSI-Bench generator."""
from osi_bench.generator import OSIBenchGenerator
from benchmarks.osi_bench import OSIBench, pattern_id_from_traj
from osi_bench.mutations import implemented_patterns


def test_generator_produces_balanced_bench():
    bench = OSIBench(attacks_per_pattern=3, benign_count=15, seed=42)
    n = len(bench)
    n_mal = sum(1 for t in bench if t.ground_truth == "malicious")
    n_ben = sum(1 for t in bench if t.ground_truth == "benign")
    assert n == n_mal + n_ben
    assert n_mal == len(implemented_patterns()) * 3
    assert n_ben == 15


def test_held_out_split_is_deterministic():
    gen = OSIBenchGenerator(attacks_per_pattern=2, benign_count=10, seed=0)
    trajs = gen.generate()
    a_train, a_test = gen.held_out_split(trajs)
    b_train, b_test = gen.held_out_split(trajs)
    assert [t.traj_id for t in a_train] == [t.traj_id for t in b_train]
    assert [t.traj_id for t in a_test] == [t.traj_id for t in b_test]
    assert len(a_train) + len(a_test) == len(trajs)


def test_pattern_id_extraction():
    bench = OSIBench(attacks_per_pattern=2, benign_count=5, seed=0)
    for t in bench:
        if t.ground_truth == "malicious":
            pid = pattern_id_from_traj(t)
            assert pid is not None
            assert pid.startswith("T.")


def test_serialization_roundtrip(tmp_path):
    gen = OSIBenchGenerator(attacks_per_pattern=2, benign_count=5, seed=0)
    trajs = gen.generate()
    p = tmp_path / "bench.jsonl"
    gen.to_jsonl(trajs, p)
    loaded = OSIBenchGenerator.from_jsonl(p)
    assert len(loaded) == len(trajs)
    for orig, new in zip(trajs, loaded):
        assert orig.traj_id == new.traj_id
        assert orig.ground_truth == new.ground_truth
        assert orig.task == new.task
        assert orig.declared_capabilities == new.declared_capabilities
        assert len(orig.steps) == len(new.steps)