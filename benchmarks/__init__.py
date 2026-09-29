"""Benchmark loaders."""
from benchmarks.base import BenchInfo, Benchmark
from benchmarks.osi_bench import OSIBench, pattern_id_from_traj

__all__ = ["BenchInfo", "Benchmark", "OSIBench", "pattern_id_from_traj"]


def AgentDojoBench(*args, **kwargs):
    from benchmarks.agentdojo import AgentDojoBench as _Cls
    return _Cls(*args, **kwargs)


def InjecAgentBench(*args, **kwargs):
    from benchmarks.injecagent import InjecAgentBench as _Cls
    return _Cls(*args, **kwargs)