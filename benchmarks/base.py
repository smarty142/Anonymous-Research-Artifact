"""Benchmark loader interface and shared types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

from schemas import Trajectory


@dataclass(frozen=True)
class BenchInfo:
    name: str
    description: str
    n_trajectories: int
    ground_truth_field: str        # which field encodes the label
    attack_pattern_field: str      # which field encodes the pattern id (or "")


@runtime_checkable
class Benchmark(Protocol):
    """Anything that yields trajectories + ground-truth labels."""

    info: BenchInfo

    def __iter__(self) -> Iterator[Trajectory]:
        ...