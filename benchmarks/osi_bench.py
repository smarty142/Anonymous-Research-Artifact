"""OSI-Bench loader (in-process generator)."""
from __future__ import annotations

from typing import Iterator

from benchmarks.base import BenchInfo
from osi_bench.generator import OSIBenchGenerator
from schemas import Trajectory


class OSIBench:
    name = "osi-bench"
    description = "OSI-Bench: 31 attack patterns × benign, generated from threat_taxonomy.yaml"

    def __init__(self, attacks_per_pattern: int = 5, benign_count: int = 50,
                 seed: int = 0) -> None:
        gen = OSIBenchGenerator(
            attacks_per_pattern=attacks_per_pattern,
            benign_count=benign_count,
            seed=seed,
        )
        self._trajs = gen.generate()
        self.info = BenchInfo(
            name=self.name, description=self.description,
            n_trajectories=len(self._trajs),
            ground_truth_field="ground_truth",
            attack_pattern_field="traj_id_suffix",
        )

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajs)

    def __len__(self) -> int:
        return len(self._trajs)


def pattern_id_from_traj(t: Trajectory) -> str | None:
    """Recover the taxonomy pattern id from a mutated OSI-Bench trajectory.

    Handles three conventions:
      - bare:                 bt_email__T.A.4
      - adaptive prefix:      adaptive_T.A.4__0
      - any embedded T.X.Y anywhere in the id
    """
    if t.ground_truth != "malicious":
        return None
    import re
    m = re.search(r"T\.[A-Z]\.\d+", t.traj_id)
    return m.group(0) if m else None