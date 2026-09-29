"""Threat taxonomy loader and query API.

Loads ``taxonomy/threat_taxonomy.yaml`` and exposes a typed view:

    >>> from taxonomy.loader import Taxonomy
    >>> tax = Taxonomy.load_default()
    >>> tax.by_id("T.A.1").name
    'unauthorized_tool_call'
    >>> tax.by_stage("tokenize")
    [Pattern(...), ...]
    >>> for p in tax.by_class("data_flow"): ...
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml


DEFAULT_PATH = Path(__file__).parent / "threat_taxonomy.yaml"


class ThreatClass(str, enum.Enum):
    TOOL_CALL = "tool_call"
    TEXT_OUTPUT = "text_output"
    DATA_FLOW = "data_flow"


class PipelineStage(str, enum.Enum):
    TOKENIZE = "tokenize"
    PARSE = "parse"
    THREAT_MATCH = "threat_match"
    DATAFLOW = "dataflow"


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Example:
    kind: str            # 'positive' | 'negative'
    input: str
    note: str = ""


@dataclass(frozen=True)
class Pattern:
    id: str
    name: str
    cls: ThreatClass
    severity: Severity
    stage: PipelineStage
    detector: str
    description: str
    examples: tuple[Example, ...] = ()
    refs: tuple[str, ...] = ()

    @property
    def positive_examples(self) -> list[Example]:
        return [e for e in self.examples if e.kind == "positive"]

    @property
    def negative_examples(self) -> list[Example]:
        return [e for e in self.examples if e.kind == "negative"]


@dataclass(frozen=True)
class Taxonomy:
    version: str
    description: str
    patterns: tuple[Pattern, ...]

    def __iter__(self) -> Iterator[Pattern]:
        return iter(self.patterns)

    def __len__(self) -> int:
        return len(self.patterns)

    def by_id(self, pid: str) -> Pattern:
        for p in self.patterns:
            if p.id == pid:
                return p
        raise KeyError(f"unknown pattern id: {pid}")

    def by_name(self, name: str) -> Pattern:
        for p in self.patterns:
            if p.name == name:
                return p
        raise KeyError(f"unknown pattern name: {name}")

    def by_class(self, cls: str | ThreatClass) -> list[Pattern]:
        if isinstance(cls, str):
            cls = ThreatClass(cls)
        return [p for p in self.patterns if p.cls is cls]

    def by_stage(self, stage: str | PipelineStage) -> list[Pattern]:
        if isinstance(stage, str):
            stage = PipelineStage(stage)
        return [p for p in self.patterns if p.stage is stage]

    def by_severity(self, sev: str | Severity) -> list[Pattern]:
        if isinstance(sev, str):
            sev = Severity(sev)
        return [p for p in self.patterns if p.severity is sev]

    def by_detector(self, detector: str) -> list[Pattern]:
        return [p for p in self.patterns if p.detector == detector]

    def pattern_ids(self) -> list[str]:
        return [p.id for p in self.patterns]

    # ---------- factories ----------

    @staticmethod
    def load_default() -> "Taxonomy":
        return Taxonomy.from_yaml(DEFAULT_PATH)

    @staticmethod
    def from_yaml(path: str | Path) -> "Taxonomy":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return Taxonomy.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "Taxonomy":
        patterns: list[Pattern] = []
        for entry in raw.get("patterns", []):
            examples = tuple(
                Example(kind=e["kind"], input=str(e["input"]), note=e.get("note", ""))
                for e in entry.get("examples", [])
            )
            patterns.append(
                Pattern(
                    id=entry["id"],
                    name=entry["name"],
                    cls=ThreatClass(entry["class"]),
                    severity=Severity(entry["severity"]),
                    stage=PipelineStage(entry["stage"]),
                    detector=entry["detector"],
                    description=entry["description"].strip(),
                    examples=examples,
                    refs=tuple(entry.get("refs", [])),
                )
            )
        # Validate uniqueness of ids
        ids = [p.id for p in patterns]
        if len(set(ids)) != len(ids):
            dupes = {x for x in ids if ids.count(x) > 1}
            raise ValueError(f"duplicate pattern ids: {dupes}")
        return Taxonomy(
            version=raw.get("version", "0.0.0"),
            description=raw.get("description", "").strip(),
            patterns=tuple(patterns),
        )