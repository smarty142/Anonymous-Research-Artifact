"""Evaluation package."""
from evaluation.metrics import (
    SystemMetrics, evaluate, evaluate_many, coverage_table,
)

__all__ = ["SystemMetrics", "evaluate", "evaluate_many", "coverage_table"]