"""Model compression demo utilities."""

from .catalog import ALGORITHMS, GPU_INSTANCES, SCHEMES
from .evals import QualityEvalPlan, build_quality_eval_plan, format_quality_eval_plan
from .planner import (
    CompressionPlan,
    MemoryEstimate,
    estimate_compression_memory,
    estimate_serving_memory,
    recommend_instances,
    select_algorithm,
)

__all__ = [
    "ALGORITHMS",
    "GPU_INSTANCES",
    "SCHEMES",
    "QualityEvalPlan",
    "CompressionPlan",
    "MemoryEstimate",
    "build_quality_eval_plan",
    "estimate_compression_memory",
    "estimate_serving_memory",
    "format_quality_eval_plan",
    "recommend_instances",
    "select_algorithm",
]
