"""Paired profile comparison runner over the same labeled question set."""

from dataclasses import dataclass
from typing import Callable, Sequence

from config.profiles import ProfileName
from eval.generation_benchmark import AnswerGoldCase, aggregate_answer_metrics, evaluate_response
from models.response import OmniRAGResponse


@dataclass(frozen=True, slots=True)
class ProfileBenchmarkVariant:
    profile: ProfileName
    execute: Callable[[AnswerGoldCase], OmniRAGResponse]


def compare_profiles(cases: Sequence[AnswerGoldCase],
                     variants: Sequence[ProfileBenchmarkVariant]) -> dict[str, dict[str, float]]:
    """Run identical labels for every profile; report metrics without ranking a winner."""
    if len({variant.profile for variant in variants}) != len(variants):
        raise ValueError("Each profile may appear only once in a comparison")
    reports = {}
    for variant in variants:
        rows = [evaluate_response(case, variant.execute(case)) for case in cases]
        reports[variant.profile.value] = aggregate_answer_metrics(rows)
    return reports
