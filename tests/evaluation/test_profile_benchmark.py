from config.profiles import ProfileName
from evaluation.generation_benchmark import AnswerGoldCase
from evaluation.profile_benchmark import ProfileBenchmarkVariant, compare_profiles
from models.response import OmniRAGResponse


def test_profile_comparison_pairs_same_labeled_case_without_claiming_winner():
    case = AnswerGoldCase("same-case", frozenset({"fact"}), frozenset(), {}, False)
    variants = [ProfileBenchmarkVariant(
        profile, lambda sample: OmniRAGResponse(answer="fact", profile_used=profile))
        for profile in ProfileName
    ]
    report = compare_profiles([case], variants)
    assert set(report) == {"lite", "core", "research"}
    assert all(values["answer_correctness"] == 1 for values in report.values())
