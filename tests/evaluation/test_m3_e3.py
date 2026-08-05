from temporal_nlg.evaluation.m3_e3 import (
    ComprehensionResponse,
    ComprehensionTask,
    ExplanationItem,
    ComprehensionQuestion,
    aggregate_comprehension,
    score_responses_against_tasks,
    UtilityResponse,
    aggregate_utility,
    CognitiveLoadResponse,
    aggregate_cognitive_load,
)


def test_m3_e3a_scores_mcq_from_answer_key() -> None:
    task = ComprehensionTask(
        item=ExplanationItem(
            explanation_id="e1", domain="medical", bucket="point", query="q", explanation_text="x"
        ),
        questions=[
            ComprehensionQuestion(
                question_id="e1.q01",
                question_type="mcq",
                prompt="Which year?",
                options=["2020", "2021"],
                correct_answer="2021",
            )
        ],
    )

    responses = [
        ComprehensionResponse(
            participant_id="p1",
            explanation_id="e1",
            question_id="e1.q01",
            question_type="mcq",
            answer="2021",
            score=None,
            response_time_sec=12.0,
        )
    ]

    scored = score_responses_against_tasks(responses, [task])
    assert scored[0].score == 1.0

    summary = aggregate_comprehension(scored)
    assert summary["overall_accuracy"] == 1.0
    assert summary["accuracy_by_bucket"]["point"] == 1.0
    assert summary["accuracy_by_domain"]["medical"] == 1.0


def test_m3_e3b_utility_improvement() -> None:
    responses = [
        UtilityResponse(
            participant_id="u1",
            task_id="t1",
            domain="finance",
            condition="with_explanation",
            success=True,
            confidence_1_5=4,
            time_sec=10,
            expert_agreement=0.8,
        ),
        UtilityResponse(
            participant_id="u2",
            task_id="t2",
            domain="finance",
            condition="without_explanation",
            success=False,
            confidence_1_5=3,
            time_sec=20,
            expert_agreement=0.6,
        ),
    ]
    summary = aggregate_utility(responses)
    assert summary["success_rate"]["with_explanation"] == 1.0
    assert summary["success_rate"]["without_explanation"] == 0.0
    assert summary["confidence_delta_vs_without"] == 1.0
    assert summary["time_reduction_vs_without"] == 0.5
    assert summary["expert_agreement_mean"] == 0.7


def test_m3_e3c_tlx_aggregation() -> None:
    responses = [
        CognitiveLoadResponse(
            participant_id="c1",
            condition="dense_text",
            tlx_mental=40,
            tlx_physical=10,
            tlx_temporal=30,
            tlx_performance=50,
            tlx_effort=40,
            tlx_frustration=20,
        ),
        CognitiveLoadResponse(
            participant_id="c2",
            condition="dense_text",
            tlx_mental=60,
            tlx_physical=10,
            tlx_temporal=30,
            tlx_performance=50,
            tlx_effort=40,
            tlx_frustration=20,
        ),
    ]
    summary = aggregate_cognitive_load(responses)
    assert "dense_text" in summary["tlx_mean_by_condition"]
    assert summary["tlx_mean_by_condition"]["dense_text"] is not None
