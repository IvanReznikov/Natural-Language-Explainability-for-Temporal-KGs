from temporal_nlg.tms.trace import TraceRecorder


def test_trace_records_rule_and_serializes():
    recorder = TraceRecorder(rand_fn=lambda: 0.0)
    trace = recorder.start_query("q1", {"user": "u1"})

    recorder.record_rule_firing(
        trace,
        rule_id="r1",
        rule_name="rule_one",
        inputs=[{"fact_id": "f1", "value": 42}],
        conclusion={"fact_id": "f2", "value": 43},
        confidence=0.9,
        latency_ms=2.5,
        meta={"stage": "initial"},
    )
    recorder.complete_query(trace)

    as_dict = trace.to_dict()
    assert as_dict["query_id"] == "q1"
    assert as_dict["duration_ms"] is not None
    assert len(as_dict["rule_traces"]) == 1
    assert as_dict["rule_traces"][0]["rule_id"] == "r1"

    json_str = recorder.to_json(trace)
    assert "r1" in json_str
    assert "rule_one" in json_str


def test_sampling_drops_trace():
    recorder = TraceRecorder(sampling_rate=0.0, rand_fn=lambda: 1.0)
    trace = recorder.start_query("q_skip")

    recorder.record_rule_firing(
        trace,
        rule_id="r2",
        rule_name="rule_two",
        inputs=[{"fact_id": "f10"}],
        conclusion={"fact_id": "f20"},
    )
    recorder.complete_query(trace)

    assert trace.dropped is True
    assert trace.rule_traces == []
    assert trace.instrumentation_overhead_ms == 0.0


def test_overhead_tracking_sets_flag():
    class IncrementingPerf:
        def __init__(self, step: float = 0.003):
            self.value = 0.0
            self.step = step

        def __call__(self) -> float:
            self.value += self.step
            return self.value

    perf = IncrementingPerf()
    recorder = TraceRecorder(max_overhead_ms=2.0, perf_fn=perf, rand_fn=lambda: 0.0)
    trace = recorder.start_query("q_overhead")

    recorder.record_rule_firing(
        trace,
        rule_id="r3",
        rule_name="rule_three",
        inputs=[{"fact_id": "f100"}],
        conclusion={"fact_id": "f200"},
    )
    recorder.complete_query(trace)

    assert trace.instrumentation_overhead_ms > 2.0
    assert trace.over_budget is True


def test_session_context_completes_trace():
    recorder = TraceRecorder(rand_fn=lambda: 0.0)
    with recorder.session("q_ctx") as trace:
        recorder.record_rule_firing(
            trace,
            rule_id="r_ctx",
            rule_name="rule_ctx",
            inputs=[{"fact_id": "f1"}],
            conclusion={"fact_id": "g1"},
        )
    assert trace.completed_at is not None
