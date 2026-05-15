from temporal_nlg.tms.query_store import QueryStore
from temporal_nlg.tms.result_store import ResultStore


def test_query_store_size_limit(tmp_path):
    path = tmp_path / "queries.jsonl"
    store = QueryStore(path=path, size_limit_bytes=200)

    store.upsert("q1", "short text", "intent_a")
    assert store.get("q1").intent == "intent_a"

    long_text = "x" * 1000
    try:
        store.upsert("q2", long_text, "intent_b")
        assert False, "expected size limit error"
    except ValueError:
        pass


def test_result_store_invalidation(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResultStore(path=path)

    store.upsert(
        result_id="r1",
        query_id="q1",
        results=[{"value": 1}],
        dependent_facts=["f1", "f2"],
        invalidation_rules=["rule_x"],
    )

    changed = store.mark_stale_by_facts({"f2"})
    assert changed and changed[0].status == "stale"

    store.upsert(
        result_id="r2",
        query_id="q2",
        results=[{"value": 2}],
        invalidation_rules=["rule_y"],
    )
    changed_rules = store.mark_stale_by_rules({"rule_y"})
    assert changed_rules and changed_rules[0].status == "stale"
