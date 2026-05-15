from temporal_nlg.tms.trigger_engine import TriggerEngine, TriggerRule, TriggerContext
from temporal_nlg.tms.query_store import QueryStore


def test_trigger_engine_fires_and_stores(tmp_path):
    store = QueryStore(path=tmp_path / "q.jsonl")

    def pred(ctx: TriggerContext) -> bool:
        return ctx.facts.get("temperature", 0) > 101

    def factory(ctx: TriggerContext):
        return {
            "query_id": f"q_{ctx.context_id}",
            "text": "What antibiotics treat this infection?",
            "intent": "medical",
            "dependencies": ["temperature_fact"],
        }

    rule = TriggerRule(rule_id="r_high_temp", description="temperature rule", predicate=pred, query_factory=factory)

    engine = TriggerEngine(store)
    ctx = TriggerContext(context_id="c1", facts={"temperature": 102}, meta={})

    triggered = engine.evaluate(ctx, [rule])
    assert triggered == ["q_c1"]
    assert store.get("q_c1").intent == "medical"

    # negative case
    ctx2 = TriggerContext(context_id="c2", facts={"temperature": 99}, meta={})
    triggered2 = engine.evaluate(ctx2, [rule])
    assert triggered2 == []
