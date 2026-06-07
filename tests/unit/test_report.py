from __future__ import annotations

import pytest

from agentcost.recorder import CostStore
from agentcost.report import (
    UNTAGGED,
    build_report,
    by_agent_run,
    by_feature,
    by_model,
    by_user,
)


def sample_store() -> CostStore:
    store = CostStore()
    store.record(
        model="gpt-4o",
        input_tokens=100,
        output_tokens=100,
        cost=1.0,
        tags={"feature": "search", "user": "alice", "agent_run": "r1"},
    )
    store.record(
        model="gpt-4o",
        input_tokens=200,
        output_tokens=200,
        cost=2.0,
        tags={"feature": "search", "user": "bob", "agent_run": "r2"},
    )
    store.record(
        model="gpt-4o-mini",
        input_tokens=50,
        output_tokens=50,
        cost=0.5,
        tags={"feature": "summarize", "user": "alice", "agent_run": "r3"},
    )
    store.record(
        model="claude-3-5-sonnet",
        input_tokens=300,
        output_tokens=300,
        cost=3.0,
        tags={"feature": "summarize", "user": "bob", "agent_run": "r3"},
    )
    return store


def test_by_feature_totals():
    report = build_report(sample_store().records, by="feature")
    rows = {r.key: r for r in report.rows}
    assert rows["search"].cost == pytest.approx(3.0)
    assert rows["summarize"].cost == pytest.approx(3.5)
    assert rows["search"].calls == 2
    assert rows["summarize"].calls == 2


def test_by_feature_token_totals():
    report = build_report(sample_store().records, by="feature")
    rows = {r.key: r for r in report.rows}
    assert rows["search"].input_tokens == 300
    assert rows["search"].output_tokens == 300
    assert rows["search"].total_tokens == 600


def test_rows_sorted_by_cost_desc():
    report = build_report(sample_store().records, by="feature")
    costs = [r.cost for r in report.rows]
    assert costs == sorted(costs, reverse=True)
    assert report.rows[0].key == "summarize"  # 3.5 > 3.0


def test_by_user_totals():
    report = build_report(sample_store().records, by="user")
    rows = {r.key: r for r in report.rows}
    assert rows["alice"].cost == pytest.approx(1.5)
    assert rows["bob"].cost == pytest.approx(5.0)


def test_by_agent_run_totals():
    report = build_report(sample_store().records, by="agent_run")
    rows = {r.key: r for r in report.rows}
    assert rows["r3"].cost == pytest.approx(3.5)  # two calls share r3
    assert rows["r3"].calls == 2
    assert rows["r1"].cost == pytest.approx(1.0)


def test_by_model_totals():
    report = build_report(sample_store().records, by="model")
    rows = {r.key: r for r in report.rows}
    assert rows["gpt-4o"].cost == pytest.approx(3.0)
    assert rows["gpt-4o"].calls == 2
    assert rows["claude-3-5-sonnet"].cost == pytest.approx(3.0)


def test_report_total_cost_and_calls():
    report = build_report(sample_store().records, by="feature")
    assert report.total_cost == pytest.approx(6.5)
    assert report.total_calls == 4


def test_top_n():
    report = build_report(sample_store().records, by="user")
    top1 = report.top(1)
    assert len(top1) == 1
    assert top1[0].key == "bob"


def test_top_n_more_than_rows():
    report = build_report(sample_store().records, by="feature")
    assert len(report.top(100)) == 2


def test_top_n_zero():
    report = build_report(sample_store().records, by="feature")
    assert report.top(0) == []


def test_top_n_negative_raises():
    report = build_report(sample_store().records, by="feature")
    with pytest.raises(ValueError):
        report.top(-1)


def test_untagged_bucket():
    store = CostStore()
    store.record(model="gpt-4o", input_tokens=1, output_tokens=1, cost=0.1)  # no tags
    report = build_report(store.records, by="feature")
    assert report.rows[0].key == UNTAGGED


def test_custom_tag_dimension():
    store = CostStore()
    store.record(model="m", input_tokens=1, output_tokens=1, cost=1.0, tags={"region": "eu"})
    store.record(model="m", input_tokens=1, output_tokens=1, cost=2.0, tags={"region": "us"})
    report = build_report(store.records, by="region")
    rows = {r.key: r for r in report.rows}
    assert rows["us"].cost == pytest.approx(2.0)


def test_empty_records():
    report = build_report([], by="feature")
    assert report.rows == []
    assert report.total_cost == 0.0


def test_convenience_aliases():
    records = sample_store().records
    assert by_feature(records).dimension == "feature"
    assert by_user(records).dimension == "user"
    assert by_agent_run(records).dimension == "agent_run"
    assert by_model(records).dimension == "model"


def test_to_table_renders():
    report = build_report(sample_store().records, by="feature")
    table = report.to_table()
    assert table.row_count >= 2
    # has the expected columns
    assert len(table.columns) == 5


def test_dimension_recorded():
    report = build_report(sample_store().records, by="user")
    assert report.dimension == "user"
