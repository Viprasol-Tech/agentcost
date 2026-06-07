from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentcost.recorder import CostRecord, CostStore


def make_record(**kw) -> CostRecord:
    base = {
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.001,
        "tags": {"feature": "search"},
        "timestamp": 1000.0,
    }
    base.update(kw)
    return CostRecord(**base)


def test_record_total_tokens():
    rec = make_record(input_tokens=100, output_tokens=50)
    assert rec.total_tokens == 150


def test_record_tag_accessor():
    rec = make_record(tags={"feature": "x"})
    assert rec.tag("feature") == "x"
    assert rec.tag("missing") == ""
    assert rec.tag("missing", "default") == "default"


def test_record_is_frozen():
    rec = make_record()
    with pytest.raises(ValidationError):
        rec.cost = 9.0  # type: ignore[misc]


def test_record_negative_validation():
    with pytest.raises(ValidationError):
        make_record(input_tokens=-1)
    with pytest.raises(ValidationError):
        make_record(cost=-1.0)


def test_store_record_returns_record():
    store = CostStore()
    rec = store.record(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.1)
    assert isinstance(rec, CostRecord)
    assert len(store) == 1
    assert store.records[0] is rec


def test_store_record_with_tags_and_ts():
    store = CostStore()
    store.record(
        model="gpt-4o", input_tokens=1, output_tokens=1, cost=0.0, tags={"user": "a"}, timestamp=5.0
    )
    assert store.records[0].tags == {"user": "a"}
    assert store.records[0].timestamp == 5.0


def test_store_record_default_timestamp_is_set():
    store = CostStore()
    store.record(model="gpt-4o", input_tokens=1, output_tokens=1, cost=0.0)
    assert store.records[0].timestamp > 0


def test_store_total_cost():
    store = CostStore()
    store.record(model="m", input_tokens=1, output_tokens=1, cost=0.5)
    store.record(model="m", input_tokens=1, output_tokens=1, cost=0.25)
    assert store.total_cost() == pytest.approx(0.75)


def test_store_add_existing():
    store = CostStore()
    store.add(make_record())
    assert len(store) == 1


def test_store_init_with_records():
    store = CostStore([make_record(), make_record()])
    assert len(store) == 2


def test_store_records_is_copy():
    store = CostStore([make_record()])
    records = store.records
    records.clear()
    assert len(store) == 1


def test_store_clear():
    store = CostStore([make_record()])
    store.clear()
    assert len(store) == 0


def test_store_iter():
    store = CostStore([make_record(), make_record()])
    assert len(list(store)) == 2


def test_json_round_trip():
    store = CostStore()
    store.record(
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        cost=0.0075,
        tags={"feature": "search", "user": "alice"},
        timestamp=1234.5,
    )
    store.record(model="gpt-4o-mini", input_tokens=10, output_tokens=5, cost=0.0001)
    data = store.to_json()
    restored = CostStore.from_json(data)
    assert len(restored) == 2
    assert restored.records[0].model == "gpt-4o"
    assert restored.records[0].tags == {"feature": "search", "user": "alice"}
    assert restored.records[0].cost == pytest.approx(0.0075)
    assert restored.total_cost() == pytest.approx(store.total_cost())


def test_to_json_is_valid_json():
    store = CostStore([make_record()])
    parsed = json.loads(store.to_json())
    assert isinstance(parsed, list)
    assert parsed[0]["model"] == "gpt-4o"


def test_from_json_rejects_non_array():
    with pytest.raises(ValueError):
        CostStore.from_json('{"not": "a list"}')


def test_save_and_load(tmp_path):
    store = CostStore()
    store.record(
        model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.0075, tags={"feature": "f"}
    )
    path = tmp_path / "store.json"
    store.save(path)
    assert path.exists()
    loaded = CostStore.load(path)
    assert len(loaded) == 1
    assert loaded.records[0].tags == {"feature": "f"}


def test_empty_store_round_trip():
    store = CostStore()
    restored = CostStore.from_json(store.to_json())
    assert len(restored) == 0
