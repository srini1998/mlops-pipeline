import os
from pathlib import Path

import pandas as pd
import pytest

from mlops import registry
from mlops.data import DataStore, ValidationError, read_arff, validate, write_weekly_batches
from mlops.pipeline import Pipeline

SOURCE = Path("data/source/electricity.arff")


@pytest.fixture(scope="module")
def frame():
    return read_arff(SOURCE)


@pytest.fixture()
def weeks(tmp_path, frame):
    return write_weekly_batches(frame, tmp_path / "staging", 10, 4)


def test_source_is_the_published_elec2_file(frame):
    assert len(frame) == 45312
    assert set(frame["label"].unique()) == {0, 1}


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns=["transfer"]), "missing columns"),
    (lambda d: d.assign(nswdemand=d["nswdemand"].where(d.index != 5)), "null values"),
    (lambda d: d.assign(nswprice=d["nswprice"] * 400), "normalized"),
    (lambda d: d.assign(label=1), "implausible"),
    (lambda d: d.head(50), "too small"),
    (lambda d: d.iloc[::-1], "half-hour order"),
])
def test_validation_rejects_bad_batches(weeks, mutate, message):
    good = pd.read_csv(weeks[0])
    validate(good)
    with pytest.raises(ValidationError, match=message):
        validate(mutate(good))


def test_datastore_deduplicates_identical_bytes(tmp_path, weeks):
    store = DataStore(tmp_path / "store")
    v1, new1 = store.put(weeks[0])
    v2, new2 = store.put(weeks[0])
    assert new1 and not new2 and v1.sha256 == v2.sha256
    assert len(store.versions()) == 1


def test_gates_promotion_and_rollback(tmp_path, weeks):
    pipe = Pipeline(tmp_path / "state")
    first = pipe.process(weeks[0])
    assert first.status == "promoted" and first.champion_after == first.challenger_version

    broken = pipe.process(weeks[1], shuffle_labels=True)
    assert broken.status == "failed_floor" and broken.champion_after == first.champion_after
    assert broken.champion_live_accuracy is not None  # the champion was scored on the unseen week first

    assert pipe.process(weeks[1]).status == "duplicate"

    bad = weeks[2].with_name("bad.csv")
    pd.read_csv(weeks[2]).assign(vicdemand=None).to_csv(bad, index=False)
    rejected = pipe.process(bad)
    assert rejected.status == "rejected" and "null" in rejected.reason

    results = [pipe.process(w) for w in weeks[2:]]
    promoted = [r for r in results if r.status == "promoted"]
    client = registry.configure(tmp_path / "state")
    if promoted:
        before = registry.champion_version(client)
        current, previous = registry.rollback(client, "test")
        assert current == before and registry.champion_version(client) == previous
    else:
        with pytest.raises(RuntimeError):
            registry.rollback(client, "nothing newer than the first model")


def test_service_serves_the_champion(tmp_path, weeks):
    from fastapi.testclient import TestClient

    state = tmp_path / "state"
    Pipeline(state).process(weeks[0])
    os.environ["MLOPS_STATE"] = str(state)
    import importlib

    import mlops.serve as serve

    importlib.reload(serve)
    with TestClient(serve.app) as client:
        assert client.get("/health").json()["model_version"] == "1"
        reading = {"day": 3, "period": 0.5, "nswprice": 0.06, "nswdemand": 0.4,
                   "vicprice": 0.003, "vicdemand": 0.42, "transfer": 0.41}
        body = client.post("/predict", json=[reading]).json()
        assert body["model_version"] == "1" and body["predictions"][0]["direction"] in ("UP", "DOWN")
        assert client.post("/predict", json=[{**reading, "nswprice": 7}]).status_code == 422


def test_real_week_with_non_monotonic_date_field_is_accepted(frame):
    """Regression: the published `date` column goes backwards inside week 75; the batch is genuine."""
    from mlops.data import ROWS_PER_WEEK

    week = frame.iloc[75 * ROWS_PER_WEEK:76 * ROWS_PER_WEEK]
    assert not week["date"].is_monotonic_increasing
    validate(week.reset_index(drop=True))
