"""Prediction service that always serves the registry's champion and hot-swaps on promotion or rollback."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import registry
from .data import FEATURES

STATE = Path(os.environ.get("MLOPS_STATE", "state"))
app = FastAPI(title="Elec2 price-direction service")
_lock = threading.Lock()
_current: dict = {"version": None, "model": None, "loaded_at": None}


class Reading(BaseModel):
    day: int = Field(ge=1, le=7)
    period: float = Field(ge=0, le=1)
    nswprice: float = Field(ge=0, le=1)
    nswdemand: float = Field(ge=0, le=1)
    vicprice: float = Field(ge=0, le=1)
    vicdemand: float = Field(ge=0, le=1)
    transfer: float = Field(ge=0, le=1)
    prev_direction: str | None = Field(None, pattern="^(UP|DOWN)$", description="direction of the previous half-hour")
    prev_nswprice: float | None = Field(None, ge=0, le=1)


def refresh() -> None:
    client = registry.configure(STATE)
    version = registry.champion_version(client)
    if version is not None and version != _current["version"]:
        model = registry.load(version)
        with _lock:
            _current.update(version=version, model=model, loaded_at=time.time())


def _watch_alias(interval: float = 5.0) -> None:
    while True:
        try:
            refresh()
        except Exception:  # keep serving the last good model if the registry is briefly unavailable
            pass
        time.sleep(interval)


@app.on_event("startup")
def _start() -> None:
    refresh()
    threading.Thread(target=_watch_alias, daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {"model_version": _current["version"], "loaded_at": _current["loaded_at"]}


@app.post("/predict")
def predict(readings: list[Reading]) -> dict:
    with _lock:
        model, version = _current["model"], _current["version"]
    if model is None:
        raise HTTPException(503, "no champion model has been promoted yet")
    rows = []
    for r in readings:
        d = r.model_dump()
        d["prev_label"] = None if d["prev_direction"] is None else float(d["prev_direction"] == "UP")
        d["nswprice_delta"] = None if d["prev_nswprice"] is None else d["nswprice"] - d["prev_nswprice"]
        rows.append(d)
    frame = pd.DataFrame(rows)[FEATURES].astype(float)
    proba = model.predict_proba(frame)[:, 1]
    return {"model_version": version,
            "predictions": [{"direction": "UP" if p >= 0.5 else "DOWN", "p_up": round(float(p), 4)} for p in proba]}
