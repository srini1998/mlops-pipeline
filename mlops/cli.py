"""mlops: replay uploads, watch the landing zone, inspect the registry, roll back, serve, simulate."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from . import registry
from .data import FEATURES, ROWS_PER_WEEK, add_features, read_arff, write_weekly_batches
from .pipeline import GateConfig, Pipeline, train_model
from .trigger import LandingWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for noisy in ("mlflow", "alembic", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("mlops.cli")
SOURCE = Path("data/source/electricity.arff")


def simulate(state: Path, weeks: int, report: Path) -> dict:
    """Replay real weekly uploads through the watcher, with two injected faults, then compare
    the pipeline's live accuracy with a never-retrained model and a persistence baseline."""
    if state.exists():
        shutil.rmtree(state)
    df = read_arff(SOURCE)
    staging = state / "staging"
    batches = write_weekly_batches(df, staging, 0, weeks)
    landing = state / "landing"
    landing.mkdir(parents=True)
    pipe = Pipeline(state)
    watcher = LandingWatcher(landing, pipe)
    corrupt_at, broken_at = weeks // 3, (2 * weeks) // 3
    results = []
    for i, src in enumerate(batches):
        if i == corrupt_at:
            # Fault 1: an upload where a sensor feed failed - prices arrive un-normalized.
            bad = landing / f"{src.stem}_corrupt.csv"
            frame = pd.read_csv(src)
            frame["nswprice"] = frame["nswprice"] * 400
            frame.to_csv(bad, index=False)
            watcher.poll_once()
            results += watcher.poll_once()
        shutil.copy(src, landing / src.name)
        watcher.poll_once()  # first poll sees the file, second confirms its size is stable
        if i == broken_at:
            # Fault 2: a training bug (labels misaligned) produces a broken challenger.
            results.append(pipe.process(landing / src.name, shuffle_labels=True))
            results.append(pipe.process(landing / src.name))  # same bytes again: must be a duplicate
            watcher.seen[src.name] = "processed-manually"
            continue
        results += watcher.poll_once()

    # Baselines on exactly the weeks where the pipeline had a champion serving.
    window = GateConfig().window_weeks
    static = train_model(pd.concat([add_features(df.iloc[w * ROWS_PER_WEEK:(w + 1) * ROWS_PER_WEEK])
                                    for w in range(window)]), seed=7)
    rows = []
    for r in results:
        if r.champion_live_accuracy is None or r.status == "rejected":
            continue
        week = int(r.batch.split("_")[1].split(".")[0])
        if week < window:
            continue  # the static model trained on these weeks; scoring it there would be leakage
        chunk = add_features(df.iloc[week * ROWS_PER_WEEK:(week + 1) * ROWS_PER_WEEK])
        persistence = (chunk["label"].shift(1).fillna(chunk["label"].iloc[0]) == chunk["label"]).mean()
        rows.append({
            "week": week,
            "pipeline_champion": r.champion_live_accuracy,
            "static_model": round(accuracy_score(chunk["label"], static.predict(chunk[FEATURES])), 4),
            "persistence": round(float(persistence), 4),
        })
    late = [x for x in rows if x["week"] >= window * 2]
    avg = lambda key, xs: round(sum(x[key] for x in xs) / len(xs), 4) if xs else None
    summary = {
        "weeks_replayed": len(batches),
        "decisions": {s: sum(r.status == s for r in results) for s in
                      ("promoted", "kept_champion", "failed_floor", "rejected", "duplicate")},
        "faults": {
            "corrupt_upload": next((r.reason for r in results if r.batch.endswith("_corrupt.csv")), None),
            "broken_challenger": next((f"{r.batch}: {r.reason}" for r in results if r.extra.get("shuffle_labels")), None),
        },
        f"live_accuracy_weeks_{window}_plus": {k: avg(k, rows) for k in ("pipeline_champion", "static_model", "persistence")},
        f"live_accuracy_weeks_{window * 2}_plus": {k: avg(k, late) for k in ("pipeline_champion", "static_model", "persistence")},
        "weekly": rows,
        "batches": [r.__dict__ for r in results],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mlops", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--state", default="state", help="pipeline state directory (registry, data store, ledgers)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("replay", help="write real weekly batches into a landing directory")
    s.add_argument("--first", type=int, default=0)
    s.add_argument("--weeks", type=int, default=4)
    s.add_argument("--to", default="state/landing")
    s = sub.add_parser("watch", help="process new uploads in the landing directory")
    s.add_argument("--landing", default="state/landing")
    s.add_argument("--once", action="store_true")
    sub.add_parser("status", help="registry history and the current champion")
    s = sub.add_parser("rollback", help="restore the previous champion")
    s.add_argument("--reason", required=True)
    s = sub.add_parser("serve", help="serve the champion over HTTP")
    s.add_argument("--port", type=int, default=8780)
    s = sub.add_parser("simulate", help="end-to-end replay with injected faults and baselines")
    s.add_argument("--weeks", type=int, default=60)
    s.add_argument("--report", default="docs/simulation.json")
    args = p.parse_args(argv)
    state = Path(args.state)

    if args.cmd == "replay":
        paths = write_weekly_batches(read_arff(SOURCE), Path(args.to), args.first, args.weeks)
        print("\n".join(str(x) for x in paths))
    elif args.cmd == "watch":
        watcher = LandingWatcher(Path(args.landing), Pipeline(state))
        if args.once:
            watcher.poll_once()
            for r in watcher.poll_once():
                print(json.dumps(r.__dict__, default=str))
        else:
            watcher.run_forever()
    elif args.cmd == "status":
        client = registry.configure(state)
        print(json.dumps({"champion": registry.champion_version(client), "versions": registry.history(client)}, indent=2))
    elif args.cmd == "rollback":
        current, previous = registry.rollback(registry.configure(state), args.reason)
        print(f"champion v{current} -> v{previous}")
    elif args.cmd == "serve":
        import os

        import uvicorn

        os.environ["MLOPS_STATE"] = str(state)
        uvicorn.run("mlops.serve:app", host="127.0.0.1", port=args.port)
    elif args.cmd == "simulate":
        summary = simulate(state, args.weeks, Path(args.report))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("weekly", "batches")}, indent=2))


if __name__ == "__main__":
    main()
