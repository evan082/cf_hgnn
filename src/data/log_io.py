"""Step 1: event log loading and case-level utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


@dataclass
class EventLogBundle:
    """In-memory event log with activity vocabulary and case traces."""

    df: pd.DataFrame
    case_col: str
    activity_col: str
    timestamp_col: str
    activity2id: Dict[str, int]
    id2activity: Dict[int, str]
    cases: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def num_activities(self) -> int:
        return len(self.activity2id)

    @property
    def case_ids(self) -> List[str]:
        return list(self.cases.keys())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "case": ["case", "case:concept:name", "case_id", "caseid", "case id"],
        "activity": ["activity", "concept:name", "event", "activity_name"],
        "timestamp": ["timestamp", "time:timestamp", "time", "datetime", "complete_timestamp"],
        "resource": ["resource", "org:resource", "org_resource", "user"],
    }
    for canon, names in aliases.items():
        for name in names:
            if name.lower() in lower:
                mapping[lower[name.lower()]] = canon
                break
    out = df.rename(columns=mapping).copy()
    required = {"case", "activity", "timestamp"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    return out


def normalize_activity_label(name: str) -> str:
    """ProcessTransformer-style activity string: lower + spaces to hyphens."""
    return str(name).lower().replace(" ", "-")


def load_event_log(
    path: str | Path,
    case_col: Optional[str] = None,
    activity_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    normalize_activity_names: bool = False,
) -> EventLogBundle:
    """Load CSV or XES event log and build per-case activity sequences."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".xes", ".gz"} or path.name.endswith(".xes.gz"):
        import pm4py

        log = pm4py.read_xes(str(path))
        df = pm4py.convert_to_dataframe(log)
        df = _normalize_columns(df)
    else:
        df = pd.read_csv(path)
        if case_col or activity_col or timestamp_col:
            rename = {}
            if case_col:
                rename[case_col] = "case"
            if activity_col:
                rename[activity_col] = "activity"
            if timestamp_col:
                rename[timestamp_col] = "timestamp"
            df = df.rename(columns=rename)
        else:
            df = _normalize_columns(df)

    df["case"] = df["case"].astype(str)
    df["activity"] = df["activity"].astype(str)
    if normalize_activity_names:
        df["activity"] = df["activity"].map(normalize_activity_label)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        # fall back to original row order within case
        df["timestamp"] = df.groupby("case").cumcount()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", origin="2020-01-01")

    df = df.sort_values(["case", "timestamp"], kind="mergesort").reset_index(drop=True)

    activities = sorted(df["activity"].unique().tolist())
    activity2id = {a: i for i, a in enumerate(activities)}
    id2activity = {i: a for a, i in activity2id.items()}

    cases: Dict[str, List[str]] = {}
    for case_id, g in df.groupby("case", sort=False):
        cases[str(case_id)] = g["activity"].tolist()

    return EventLogBundle(
        df=df,
        case_col="case",
        activity_col="activity",
        timestamp_col="timestamp",
        activity2id=activity2id,
        id2activity=id2activity,
        cases=cases,
    )


def cases_to_pm4py_log(cases: Dict[str, Sequence[str]]):
    """Convert case->activity sequences to a pm4py event log."""
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from datetime import datetime, timedelta

    log = EventLog()
    base = datetime(2020, 1, 1)
    for case_id, acts in cases.items():
        trace = Trace()
        trace.attributes["concept:name"] = str(case_id)
        for i, act in enumerate(acts):
            ev = Event()
            ev["concept:name"] = str(act)
            ev["time:timestamp"] = base + timedelta(hours=i)
            trace.append(ev)
        log.append(trace)
    return log
