"""Step 7: prefix alignment against reference Petri net + offline cache."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ..data.log_io import cases_to_pm4py_log


@dataclass
class AlignmentDiagnosis:
    case_id: str
    prefix_len: int
    fitness: float
    total_cost: float
    log_moves: Dict[str, int] = field(default_factory=dict)
    model_moves: Dict[str, int] = field(default_factory=dict)
    sync_moves: Dict[str, int] = field(default_factory=dict)
    per_activity_cost: Dict[str, float] = field(default_factory=dict)
    deviation_transitions: List[Tuple[str, str]] = field(default_factory=list)
    recent_deviation: int = 0  # 1 if last move is a deviation
    # fitness ignoring unfinished-case model moves (sync / (sync+log))
    prefix_fitness: float = 1.0

    def key(self) -> Tuple[str, int]:
        return (self.case_id, self.prefix_len)


def _trace_from_prefix(acts: Sequence[str]):
    from pm4py.objects.log.obj import Trace, Event
    from datetime import datetime, timedelta

    tr = Trace()
    base = datetime(2020, 1, 1)
    for i, a in enumerate(acts):
        ev = Event()
        ev["concept:name"] = str(a)
        ev["time:timestamp"] = base + timedelta(hours=i)
        tr.append(ev)
    return tr


def align_prefix(
    prefix_acts: Sequence[str],
    net,
    im,
    fm,
    timeout_sec: float | None = None,
) -> AlignmentDiagnosis:
    """
    Align a single prefix to (net, im, fm) and extract diagnosis features.
    Long prefixes fall back to token-based replay (much faster on BPIC-scale logs).
    """
    empty = AlignmentDiagnosis(
        case_id="",
        prefix_len=len(prefix_acts),
        fitness=1.0,
        total_cost=0.0,
        prefix_fitness=1.0,
    )
    if not prefix_acts:
        return empty

    # Long prefixes are expensive under alignment; use token replay.
    if len(prefix_acts) > 20:
        return _token_replay_diagnosis(prefix_acts, net, im, fm, empty)

    return _align_prefix_core(prefix_acts, net, im, fm, empty)


def _token_replay_diagnosis(prefix_acts, net, im, fm, empty: AlignmentDiagnosis) -> AlignmentDiagnosis:
    """Fast approximate diagnosis via token-based replay."""
    try:
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

        trace = _trace_from_prefix(prefix_acts)
        log = cases_to_pm4py_log({"tmp": list(prefix_acts)})
        replayed = token_replay.apply(log, net, im, fm)
        r0 = replayed[0] if replayed else {}
        # fitness-like: consumed/produced tokens
        if isinstance(r0, dict):
            fit = float(r0.get("trace_fitness", r0.get("fitness", 1.0)))
            missing = r0.get("missing_tokens", 0) or 0
            remaining = r0.get("remaining_tokens", 0) or 0
            consumed = r0.get("consumed_tokens", 0) or 0
            produced = r0.get("produced_tokens", 0) or 0
        else:
            fit = float(getattr(r0, "trace_fitness", 1.0))
            missing = getattr(r0, "missing_tokens", 0) or 0
            remaining = getattr(r0, "remaining_tokens", 0) or 0
            consumed = getattr(r0, "consumed_tokens", 0) or 0
            produced = getattr(r0, "produced_tokens", 0) or 0
        # Approximate log/model move signal from missing/remaining
        log_moves = {"__missing__": int(missing)} if missing else {}
        model_moves = {"__remaining__": int(remaining)} if remaining else {}
        n_sync = max(float(consumed), 1.0)
        n_log = float(missing)
        pref = n_sync / max(n_sync + n_log, 1.0)
        return AlignmentDiagnosis(
            case_id="",
            prefix_len=len(prefix_acts),
            fitness=float(fit),
            total_cost=float(missing + remaining),
            log_moves=log_moves,
            model_moves=model_moves,
            sync_moves={"__sync__": int(consumed)},
            per_activity_cost={},
            deviation_transitions=[],
            recent_deviation=1 if missing or remaining else 0,
            prefix_fitness=float(pref),
        )
    except Exception:
        return empty


def _align_prefix_core(prefix_acts, net, im, fm, empty: AlignmentDiagnosis) -> AlignmentDiagnosis:
    import pm4py

    try:
        trace = _trace_from_prefix(prefix_acts)
        from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

        aligned = alignments.apply(trace, net, im, fm)
    except Exception:
        try:
            log = cases_to_pm4py_log({"tmp": prefix_acts})
            diagnostics = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
            aligned = diagnostics[0]
        except Exception:
            return empty

    # Parse alignment moves. pm4py returns dict with 'alignment', 'cost', 'fitness'
    if isinstance(aligned, dict):
        alignment = aligned.get("alignment", [])
        cost = float(aligned.get("cost", 0.0))
        fitness = float(aligned.get("fitness", aligned.get("fitness_trace", 1.0)))
    else:
        alignment = getattr(aligned, "alignment", []) or []
        cost = float(getattr(aligned, "cost", 0.0))
        fitness = float(getattr(aligned, "fitness", 1.0))

    log_moves: Dict[str, int] = {}
    model_moves: Dict[str, int] = {}
    sync_moves: Dict[str, int] = {}
    per_cost: Dict[str, float] = {}
    deviations: List[Tuple[str, str]] = []
    recent_dev = 0

    prev_sync_or_log = None
    for move in alignment:
        # move typically ((log_label, model_label), (log_trans, model_trans)) or (('A','>>'), ...)
        if isinstance(move, tuple) and len(move) >= 1:
            pair = move[0] if isinstance(move[0], tuple) else move
        else:
            continue
        if not isinstance(pair, tuple) or len(pair) < 2:
            continue
        log_lab, model_lab = pair[0], pair[1]
        skip = ">>"
        is_log = log_lab != skip and (model_lab == skip or model_lab is None)
        is_model = model_lab != skip and model_lab is not None and (log_lab == skip or log_lab is None)
        is_sync = (
            log_lab != skip
            and model_lab != skip
            and log_lab is not None
            and model_lab is not None
            and log_lab == model_lab
        )

        if is_sync:
            sync_moves[str(log_lab)] = sync_moves.get(str(log_lab), 0) + 1
            per_cost[str(log_lab)] = per_cost.get(str(log_lab), 0.0)
            if prev_sync_or_log is not None:
                pass
            prev_sync_or_log = str(log_lab)
            recent_dev = 0
        elif is_log:
            log_moves[str(log_lab)] = log_moves.get(str(log_lab), 0) + 1
            per_cost[str(log_lab)] = per_cost.get(str(log_lab), 0.0) + 1.0
            if prev_sync_or_log is not None:
                deviations.append((prev_sync_or_log, str(log_lab)))
            recent_dev = 1
        elif is_model:
            model_moves[str(model_lab)] = model_moves.get(str(model_lab), 0) + 1
            per_cost[str(model_lab)] = per_cost.get(str(model_lab), 0.0) + 1.0
            recent_dev = 1

    n_sync = float(sum(sync_moves.values()))
    n_log = float(sum(log_moves.values()))
    # Prefix-aware fitness: do not punish incomplete traces (model moves to final).
    prefix_fitness = n_sync / max(n_sync + n_log, 1.0)

    return AlignmentDiagnosis(
        case_id="",
        prefix_len=len(prefix_acts),
        fitness=fitness,
        total_cost=cost,
        log_moves=log_moves,
        model_moves=model_moves,
        sync_moves=sync_moves,
        per_activity_cost=per_cost,
        deviation_transitions=deviations,
        recent_deviation=recent_dev,
        prefix_fitness=prefix_fitness,
    )


# Keep old body removed — core moved above.


def _cache_path(cache_dir: Path, case_id: str, prefix_len: int) -> Path:
    h = hashlib.md5(f"{case_id}::{prefix_len}".encode()).hexdigest()[:16]
    return cache_dir / f"{h}.pkl"


class AlignmentCache:
    """Offline cache of prefix alignment diagnoses keyed by (case_id, prefix_len)."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self._index: Dict[str, str] = {}
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def _k(self, case_id: str, prefix_len: int) -> str:
        return f"{case_id}::{prefix_len}"

    def has(self, case_id: str, prefix_len: int) -> bool:
        return self._k(case_id, prefix_len) in self._index

    def get(self, case_id: str, prefix_len: int) -> Optional[AlignmentDiagnosis]:
        key = self._k(case_id, prefix_len)
        if key not in self._index:
            return None
        path = self.cache_dir / self._index[key]
        with open(path, "rb") as f:
            return pickle.load(f)

    def put(self, diag: AlignmentDiagnosis) -> None:
        key = self._k(diag.case_id, diag.prefix_len)
        fname = f"{hashlib.md5(key.encode()).hexdigest()[:16]}.pkl"
        path = self.cache_dir / fname
        with open(path, "wb") as f:
            pickle.dump(diag, f)
        self._index[key] = fname

    def flush_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def build_for_samples(
        self,
        samples,
        net,
        im,
        fm,
        show_progress: bool = True,
    ) -> Dict[Tuple[str, int], AlignmentDiagnosis]:
        """Align all prefix samples; reuse cache when present."""
        result: Dict[Tuple[str, int], AlignmentDiagnosis] = {}
        iterator = samples
        if show_progress:
            iterator = tqdm(samples, desc="aligning prefixes")
        n_done = 0
        for s in iterator:
            key = (s.case_id, s.prefix_len)
            if self.has(s.case_id, s.prefix_len):
                diag = self.get(s.case_id, s.prefix_len)
            else:
                diag = align_prefix(s.prefix_acts, net, im, fm)
                diag.case_id = s.case_id
                diag.prefix_len = s.prefix_len
                self.put(diag)
            result[key] = diag
            n_done += 1
            if n_done % 100 == 0:
                self.flush_index()
        self.flush_index()
        return result
