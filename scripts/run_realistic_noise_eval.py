#!/usr/bin/env python
"""
Realistic recording-error noise eval (Helpdesk by default).

Isolates two noise types (vs mixed skip/insert/swap ladder):
  - missing  = skip only   (incomplete logging)
  - reorder  = swap only   (adjacent out-of-order)

Protocol matches SPC study: mutate test cases, rebuild prefixes, evaluate
Base / HardMask / SoftLegal / SPC from existing checkpoints.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_spc_study import _build_model, eval_transformer_on_samples  # noqa: E402
from scripts.utils import load_config, save_json  # noqa: E402
from src.data.prefixes import build_prefix_dataset  # noqa: E402
from src.experiments import inject_control_flow_noise  # noqa: E402

# (folder_name, modes, rate)
DEFAULT_SCENARIOS: List[Tuple[str, Sequence[str], float]] = [
    ("missing_0.1", ("skip",), 0.1),
    ("missing_0.2", ("skip",), 0.2),
    ("reorder_0.1", ("swap",), 0.1),
    ("reorder_0.2", ("swap",), 0.2),
]

VARIANTS = ("base", "hardmask", "soft", "spc")


def _build_scenario_prefixes(
    cfg: Dict[str, Any],
    out_dir: Path,
    name: str,
    modes: Sequence[str],
    rate: float,
    force: bool,
) -> Path:
    noise_dir = out_dir / f"noise_{name}"
    pref_path = noise_dir / "prefixes_test.pkl"
    if pref_path.exists() and not force:
        return pref_path

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)
    test_only = {cid: bundle.cases[cid] for cid in split.test_ids}
    activities = list(bundle.activity2id.keys())
    seed = int(cfg.get("seed", 42)) + int(rate * 1000) + (11 if "missing" in name else 17)
    noisy_test = inject_control_flow_noise(
        test_only,
        activities,
        rate=rate,
        seed=seed,
        modes=modes,
        per_event=True,
    )
    samples = build_prefix_dataset(
        noisy_test,
        split.test_ids,
        bundle.activity2id,
        cfg.get("min_prefix_len", 1),
        max_prefix_len=cfg.get("max_prefix_len"),
    )
    noise_dir.mkdir(parents=True, exist_ok=True)
    with open(pref_path, "wb") as f:
        pickle.dump(samples, f)
    meta = {
        "name": name,
        "modes": list(modes),
        "rate": rate,
        "seed": seed,
        "n_prefixes": len(samples),
        "protocol": "mutate_test_cases_rebuild_prefixes",
    }
    save_json(noise_dir / "meta.json", meta)
    print(f"[{name}] prefixes={len(samples)} modes={modes} -> {noise_dir}", flush=True)
    return pref_path


def _eval_kwargs(variant: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    if variant == "hardmask":
        return {"use_legal_mask": True, "gate_tau": None}
    if variant == "spc":
        tau = metrics.get("gate_tau")
        if tau is None:
            tau = (metrics.get("test") or {}).get("gate_tau")
        return {
            "use_legal_mask": False,
            "gate_tau": float(tau) if tau is not None else None,
        }
    return {"use_legal_mask": False, "gate_tau": None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Realistic missing/reorder noise eval")
    ap.add_argument("--config", default=str(ROOT / "configs" / "helpdesk_spc.yaml"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--force", action="store_true", help="Rebuild noisy prefixes")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    report_dir = out_dir / "realistic_noise"
    report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    bs = int(cfg.get("batch_size", 32))

    with open(out_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    # ensure scenario caches
    scenario_paths: Dict[str, Path] = {}
    for name, modes, rate in DEFAULT_SCENARIOS:
        scenario_paths[name] = _build_scenario_prefixes(
            cfg, out_dir, name, modes, rate, force=args.force
        )

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            tag = f"{variant}_seed{seed}"
            pt_path = out_dir / f"{tag}.pt"
            metrics_path = out_dir / f"{tag}_metrics.json"
            if not pt_path.exists():
                print(f"skip missing ckpt {pt_path.name}", flush=True)
                continue
            metrics = {}
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            eval_kw = _eval_kwargs(variant, metrics)

            model = _build_model(cfg, bundle, datasets, out_dir, device)
            model.load_state_dict(torch.load(pt_path, map_location=device))
            model.eval()

            row: Dict[str, Any] = {
                "seed": seed,
                "variant": variant,
                "gate_tau": eval_kw.get("gate_tau"),
            }
            for name, pref_path in scenario_paths.items():
                with open(pref_path, "rb") as f:
                    samples = pickle.load(f)
                m = eval_transformer_on_samples(
                    model,
                    samples,
                    out_dir,
                    bundle.num_activities,
                    bs,
                    device,
                    use_legal_mask=bool(eval_kw.get("use_legal_mask")),
                    gate_tau=eval_kw.get("gate_tau"),
                )
                row[f"{name}_accuracy"] = m.get("accuracy")
                row[f"{name}_legal_rate"] = m.get("legal_rate")
                row[f"{name}_gate_fire_rate"] = m.get("gate_fire_rate")
                print(
                    f"[{variant} seed={seed} {name}] "
                    f"acc={m.get('accuracy'):.4f} legal={m.get('legal_rate')}",
                    flush=True,
                )
            rows.append(row)

    # aggregate
    by_var: Dict[str, List[Dict]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)

    metric_keys = []
    for name, _, _ in DEFAULT_SCENARIOS:
        metric_keys.extend([f"{name}_accuracy", f"{name}_legal_rate"])

    def _agg(key: str) -> Dict[str, Any]:
        out = {}
        for var, rs in by_var.items():
            vals = [float(r[key]) for r in rs if r.get(key) is not None]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[var] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "values": vals,
            }
        return out

    summary = {k: _agg(k) for k in metric_keys}
    payload = {
        "dataset": cfg.get("dataset"),
        "scenarios": [
            {"name": n, "modes": list(m), "rate": r} for n, m, r in DEFAULT_SCENARIOS
        ],
        "note": (
            "Realistic recording errors isolated from mixed CF ladder: "
            "missing=skip-only, reorder=adjacent-swap-only; per-event Bernoulli."
        ),
        "per_run": rows,
        "summary": summary,
    }
    save_json(report_dir / "realistic_noise.json", payload)

    lines = [
        "Realistic noise (missing / reorder)",
        f"dataset={cfg.get('dataset')}",
        payload["note"],
        "",
    ]
    for k, agg in summary.items():
        lines.append(k)
        for var, st in agg.items():
            lines.append(f"  {var}: {st['mean']:.4f}±{st['std']:.4f}")
        lines.append("")
    text = "\n".join(lines)
    (report_dir / "realistic_noise.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Wrote {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
