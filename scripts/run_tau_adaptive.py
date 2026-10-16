#!/usr/bin/env python

"""

Adaptive τ selection on validation (Acc budget → max PCP), then test eval.



Compares selected percentile vs fixed p70 using SoftLegal checkpoints

(inference-only; no retrain).

"""



from __future__ import annotations



import argparse

import json

import pickle

import sys

from pathlib import Path

from typing import Any, Dict, List



import numpy as np

import torch

from torch.utils.data import DataLoader



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))



from scripts.run_spc_study import (  # noqa: E402

    _build_model,

    _build_noise_prefixes_fast,

    eval_transformer_on_samples,

)

from scripts.tau_gate import (  # noqa: E402

    DEFAULT_ACC_BUDGET,

    DEFAULT_PERCENTILES,

    build_noise_prefixes_for_split,

    select_tau_adaptive,

    val_entropy_quantile,

)

from scripts.utils import load_config, save_json  # noqa: E402

from src.data.prefixes import PrefixDataset, collate_pad  # noqa: E402



DEFAULT_CONFIGS = [

    "configs/helpdesk_spc.yaml",

    "configs/traffic_spc.yaml",

    "configs/bpic2012_spc.yaml",

    "configs/sepsis_spc.yaml",

    "configs/bpic2019_spc.yaml",

]





def _pick_soft_ckpt(out_dir: Path, seed: int) -> Path:

    soft = out_dir / f"soft_seed{seed}.pt"

    spc = out_dir / f"spc_seed{seed}.pt"

    if soft.exists():

        return soft

    if spc.exists():

        return spc

    raise FileNotFoundError(f"No soft/spc checkpoint for seed={seed} in {out_dir}")





def _val_loader(datasets, legal_masks, num_activities, batch_size):

    ds = PrefixDataset(datasets["val"], legal_masks=legal_masks, num_activities=num_activities)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)





def run_one(

    cfg_path: str,

    seeds: List[int],

    percentiles: List[float],

    acc_budget: float,

    noise_rate: float,

    device: torch.device,

) -> Dict[str, Any]:

    cfg = load_config(cfg_path)

    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")

    bs = int(cfg.get("batch_size", 32))



    with open(out_dir / "bundle.pkl", "rb") as f:

        bundle = pickle.load(f)

    with open(out_dir / "prefixes.pkl", "rb") as f:

        datasets = pickle.load(f)

    with open(out_dir / "legal_next.pkl", "rb") as f:

        legal_masks = pickle.load(f)["masks"]



    # ensure test + val noise caches

    test_noise = out_dir / f"noise_{noise_rate}" / "prefixes_test.pkl"

    if not test_noise.exists():

        _build_noise_prefixes_fast(cfg, out_dir, float(noise_rate))

    val_noise_path = out_dir / f"noise_{noise_rate}" / "prefixes_val.pkl"

    if val_noise_path.exists():

        with open(val_noise_path, "rb") as f:

            noisy_val = pickle.load(f)

    else:

        noisy_val, _ = build_noise_prefixes_for_split(cfg, out_dir, float(noise_rate), "val")



    with open(test_noise, "rb") as f:

        noisy_test = pickle.load(f)



    rows: List[Dict[str, Any]] = []

    for seed in seeds:

        pt_path = _pick_soft_ckpt(out_dir, seed)

        print(f"[{cfg.get('dataset')}] seed={seed} ckpt={pt_path.name}", flush=True)

        model = _build_model(cfg, bundle, datasets, out_dir, device)

        model.load_state_dict(torch.load(pt_path, map_location=device))

        model.eval()



        vloader = _val_loader(datasets, legal_masks, bundle.num_activities, bs)

        sel = select_tau_adaptive(

            model,

            val_loader=vloader,

            noisy_val_samples=noisy_val,

            out_dir=out_dir,

            num_activities=bundle.num_activities,

            batch_size=bs,

            device=device,

            percentiles=percentiles,

            acc_budget=acc_budget,

        )

        tau_p70 = val_entropy_quantile(model, vloader, device, 0.70)



        def _pack(name: str, tau: float | None) -> Dict[str, Any]:

            clean = eval_transformer_on_samples(

                model,

                datasets["test"],

                out_dir,

                bundle.num_activities,

                bs,

                device,

                use_legal_mask=False,

                gate_tau=tau,

            )

            noise = eval_transformer_on_samples(

                model,

                noisy_test,

                out_dir,

                bundle.num_activities,

                bs,

                device,

                use_legal_mask=False,

                gate_tau=tau,

            )

            return {

                "setting": name,

                "gate_tau": tau,

                "clean_accuracy": clean.get("accuracy"),

                "clean_legal_rate": clean.get("legal_rate"),

                "noise_accuracy": noise.get("accuracy"),

                "noise_legal_rate": noise.get("legal_rate"),

                "gate_fire_rate": noise.get("gate_fire_rate"),

            }



        soft_row = _pack("soft", None)

        p70_row = _pack("p70", float(tau_p70))

        ada_row = _pack("adaptive", float(sel["gate_tau"]))

        ada_row["selected_percentile"] = sel["selected_percentile"]

        ada_row["selected_label"] = sel["selected_label"]

        ada_row["rule"] = sel["rule"]

        ada_row["val_select"] = sel



        for r in (soft_row, p70_row, ada_row):

            r["seed"] = seed

            r["ckpt"] = pt_path.name

            rows.append(r)



        print(

            f"  select={sel['selected_label']} rule={sel['rule']} "

            f"test_nPCP soft={soft_row['noise_legal_rate']:.4f} "

            f"p70={p70_row['noise_legal_rate']:.4f} "

            f"ada={ada_row['noise_legal_rate']:.4f}",

            flush=True,

        )



    # aggregate

    by_set: Dict[str, List[Dict]] = {}

    for r in rows:

        by_set.setdefault(r["setting"], []).append(r)



    def _agg(setting: str, key: str):

        vals = [float(r[key]) for r in by_set.get(setting, []) if r.get(key) is not None]

        if not vals:

            return None

        arr = np.asarray(vals, dtype=float)

        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "values": vals}



    labels = [r["selected_label"] for r in by_set.get("adaptive", [])]

    summary = {

        "soft": {k: _agg("soft", k) for k in ("clean_accuracy", "noise_accuracy", "noise_legal_rate")},

        "p70": {k: _agg("p70", k) for k in ("clean_accuracy", "noise_accuracy", "noise_legal_rate")},

        "adaptive": {k: _agg("adaptive", k) for k in ("clean_accuracy", "noise_accuracy", "noise_legal_rate")},

        "selected_labels": labels,

    }

    return {

        "dataset": cfg.get("dataset"),

        "config": cfg_path,

        "seeds": seeds,

        "percentiles": percentiles,

        "acc_budget": acc_budget,

        "noise_rate": noise_rate,

        "per_run": rows,

        "summary": summary,

    }





def _fmt_row(name: str, s: Dict) -> str:

    def g(block, key):

        b = block.get(key)

        return f"{b['mean']:.4f}±{b['std']:.4f}" if b else "n/a"



    return (

        f"{name:10s} cleanAcc={g(s, 'clean_accuracy')}  "

        f"nAcc={g(s, 'noise_accuracy')}  nPCP={g(s, 'noise_legal_rate')}"

    )





def main() -> int:

    ap = argparse.ArgumentParser(description="Adaptive SPC gate τ on val Acc budget")

    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)

    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])

    ap.add_argument("--percentiles", nargs="+", type=float, default=DEFAULT_PERCENTILES)

    ap.add_argument("--acc_budget", type=float, default=DEFAULT_ACC_BUDGET)

    ap.add_argument("--noise_rate", type=float, default=0.2)

    ap.add_argument(

        "--out-dir",

        type=str,

        default=str(ROOT / "outputs" / "tau_adaptive"),

    )

    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))



    all_results = []

    lines = [

        "Adaptive τ (val noise Acc budget → max PCP)",

        f"budget={args.acc_budget}  noise_rate={args.noise_rate}  percentiles={args.percentiles}",

        "",

    ]

    for cfg_path in args.configs:

        cfg_full = str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path

        if not Path(cfg_full).exists():

            cfg_full = cfg_path

        print(f"== adaptive {cfg_full} ==", flush=True)

        try:

            ds = run_one(

                cfg_full,

                list(args.seeds),

                list(args.percentiles),

                float(args.acc_budget),

                float(args.noise_rate),

                device,

            )

        except FileNotFoundError as e:

            print(f"skip: {e}", flush=True)

            continue

        all_results.append(ds)

        save_json(out_dir / f"tau_adaptive_{ds['dataset']}.json", ds)

        s = ds["summary"]

        lines.append(f"dataset={ds['dataset']}  selected={s['selected_labels']}")

        lines.append("  " + _fmt_row("soft", s["soft"]))

        lines.append("  " + _fmt_row("p70", s["p70"]))

        lines.append("  " + _fmt_row("adaptive", s["adaptive"]))

        lines.append("")



    payload = {

        "acc_budget": args.acc_budget,

        "noise_rate": args.noise_rate,

        "percentiles": args.percentiles,

        "seeds": args.seeds,

        "datasets": [d["dataset"] for d in all_results],

        "note": (

            "τ selected on val noisy Acc budget vs Soft; maximize val PCP; "

            "test metrics reported for soft / fixed p70 / adaptive."

        ),

        "results": all_results,

    }

    save_json(out_dir / "tau_adaptive.json", payload)

    txt = "\n".join(lines) + "\n"

    (out_dir / "tau_adaptive.txt").write_text(txt, encoding="utf-8")

    print(txt, flush=True)

    print(f"Wrote {out_dir}", flush=True)

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


