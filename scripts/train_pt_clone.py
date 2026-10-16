#!/usr/bin/env python
"""Train/eval a faithful ProcessTransformer clone on helpdesk_pt data."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_json, set_seed
from src.models.pt_clone import ProcessTransformerClone, prepad_batch
from src.train.trainer import compute_prefix_k_metrics


class PTPrefixDS(Dataset):
    def __init__(self, samples, x_word: dict, y_word: dict):
        self.rows = []
        for s in samples:
            # map activity names -> PT-style token ids (after optional normalize in prepare)
            ids = [x_word[a] for a in s.prefix_acts]
            y = y_word[s.label_act]
            self.rows.append((ids, y, s.prefix_len))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, max_len: int, pad_id: int = 0):
    seqs, ys, ks = zip(*batch)
    x = prepad_batch(seqs, max_len, pad_id=pad_id)
    y = torch.tensor(ys, dtype=torch.long)
    return {"x": x, "y": y, "k": list(ks)}


def eval_loader(model, loader, device):
    model.eval()
    ys, ps, ks = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            pred = logits.argmax(dim=-1).cpu().tolist()
            ys.extend(batch["y"].tolist())
            ps.extend(pred)
            ks.extend(batch["k"])
    m = compute_prefix_k_metrics(ys, ps, ks)
    # PT next_activity.py quirk: append last k Acc again then mean
    by_k = {}
    for yt, yp, k in zip(ys, ps, ks):
        by_k.setdefault(int(k), []).append((yt, yp))
    per_k = []
    for k in sorted(by_k):
        a = [t for t, _ in by_k[k]]
        b = [p for _, p in by_k[k]]
        per_k.append(float(accuracy_score(a, b)))
    pt_quirk = float(np.mean(per_k + [per_k[-1]])) if per_k else 0.0
    m["accuracy_pt_quirk"] = pt_quirk
    m["f1_weighted"] = float(f1_score(ys, ps, average="weighted", zero_division=0))
    return m


def build_vocabs(train_samples, normalize: bool):
    def norm(a: str) -> str:
        if not normalize:
            return a
        return a.lower().replace(" ", "-")

    acts = sorted({norm(a) for s in train_samples for a in s.prefix_acts + [s.label_act]})
    # PT: x has [PAD],[UNK] then activities; y is activities only (0..C-1)
    x_word = {"[PAD]": 0, "[UNK]": 1}
    for a in acts:
        x_word[a] = len(x_word)
    y_word = {a: i for i, a in enumerate(acts)}
    return x_word, y_word, norm


def remap_samples(samples, norm_fn):
    out = []
    for s in samples:
        # shallow remap activity strings
        class _S:
            pass

        o = _S()
        o.prefix_acts = [norm_fn(a) for a in s.prefix_acts]
        o.label_act = norm_fn(s.label_act)
        o.prefix_len = s.prefix_len
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "helpdesk_pt.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no_normalize_acts", action="store_true", help="Skip remapping (prepare already normalized)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 100))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 12))
    lr = args.lr if args.lr is not None else float(cfg.get("lr", 1e-3))
    # If prepare already normalized, remap is identity; still safe to normalize again.
    do_norm = (not args.no_normalize_acts) and bool(cfg.get("normalize_activity_names", True))

    set_seed(seed)
    out_dir = ROOT / cfg.get("output_dir", "outputs") / cfg.get("dataset", "run")
    device = torch.device(cfg.get("device", "cpu"))

    with open(out_dir / "prefixes.pkl", "rb") as f:
        datasets = pickle.load(f)

    x_word, y_word, norm_fn = build_vocabs(datasets["train"], do_norm)
    train_s = remap_samples(datasets["train"], norm_fn)
    test_s = remap_samples(datasets["test"], norm_fn)

    max_len = max(len(s.prefix_acts) for s in train_s)
    train_ds = PTPrefixDS(train_s, x_word, y_word)
    test_ds = PTPrefixDS(test_s, x_word, y_word)

    def _collate(b):
        return collate(b, max_len=max_len, pad_id=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    embed_dim = int(cfg.get("tf_embed_dim", 36))
    num_heads = int(cfg.get("tf_nhead", 4))
    ff_dim = int(cfg.get("tf_ff_dim", 64))
    dropout = float(cfg.get("tf_dropout", 0.1))

    model = ProcessTransformerClone(
        vocab_size=len(x_word),
        num_activities=len(y_word),
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))
    crit = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total, n, correct = 0.0, 0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
            correct += int((logits.argmax(-1) == y).sum().item())
        train_acc = correct / max(n, 1)
        history.append({"epoch": ep, "loss": total / max(n, 1), "train_acc": train_acc})
        print(f"[PT-clone] epoch {ep:03d} loss={total/max(n,1):.4f} train_acc={train_acc:.4f}", flush=True)
        if train_acc > best_acc:
            best_acc = train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = eval_loader(model, test_loader, device)
    print("TEST", test_m, flush=True)

    tag = f"pt_clone_seed{seed}"
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    save_json(
        out_dir / f"{tag}_metrics.json",
        {
            "test": test_m,
            "best_train_acc": best_acc,
            "max_len": max_len,
            "vocab_size": len(x_word),
            "num_activities": len(y_word),
            "normalize_acts": do_norm,
            "embed_dim": embed_dim,
            "num_layers": 1,
            "nhead": num_heads,
            "ff_dim": ff_dim,
            "history": history,
            "note": "Faithful PT clone: prepad+GAP+1block",
        },
    )


if __name__ == "__main__":
    main()
