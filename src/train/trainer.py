"""Training utilities and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainResult:
    history: List[Dict]
    best_val_acc: float
    best_state: Dict


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_prefix_k_metrics(
    y_true: List[int], y_pred: List[int], prefix_lens: List[int]
) -> Dict[str, float]:
    """
    ProcessTransformer-style Acc: mean of per-prefix-length accuracies.
    Also returns sample-micro Acc for comparison.
    """
    micro = compute_metrics(y_true, y_pred)
    by_k: Dict[int, List[Tuple[int, int]]] = {}
    for yt, yp, k in zip(y_true, y_pred, prefix_lens):
        by_k.setdefault(int(k), []).append((int(yt), int(yp)))
    per_k = []
    for k in sorted(by_k):
        ys = [a for a, _ in by_k[k]]
        ps = [b for _, b in by_k[k]]
        per_k.append(float(accuracy_score(ys, ps)))
    acc_macro_k = float(np.mean(per_k)) if per_k else 0.0
    return {
        **micro,
        "accuracy_macro_k": acc_macro_k,
        "n_prefix_lengths": float(len(per_k)),
    }


def train_lstm_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    aug_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    from .augment import mutate_prefix_ids

    model.train()
    total, n = 0.0, 0
    rng = rng or np.random.default_rng(0)
    for batch in loader:
        labels = batch["label"].to(device)
        graph_feat = batch.get("graph_feat")
        if graph_feat is not None:
            graph_feat = graph_feat.to(device)

        prefs = []
        for i, L in enumerate(batch["lengths"].tolist()):
            pref = batch["prefix_ids"][i, :L].tolist()
            if aug_rate > 0 and rng.random() < aug_rate:
                pref2, _, ok = mutate_prefix_ids(pref, model.num_activities, rng)
                if ok and len(pref2) >= 1:
                    pref = pref2
            prefs.append(pref)

        max_len = max(len(p) for p in prefs)
        prefix = torch.zeros(len(prefs), max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(len(prefs), dtype=torch.long, device=device)
        for i, p in enumerate(prefs):
            prefix[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            lengths[i] = len(p)

        optimizer.zero_grad()
        logits = model(prefix, lengths, graph_feat=graph_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_lstm(model, loader, device) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        prefix = batch["prefix_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["label"].to(device)
        graph_feat = batch.get("graph_feat")
        if graph_feat is not None:
            graph_feat = graph_feat.to(device)
        logits = model(prefix, lengths, graph_feat=graph_feat)
        pred = logits.argmax(dim=-1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
    return compute_metrics(ys, ps)


def fit_lstm(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    aug_rate: float = 0.0,
    seed: int = 0,
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_acc = -1.0
    best_state = None
    bad = 0
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        tr_loss = train_lstm_epoch(
            model, train_loader, optimizer, device, criterion, aug_rate=aug_rate, rng=rng
        )
        val_m = eval_lstm(model, val_loader, device)
        row = {"epoch": ep, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"[LSTM] epoch {ep:03d} loss={tr_loss:.4f} "
            f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1_macro']:.4f}"
            + (f" aug={aug_rate}" if aug_rate > 0 else "")
        )
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[LSTM] early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(history=history, best_val_acc=best_acc, best_state=best_state)


def train_hgnn_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    allowed_pairs=None,
    diagnoses=None,
    activity2id=None,
    use_extra_rels: bool = False,
    aug_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    from ..conformance.features import deviation_pairs_from_diag
    from .augment import augment_batch_prefixes

    model.train()
    total, n = 0.0, 0
    rng = rng or np.random.default_rng(0)
    use_graph = getattr(model, "graph_feat_dim", 0) > 0
    use_node = getattr(model, "node_conf_dim", 0) > 0

    for batch in loader:
        labels = batch["label"].to(device)
        prefs = []
        for i, L in enumerate(batch["lengths"].tolist()):
            prefs.append(batch["prefix_ids"][i, :L].tolist())
        graph_feat = batch.get("graph_feat")
        if graph_feat is not None:
            graph_feat = graph_feat.to(device)
        node_table = batch.get("node_feat")
        if node_table is not None:
            node_table = node_table.to(device)

        deviation_maps = None
        if use_extra_rels and diagnoses is not None and activity2id is not None:
            deviation_maps = []
            for cid, plen in zip(batch["case_ids"], batch["prefix_lens"]):
                diag = diagnoses.get((cid, plen))
                if diag is None:
                    deviation_maps.append(set())
                else:
                    deviation_maps.append(deviation_pairs_from_diag(diag, activity2id))

        if aug_rate > 0:
            prefs, graph_feat, node_table, deviation_maps = augment_batch_prefixes(
                prefs,
                num_activities=model.num_activities,
                aug_rate=aug_rate,
                rng=rng,
                graph_feat=graph_feat,
                node_table=node_table,
                use_graph_feat=use_graph,
                use_node_feat=use_node,
                use_extra_rels=use_extra_rels,
                base_deviation_maps=deviation_maps,
            )

        optimizer.zero_grad()
        logits = model.forward_batch(
            prefs,
            graph_feat=graph_feat,
            node_conf_table=node_table,
            allowed_pairs=allowed_pairs,
            deviation_maps=deviation_maps,
            use_extra_rels=use_extra_rels,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_hgnn(
    model,
    loader,
    device,
    allowed_pairs=None,
    diagnoses=None,
    activity2id=None,
    use_extra_rels: bool = False,
) -> Dict[str, float]:
    from ..conformance.features import deviation_pairs_from_diag

    model.eval()
    ys, ps = [], []
    for batch in loader:
        labels = batch["label"].to(device)
        prefs = []
        for i, L in enumerate(batch["lengths"].tolist()):
            prefs.append(batch["prefix_ids"][i, :L].tolist())
        graph_feat = batch.get("graph_feat")
        if graph_feat is not None:
            graph_feat = graph_feat.to(device)
        node_table = batch.get("node_feat")
        if node_table is not None:
            node_table = node_table.to(device)
        deviation_maps = None
        if use_extra_rels and diagnoses is not None:
            deviation_maps = []
            for cid, plen in zip(batch["case_ids"], batch["prefix_lens"]):
                diag = diagnoses.get((cid, plen))
                if diag is None:
                    deviation_maps.append(set())
                else:
                    deviation_maps.append(deviation_pairs_from_diag(diag, activity2id))
        logits = model.forward_batch(
            prefs,
            graph_feat=graph_feat,
            node_conf_table=node_table,
            allowed_pairs=allowed_pairs,
            deviation_maps=deviation_maps,
            use_extra_rels=use_extra_rels,
        )
        pred = logits.argmax(dim=-1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
    return compute_metrics(ys, ps)


@torch.no_grad()
def predict_hgnn(
    model,
    loader,
    device,
    allowed_pairs=None,
    diagnoses=None,
    activity2id=None,
    use_extra_rels: bool = False,
):
    """Return (y_true, y_pred) int lists for significance tests."""
    from ..conformance.features import deviation_pairs_from_diag

    model.eval()
    ys, ps = [], []
    for batch in loader:
        labels = batch["label"].to(device)
        prefs = []
        for i, L in enumerate(batch["lengths"].tolist()):
            prefs.append(batch["prefix_ids"][i, :L].tolist())
        graph_feat = batch.get("graph_feat")
        if graph_feat is not None:
            graph_feat = graph_feat.to(device)
        node_table = batch.get("node_feat")
        if node_table is not None:
            node_table = node_table.to(device)
        deviation_maps = None
        if use_extra_rels and diagnoses is not None:
            deviation_maps = []
            for cid, plen in zip(batch["case_ids"], batch["prefix_lens"]):
                diag = diagnoses.get((cid, plen))
                if diag is None:
                    deviation_maps.append(set())
                else:
                    deviation_maps.append(deviation_pairs_from_diag(diag, activity2id))
        logits = model.forward_batch(
            prefs,
            graph_feat=graph_feat,
            node_conf_table=node_table,
            allowed_pairs=allowed_pairs,
            deviation_maps=deviation_maps,
            use_extra_rels=use_extra_rels,
        )
        pred = logits.argmax(dim=-1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
    return ys, ps


def fit_hgnn(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    allowed_pairs=None,
    diagnoses=None,
    activity2id=None,
    use_extra_rels: bool = False,
    aug_rate: float = 0.0,
    seed: int = 0,
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_acc = -1.0
    best_state = None
    bad = 0
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        tr_loss = train_hgnn_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            allowed_pairs=allowed_pairs,
            diagnoses=diagnoses,
            activity2id=activity2id,
            use_extra_rels=use_extra_rels,
            aug_rate=aug_rate,
            rng=rng,
        )
        val_m = eval_hgnn(
            model,
            val_loader,
            device,
            allowed_pairs=allowed_pairs,
            diagnoses=diagnoses,
            activity2id=activity2id,
            use_extra_rels=use_extra_rels,
        )
        row = {"epoch": ep, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"[HGNN] epoch {ep:03d} loss={tr_loss:.4f} "
            f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1_macro']:.4f}"
            + (f" aug={aug_rate}" if aug_rate > 0 else "")
        )
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[HGNN] early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(history=history, best_val_acc=best_acc, best_state=best_state)


def train_transformer_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    use_legal_mask: bool = False,
    soft_legal_lambda: float = 0.0,
) -> Tuple[float, float]:
    """Returns (mean_loss, batch_train_accuracy)."""
    from ..conformance.legal_next import apply_legal_mask_to_logits, soft_illegal_loss

    model.train()
    total, n, correct = 0.0, 0, 0
    for batch in loader:
        prefix = batch["prefix_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["label"].to(device)
        key_pad = ~batch["mask"].to(device)
        time_feat = batch.get("time_feat")
        if time_feat is not None:
            time_feat = time_feat.to(device)
        resource_ids = batch.get("resource_ids")
        if resource_ids is not None:
            resource_ids = resource_ids.to(device)
        attr_ids = batch.get("attr_ids")
        if attr_ids is not None:
            attr_ids = attr_ids.to(device)
        legal_mask = batch.get("legal_mask")
        if legal_mask is not None:
            legal_mask = legal_mask.to(device)
        optimizer.zero_grad()
        logits = model(
            prefix,
            lengths,
            key_padding_mask=key_pad,
            time_feat=time_feat,
            resource_ids=resource_ids,
            attr_ids=attr_ids,
        )
        logits_for_ce = logits
        if use_legal_mask and legal_mask is not None:
            logits_for_ce = apply_legal_mask_to_logits(logits, legal_mask)
        loss = criterion(logits_for_ce, labels)
        if soft_legal_lambda > 0 and legal_mask is not None:
            # Soft illegal on raw logits (before hard mask)
            loss = loss + float(soft_legal_lambda) * soft_illegal_loss(
                logits, legal_mask, entropy_weight=True
            )
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
        correct += int((logits_for_ce.argmax(dim=-1) == labels).sum().item())
    return total / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def eval_transformer(
    model,
    loader,
    device,
    use_legal_mask: bool = False,
    gate_tau: float | None = None,
    uncertainty: str = "entropy",
) -> Dict[str, float]:
    from ..conformance.legal_next import (
        apply_gated_legal_mask,
        apply_legal_mask_to_logits,
        prediction_uncertainty,
    )

    model.eval()
    ys, ps, ks = [], [], []
    legal_hits, legal_n = 0, 0
    u_scores: List[float] = []
    gated_n = 0
    for batch in loader:
        prefix = batch["prefix_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["label"].to(device)
        key_pad = ~batch["mask"].to(device)
        time_feat = batch.get("time_feat")
        if time_feat is not None:
            time_feat = time_feat.to(device)
        resource_ids = batch.get("resource_ids")
        if resource_ids is not None:
            resource_ids = resource_ids.to(device)
        attr_ids = batch.get("attr_ids")
        if attr_ids is not None:
            attr_ids = attr_ids.to(device)
        legal_mask = batch.get("legal_mask")
        if legal_mask is not None:
            legal_mask = legal_mask.to(device)
        logits = model(
            prefix,
            lengths,
            key_padding_mask=key_pad,
            time_feat=time_feat,
            resource_ids=resource_ids,
            attr_ids=attr_ids,
        )
        if legal_mask is not None:
            u = prediction_uncertainty(logits, kind=uncertainty)
            u_scores.extend(u.cpu().tolist())
            if use_legal_mask:
                logits = apply_legal_mask_to_logits(logits, legal_mask)
            elif gate_tau is not None and gate_tau >= 0:
                gated_n += int((u >= float(gate_tau)).sum().item())
                logits = apply_gated_legal_mask(
                    logits,
                    legal_mask,
                    tau=float(gate_tau),
                    uncertainty=uncertainty,
                )
        pred = logits.argmax(dim=-1)
        if legal_mask is not None:
            legal_hits += int(
                legal_mask[torch.arange(legal_mask.size(0), device=device), pred].sum().item()
            )
            legal_n += int(pred.numel())
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        ks.extend(batch.get("prefix_lens", lengths.cpu().tolist()))
    out = compute_prefix_k_metrics(ys, ps, ks)
    if legal_n > 0:
        out["legal_rate"] = float(legal_hits / legal_n)
    if u_scores:
        out["uncertainty_mean"] = float(np.mean(u_scores))
        out["uncertainty_p70"] = float(np.quantile(u_scores, 0.70))
        out["uncertainty_p80"] = float(np.quantile(u_scores, 0.80))
        # Keep entropy_* keys when using entropy (backward compatible for calibration)
        if str(uncertainty).lower() in ("entropy", "ent", "h"):
            out["entropy_mean"] = out["uncertainty_mean"]
            out["entropy_p70"] = out["uncertainty_p70"]
            out["entropy_p80"] = out["uncertainty_p80"]
    if gate_tau is not None and gate_tau >= 0 and legal_n > 0:
        out["gate_fire_rate"] = float(gated_n / legal_n)
        out["gate_tau"] = float(gate_tau)
        out["uncertainty"] = str(uncertainty)
    return out


def fit_transformer(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    use_legal_mask: bool = False,
    soft_legal_lambda: float = 0.0,
    gate_tau: float | None = None,
    label_smoothing: float = 0.0,
    early_stop_metric: str = "accuracy",
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = []
    best_acc = -1.0
    best_state = None
    bad = 0
    if use_legal_mask:
        tag = "TF+mask"
    elif soft_legal_lambda > 0 and gate_tau is not None and gate_tau >= 0:
        tag = "TF+SPC"
    elif soft_legal_lambda > 0:
        tag = "TF+soft"
    else:
        tag = "TF"
    # ProcessTransformer checkpoints on train accuracy; we support that for protocol align.
    use_train_score = early_stop_metric in ("train_accuracy", "train_acc")
    metric_key = (
        "accuracy"
        if use_train_score
        else (early_stop_metric if early_stop_metric in ("accuracy", "accuracy_macro_k") else "accuracy")
    )
    has_val = val_loader is not None and len(getattr(val_loader, "dataset", [])) > 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_transformer_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            use_legal_mask=use_legal_mask,
            soft_legal_lambda=soft_legal_lambda,
        )
        # During training selection, evaluate without gate (unless hard-mask mode)
        eval_tau = None if use_legal_mask else None
        if use_train_score:
            score = float(tr_acc)
            if has_val and val_loader is not train_loader:
                val_m = eval_transformer(
                    model,
                    val_loader,
                    device,
                    use_legal_mask=use_legal_mask,
                    gate_tau=eval_tau,
                )
            else:
                val_m = {
                    "accuracy": score,
                    "f1_macro": 0.0,
                    "f1_weighted": 0.0,
                    "accuracy_macro_k": 0.0,
                }
        elif not has_val:
            score_m = eval_transformer(
                model, train_loader, device, use_legal_mask=use_legal_mask, gate_tau=eval_tau
            )
            score = float(score_m["accuracy"])
            val_m = score_m
        else:
            val_m = eval_transformer(
                model, val_loader, device, use_legal_mask=use_legal_mask, gate_tau=eval_tau
            )
            score = float(val_m.get(metric_key, val_m["accuracy"]))
        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "train_accuracy": float(tr_acc),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        print(
            f"[{tag}] epoch {ep:03d} loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_acc={val_m['accuracy']:.4f} val_macro_k={val_m.get('accuracy_macro_k', 0):.4f} "
            f"score={score:.4f} val_f1={val_m.get('f1_macro', 0):.4f}",
            flush=True,
        )
        if score > best_acc:
            best_acc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(history=history, best_val_acc=best_acc, best_state=best_state)
