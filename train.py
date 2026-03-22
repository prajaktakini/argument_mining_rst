"""
train.py — training entry point for Tasks 0 (relation ID) + 1 (evidence detection)
Dataset: UKP Essays AAEC (loaded automatically from HuggingFace pie/aae2)

Usage:
    # Full model
    python train.py

    # Ablation: untyped edges (all RST edge types → one weight matrix)
    python train.py --ablation_untyped_edges

    # Ablation: no nuclearity salience weights
    python train.py --ablation_no_nuclearity

    # Ablation: RoBERTa baseline (no R-GCN at all)
    python train.py --ablation_no_graph

    # Single-task: relation identification only
    python train.py --single_task relation

    # Single-task: evidence detection only
    python train.py --single_task evidence

    # Override hyperparameters
    python train.py --epochs 20 --lr_rgcn 5e-4 --seed 0
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from rst_am_config import DataConfig, ModelConfig, TrainConfig
from data.datasets import AAECLoader, GraphDatasetBuilder, dataset_stats
from data.rst_pipeline import RSTParser, SpanAligner
from evaluate import Evaluator, compare_ablations
from models.multitask import RSTArgumentMiner

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="RST-AM: Tasks 0 + 1 on AAEC")

    # Ablation flags
    p.add_argument("--ablation_untyped_edges", action="store_true",
                   help="Collapse all RST edge types → one R-GCN weight matrix")
    p.add_argument("--ablation_no_nuclearity", action="store_true",
                   help="Set all node weights to 1.0 (ignore nuclearity)")
    p.add_argument("--ablation_no_graph",      action="store_true",
                   help="Skip R-GCN entirely — RoBERTa-only baseline")
    p.add_argument("--single_task", type=str, default=None,
                   choices=["relation", "evidence"],
                   help="Train only one head (single-task ablation)")

    # Hyperparameters
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr_rgcn",    type=float, default=None)
    p.add_argument("--lr_encoder", type=float, default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--neg_ratio",  type=float, default=1.0,
                   help="Ratio of negative relation pairs to positive (default 1.0)")

    # Paths
    p.add_argument("--cache_dir",    type=str, default=None)
    p.add_argument("--output_dir",   type=str, default="outputs/")
    p.add_argument("--force_reparse", action="store_true",
                   help="Ignore cached RST graphs and re-parse everything")

    # Logging
    p.add_argument("--use_wandb",  action="store_true")
    p.add_argument("--run_name",   type=str, default=None)

    return p.parse_args()


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    parts = ["rst_am"]
    if args.ablation_untyped_edges: parts.append("untyped")
    if args.ablation_no_nuclearity: parts.append("no_nuc")
    if args.ablation_no_graph:      parts.append("no_graph")
    if args.single_task:            parts.append(f"single_{args.single_task}")
    return "_".join(parts)


def main():
    args = parse_args()
    run_name = build_run_name(args)

    # ── Config ──────────────────────────────────────────────────────────────
    dcfg = DataConfig()
    mcfg = ModelConfig()
    tcfg = TrainConfig()

    # Apply ablation flags to both model and train config
    for flag in ("ablation_untyped_edges", "ablation_no_nuclearity", "ablation_no_graph"):
        val = getattr(args, flag)
        setattr(tcfg, flag, val)
        setattr(mcfg, flag, val)

    # Apply hyperparameter overrides
    for attr in ("epochs", "batch_size", "lr_rgcn", "lr_encoder", "seed"):
        v = getattr(args, attr)
        if v is not None:
            setattr(tcfg, attr, v)
    if args.cache_dir:
        dcfg.cache_dir = args.cache_dir

    tcfg.use_wandb = args.use_wandb
    tcfg.run_name  = run_name

    set_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Run: %s", device, run_name)

    # ── W&B ─────────────────────────────────────────────────────────────────
    if tcfg.use_wandb:
        import wandb
        wandb.init(
            project=tcfg.wandb_project,
            name=run_name,
            config={**vars(mcfg), **vars(tcfg)},
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    loader  = AAECLoader(neg_ratio=args.neg_ratio, seed=tcfg.seed)
    parser = RSTParser(dmrst_dir=dcfg.rst_model, device=device)

    aligner  = SpanAligner(overlap_thresh=dcfg.overlap_thresh)
    builder  = GraphDatasetBuilder(
        rst_parser=parser,
        span_aligner=aligner,
        cache_dir=dcfg.cache_dir,
        force_reparse=args.force_reparse,
    )

    logger.info("Loading AAEC splits...")
    train_docs = loader.load("train")
    dev_docs   = loader.load("validation")
    test_docs  = loader.load("test")

    logger.info("AAEC train stats: %s", dataset_stats(train_docs))

    logger.info("Building graph samples...")
    train_samples = builder.build(train_docs)
    dev_samples   = builder.build(dev_docs)
    test_samples  = builder.build(test_docs)
    logger.info(
        "Samples — train: %d | dev: %d | test: %d",
        len(train_samples), len(dev_samples), len(test_samples),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = RSTArgumentMiner(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    encoder_params = list(model.encoder.parameters())
    other_params   = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]
    optimizer = torch.optim.AdamW([
        {"params": other_params,   "lr": tcfg.lr_rgcn},
        {"params": encoder_params, "lr": tcfg.lr_encoder},
    ], weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg.epochs
    )

    evaluator  = Evaluator(device=device)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    no_improve = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, tcfg.epochs + 1):

        # Unfreeze encoder after warmup
        if epoch == mcfg.warmup_epochs + 1:
            model.encoder.unfreeze()
            logger.info("Epoch %d: encoder unfrozen", epoch)

        model.train()
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(train_samples), tcfg.batch_size):
            batch = train_samples[i : i + tcfg.batch_size]
            if not batch:
                continue

            optimizer.zero_grad()
            loss = _batch_loss(model, batch, args.single_task, device)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # ── Evaluation ───────────────────────────────────────────────────
        if epoch % tcfg.eval_every == 0:
            metrics   = evaluator.evaluate(model, dev_samples)
            dev_score = _primary_metric(metrics, args.single_task)

            logger.info(
                "Epoch %3d | loss=%.4f | dev=%.4f | rel_f1=%.4f | ev_f1=%.4f",
                epoch, avg_loss, dev_score,
                metrics.get("relation_macro_f1", 0.0),
                metrics.get("evidence_f1", 0.0),
            )

            if tcfg.use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": avg_loss, **metrics})

            if dev_score > best_score:
                best_score = dev_score
                no_improve = 0
                ckpt = output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                }, ckpt)
                logger.info("  → saved best model (score=%.4f)", best_score)
            else:
                no_improve += 1
                if no_improve >= tcfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

    # ── Test evaluation ──────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluator.evaluate(model, test_samples, verbose=True)
    logger.info(
        "TEST | rel_macro_f1=%.4f | rel_support_f1=%.4f | rel_attack_f1=%.4f"
        " | ev_f1=%.4f | ev_prec=%.4f | ev_rec=%.4f",
        test_metrics.get("relation_macro_f1",  0.0),
        test_metrics.get("relation_support_f1", 0.0),
        test_metrics.get("relation_attack_f1",  0.0),
        test_metrics.get("evidence_f1",         0.0),
        test_metrics.get("evidence_precision",  0.0),
        test_metrics.get("evidence_recall",     0.0),
    )

    # Save
    with open(output_dir / "test_results.json", "w") as f:
        json.dump({"run_name": run_name, **test_metrics}, f, indent=2)

    if tcfg.use_wandb:
        import wandb
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})
        wandb.finish()

    return test_metrics


# ─── Batch loss ───────────────────────────────────────────────────────────────


def _batch_loss(model, batch, single_task, device) -> torch.Tensor:
    """Compute combined loss for one mini-batch of AMGraphSamples."""

    rel_weights = torch.tensor([1.0, 15.0, 1.0], device=device)
    ev_weights  = torch.tensor([1.0, 2.0],        device=device)

    total = torch.tensor(0.0, requires_grad=True, device=device)

    for sample in batch:
        h = model.encode(sample).to(device)   # (n_edus, D)

        # ── Pool EDU embeddings → span embeddings (shared by both tasks) ──
        n_spans = sample.n_spans
        D       = h.size(-1)
        span_h  = torch.zeros(n_spans, D, device=device)
        count   = torch.zeros(n_spans, 1, device=device)

        for edu_idx, span_idx in sample.edu_to_span.items():
            if span_idx is None or span_idx >= n_spans:
                continue
            if edu_idx >= h.size(0):
                continue
            span_h[span_idx] += h[edu_idx]
            count[span_idx]  += 1

        fallback = h.mean(dim=0)
        unaligned = (count.squeeze(-1) == 0)
        span_h[unaligned] = fallback
        span_h = span_h / count.clamp(min=1)   # (n_spans, D)

        # ── Task 0: relation identification ───────────────────────────────
        if single_task != "evidence" and sample.relations:
            # filter out any pairs with out-of-range span indices
            valid = [(r.src_id, r.tgt_id, r.rel_type)
                     for r in sample.relations
                     if r.src_id < n_spans and r.tgt_id < n_spans]
            if valid:
                pairs = torch.tensor(
                    [[s, t] for s, t, _ in valid],
                    dtype=torch.long, device=device,
                )
                labels = torch.tensor(
                    [rt for _, _, rt in valid],
                    dtype=torch.long, device=device,
                )
                logits = model.forward_relation(span_h, pairs)
                loss_r = torch.nn.functional.cross_entropy(
                    logits, labels, weight=rel_weights
                )
                total = total + model.task_weights[0] * loss_r

        # ── Task 1: evidence detection ─────────────────────────────────────
        if single_task != "relation" and sample.span_labels:
            labels_ev = torch.tensor(
                sample.span_labels, dtype=torch.long, device=device
            )
            logits_ev = model.forward_evidence(span_h)
            loss_e    = torch.nn.functional.cross_entropy(
                logits_ev, labels_ev, weight=ev_weights
            )
            total = total + model.task_weights[1] * loss_e

    return total

def _primary_metric(metrics: dict, single_task) -> float:
    """Pick the headline dev metric based on which tasks are active."""
    if single_task == "relation":
        return metrics.get("relation_macro_f1", 0.0)
    if single_task == "evidence":
        return metrics.get("evidence_f1", 0.0)
    # Multi-task: average of both
    r = metrics.get("relation_macro_f1", 0.0)
    e = metrics.get("evidence_f1", 0.0)
    return (r + e) / 2


if __name__ == "__main__":
    main()
