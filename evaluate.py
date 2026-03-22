"""
evaluate.py  (Tasks 0 + 1)

Task 0 — Relation identification:
    Macro-F1 across support / attack / none.
    Also reports per-class P/R/F1 so you can see if attack is being learned.

Task 1 — Evidence detection:
    Binary F1, Precision, Recall for the Premise (evidence) class.
    Also reports accuracy.
"""

import logging
from typing import Dict, List, Optional

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
)

from data.datasets import REL_NAMES

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, device: str = "cpu"):
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model,
        samples: list,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Run both task heads over all samples and return metrics dict.

        Returns keys:
            relation_macro_f1, relation_support_f1, relation_attack_f1,
            relation_none_f1, evidence_f1, evidence_precision,
            evidence_recall, evidence_accuracy
        """
        model.eval()

        rel_preds, rel_labels = [], []
        ev_preds,  ev_labels  = [], []

        for sample in samples:
            h = model.encode(sample).to(self.device)

            # ── Task 0: relations ──────────────────────────────────────────
            if sample.relations:
                pairs = torch.tensor(
                    [[r.src_id, r.tgt_id] for r in sample.relations],
                    dtype=torch.long, device=self.device,
                )
                # clamp to valid span range
                max_span = h.size(0) - 1
                pairs    = pairs.clamp(0, max_span)

                logits = model.forward_relation(h, pairs)
                preds  = logits.argmax(dim=-1).cpu().tolist()
                labels = [r.rel_type for r in sample.relations]

                rel_preds.extend(preds)
                rel_labels.extend(labels)

            # ── Task 1: evidence ───────────────────────────────────────────
            if sample.span_labels:
                # Map span → EDU node via edu_to_span (reverse lookup)
                # For spans that align to at least one EDU, use the
                # mean embedding of all aligned EDUs as the span embedding.
                span_embs = self._pool_span_embeddings(h, sample)

                if span_embs is not None:
                    logits  = model.forward_evidence(span_embs)
                    preds   = logits.argmax(dim=-1).cpu().tolist()
                    ev_preds.extend(preds)
                    ev_labels.extend(sample.span_labels)

        metrics: Dict[str, float] = {}

        # ── Relation metrics ───────────────────────────────────────────────
        if rel_preds:
            metrics["relation_macro_f1"] = float(f1_score(
                rel_labels, rel_preds, average="macro",
                labels=[0, 1, 2], zero_division=0,
            ))
            # Per-class
            per_class = f1_score(
                rel_labels, rel_preds, average=None,
                labels=[0, 1, 2], zero_division=0,
            )
            for i, name in enumerate(REL_NAMES):
                metrics[f"relation_{name}_f1"] = float(per_class[i])

            if verbose:
                logger.info(
                    "Relation classification report:\n%s",
                    classification_report(
                        rel_labels, rel_preds,
                        target_names=REL_NAMES, zero_division=0,
                    )
                )

        # ── Evidence metrics ───────────────────────────────────────────────
        if ev_preds:
            metrics["evidence_f1"] = float(f1_score(
                ev_labels, ev_preds, average="binary",
                pos_label=1, zero_division=0,
            ))
            metrics["evidence_precision"] = float(precision_score(
                ev_labels, ev_preds, average="binary",
                pos_label=1, zero_division=0,
            ))
            metrics["evidence_recall"] = float(recall_score(
                ev_labels, ev_preds, average="binary",
                pos_label=1, zero_division=0,
            ))
            metrics["evidence_accuracy"] = float(accuracy_score(
                ev_labels, ev_preds
            ))

            if verbose:
                logger.info(
                    "Evidence detection report:\n%s",
                    classification_report(
                        ev_labels, ev_preds,
                        target_names=["not-evidence", "evidence"],
                        zero_division=0,
                    )
                )

        return metrics

    # ── helper ────────────────────────────────────────────────────────────────

    def _pool_span_embeddings(self, h: torch.Tensor, sample) -> Optional[torch.Tensor]:
        """
        Build one embedding per annotated span by mean-pooling the embeddings
        of all EDUs that are aligned to that span.

        Falls back to a zero vector for spans with no aligned EDU.
        """
        n_spans = sample.n_spans
        if n_spans == 0:
            return None

        D     = h.size(-1)
        out   = torch.zeros(n_spans, D, device=self.device)
        count = torch.zeros(n_spans, 1, device=self.device)

        for edu_idx, span_idx in sample.edu_to_span.items():
            if span_idx is None or span_idx >= n_spans:
                continue
            if edu_idx >= h.size(0):
                continue
            out[span_idx]   += h[edu_idx]
            count[span_idx] += 1

        # For unaligned spans use the mean of all node embeddings as fallback
        fallback = h.mean(dim=0)
        unaligned = (count.squeeze(-1) == 0)
        out[unaligned] = fallback

        return out / count.clamp(min=1)


# ─── Ablation comparison table ────────────────────────────────────────────────

def compare_ablations(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print side-by-side comparison of ablation results.

    Args:
        results: OrderedDict of {run_name: metrics_dict}.
                 First entry is treated as the full model (reference).

    Example:
        compare_ablations({
            "full_model":     {"relation_macro_f1": 0.68, "evidence_f1": 0.71},
            "untyped_edges":  {"relation_macro_f1": 0.63, "evidence_f1": 0.69},
            "no_nuclearity":  {"relation_macro_f1": 0.65, "evidence_f1": 0.70},
            "no_graph":       {"relation_macro_f1": 0.58, "evidence_f1": 0.64},
        })
    """
    KEY_ORDER = [
        "relation_macro_f1",
        "relation_support_f1",
        "relation_attack_f1",
        "relation_none_f1",
        "evidence_f1",
        "evidence_precision",
        "evidence_recall",
    ]
    keys     = [k for k in KEY_ORDER if any(k in m for m in results.values())]
    col_w    = 12
    run_w    = max(len(r) for r in results) + 2
    header   = f"{'Run':<{run_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    sep      = "─" * len(header)
    ref_vals = list(results.values())[0]

    print(f"\n{sep}\n{header}\n{sep}")
    for i, (name, metrics) in enumerate(results.items()):
        row = f"{name:<{run_w}}"
        for k in keys:
            v    = metrics.get(k, float("nan"))
            diff = v - ref_vals.get(k, v)
            if i == 0 or abs(diff) < 1e-6:
                row += f"{v:>{col_w}.4f}"
            else:
                marker = f"({diff:+.3f})"
                row   += f"{v:>{col_w - len(marker)}.4f}{marker}"
        print(row)
    print(f"{sep}\n")
