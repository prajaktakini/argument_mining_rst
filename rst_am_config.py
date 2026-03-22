"""
config.py — all hyperparameters and paths in one place.
Edit this file before running any experiment.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ─── RST relation → argumentative edge type mapping ───────────────────────────
# Derived from Mann & Thompson (1988) and Peldszus & Stede (2013).
# Key: lowercased RST relation label from DMRST parser output
# Value: edge type index used in R-GCN weight matrices
RST_TO_EDGE = {
    # Supporting relations
    "evidence":     0,   # SUPPORTS  — most direct AM correlate
    "cause":        0,   # SUPPORTS
    "result":       0,   # SUPPORTS
    "justify":      0,   # SUPPORTS
    "purpose":      0,   # SUPPORTS
    # Attacking / opposing relations
    "contrast":     1,   # ATTACKS
    "antithesis":   1,   # ATTACKS
    # Conceding relations
    "concession":   2,   # CONCEDES
    # Elaborating (non-core AM, but keep as separate edge type)
    "elaboration":  3,   # ELABORATES
    "preparation":  3,
    "background":   3,
    # Summary / restatement
    "restatement":  4,   # SUMMARISES
    "summary":      4,
    # Conditional
    "condition":    5,   # CONDITIONAL
    # Temporal / structural (least argumentative — model should learn to ignore)
    "sequence":     6,   # SEQUENCE
    "joint":        6,
    "same-unit":    6,
    "list":         6,
}

EDGE_TYPE_NAMES = [
    "SUPPORTS",      # 0
    "ATTACKS",       # 1
    "CONCEDES",      # 2
    "ELABORATES",    # 3
    "SUMMARISES",    # 4
    "CONDITIONAL",   # 5
    "SEQUENCE",      # 6
]

NUM_EDGE_TYPES = len(EDGE_TYPE_NAMES)

# Nuclearity salience weights
NUCLEUS_WEIGHT   = 1.0
SATELLITE_WEIGHT = 0.6
MULTINUC_WEIGHT  = 0.8   # for Joint, List, Contrast (both nodes equal)


@dataclass
class DataConfig:
    # AAEC via HuggingFace (pie/aae2) — no manual download needed
    aaec_hf_name:    str = "pie/aae2"
    cache_dir:       str = "data/processed"

    # RST parser checkpoint (DMRST — download separately)
    rst_model:       str = "DMRST_Parser" 

    # Preprocessing
    max_doc_tokens:  int   = 512
    min_edu_tokens:  int   = 3
    overlap_thresh:  float = 0.5


@dataclass
class ModelConfig:
    # Encoder
    encoder_name:     str   = "roberta-base"
    encoder_hidden:   int   = 768
    freeze_encoder:   bool  = True   # unfreeze after warmup_epochs
    warmup_epochs:    int   = 3

    # R-GCN
    rgcn_hidden:      int   = 256
    rgcn_layers:      int   = 2
    rgcn_dropout:     float = 0.2
    num_edge_types:   int   = NUM_EDGE_TYPES
    use_nuclearity:   bool  = True   # apply salience weights in aggregation

    # Two task heads — both trained on AAEC
    # Head 0: relation identification  (3-way: support / attack / none)
    # Head 1: evidence detection        (binary: is this span a Premise?)
    head_hidden:      int   = 128
    task_weights:     List[float] = field(
        default_factory=lambda: [1.0, 1.0]
    )


@dataclass
class TrainConfig:
    seed:            int   = 42
    epochs:          int   = 30
    batch_size:      int   = 16        # graphs per batch
    lr_rgcn:         float = 1e-3
    lr_encoder:      float = 1e-4      # 10× smaller for fine-tuning encoder
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0

    # Evaluation
    eval_every:      int   = 1         # eval on dev set every N epochs
    patience:        int   = 5         # early stopping

    # Logging
    use_wandb:       bool  = False
    wandb_project:   str   = "rst-am"
    run_name:        Optional[str] = None

    # Ablations — set via CLI flags in train.py
    ablation_untyped_edges:   bool = False  # collapse all edge types → one
    ablation_no_nuclearity:   bool = False  # set all node weights to 1.0
    ablation_single_task:     bool = False  # train each head independently
    ablation_no_graph:        bool = False  # RoBERTa-only baseline (no R-GCN)


# ─── convenience accessor ────────────────────────────────────────────────────
def get_configs():
    return DataConfig(), ModelConfig(), TrainConfig()
