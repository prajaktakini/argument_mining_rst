"""
models/multitask.py  (Tasks 0 + 1 only)

Architecture:
  RoBERTa encoder  →  R-GCN (typed RST edges, nuclearity weights)
                    →  Head 0: relation identification  (3-class)
                    →  Head 1: evidence detection        (binary)

Ablation flags (set in ModelConfig / CLI):
  ablation_untyped_edges  — all RST edge types share one weight matrix
  ablation_no_nuclearity  — node weights fixed to 1.0
  ablation_no_graph       — skip R-GCN entirely (RoBERTa-only baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from rst_am_config import ModelConfig, NUM_EDGE_TYPES


# ─── Relational GCN layer ─────────────────────────────────────────────────────

class RGCNLayer(nn.Module):
    """
    One R-GCN layer with per-relation weight matrices.

    Node update rule:
        h_i' = LayerNorm( ReLU( Σ_r  Σ_{j∈N_r(i)}  w_j * W_r * h_j
                                + W_self * h_i ) )

    w_j = nuclearity salience weight of node j (1.0 / 0.6 / 0.8).
    When ablation_no_nuclearity is set, w_j = 1.0 for all nodes.
    """

    def __init__(
        self,
        in_dim:         int,
        out_dim:        int,
        num_relations:  int,
        dropout:        float = 0.2,
        use_nuclearity: bool  = True,
    ):
        super().__init__()
        self.num_relations  = num_relations
        self.use_nuclearity = use_nuclearity

        # One weight matrix per RST relation type
        self.W_rel  = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])
        # Self-loop weight matrix
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.drop   = nn.Dropout(dropout)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(
        self,
        x:            torch.Tensor,            # (N, in_dim)
        edge_indices: Dict[int, torch.Tensor], # etype_idx → (2, E)
        node_weights: Optional[torch.Tensor],  # (N,) nuclearity salience
    ) -> torch.Tensor:                         # (N, out_dim)

        N   = x.size(0)
        out = self.W_self(x)   # self-loop

        for etype in range(self.num_relations):
            ei = edge_indices.get(etype)
            if ei is None or ei.size(1) == 0:
                continue

            src, tgt = ei[0].to(x.device), ei[1].to(x.device)
            feats    = x[src]                              # (E, in_dim)

            # Nuclearity salience weighting
            if self.use_nuclearity and node_weights is not None:
                feats = feats * node_weights[src].unsqueeze(-1)

            msg = self.W_rel[etype](feats)                 # (E, out_dim)

            # Scatter-mean aggregation into target nodes
            agg   = torch.zeros(N, msg.size(-1), device=x.device)
            count = torch.zeros(N, 1,       device=x.device)
            agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
            count.scatter_add_(0, tgt.unsqueeze(-1),
                               torch.ones(src.size(0), 1, device=x.device))
            out = out + agg / count.clamp(min=1)

        return self.drop(self.norm(F.relu(out)))


# ─── Task head ────────────────────────────────────────────────────────────────

class TaskHead(nn.Module):
    """2-layer MLP for one AM subtask."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Full model ───────────────────────────────────────────────────────────────

class RSTArgumentMiner(nn.Module):
    """
    Two-task RST argument mining model.

    Forward pass per document:
      1. Encode EDU texts with RoBERTa → (N, 768) node features
      2. Run R-GCN over typed RST graph → (N, 256) graph embeddings
      3. Task 0 head: concatenate (src, tgt) embeddings → 3-class logits
      4. Task 1 head: per-node → 2-class logits (Premise or not)

    Ablation modes (set via ModelConfig flags):
      ablation_no_graph     — skip R-GCN; project RoBERTa output directly
      ablation_untyped_edges — all edges share one R-GCN weight matrix
      ablation_no_nuclearity — uniform node weights (w=1.0)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        from models.encoder import SpanEncoder
        self.encoder = SpanEncoder(
            model_name=cfg.encoder_name,
            freeze=cfg.freeze_encoder,
        )

        # ── R-GCN ────────────────────────────────────────────────────────
        self.use_graph = not getattr(cfg, "ablation_no_graph", False)
        n_rel = (1 if getattr(cfg, "ablation_untyped_edges", False)
                 else cfg.num_edge_types)

        if self.use_graph:
            layers, in_d = [], cfg.encoder_hidden
            for _ in range(cfg.rgcn_layers):
                layers.append(RGCNLayer(
                    in_dim=in_d,
                    out_dim=cfg.rgcn_hidden,
                    num_relations=n_rel,
                    dropout=cfg.rgcn_dropout,
                    use_nuclearity=not getattr(cfg, "ablation_no_nuclearity", False),
                ))
                in_d = cfg.rgcn_hidden
            self.rgcn = nn.ModuleList(layers)
            head_in = cfg.rgcn_hidden
        else:
            self.proj = nn.Linear(cfg.encoder_hidden, cfg.rgcn_hidden)
            head_in   = cfg.rgcn_hidden

        # ── Task heads ────────────────────────────────────────────────────
        # Head 0 — relation identification:
        #   input = concat(src_emb, tgt_emb) → 3 classes (support/attack/none)
        self.head_relation = TaskHead(head_in * 2, cfg.head_hidden, 3)

        # Head 1 — evidence detection:
        #   input = single span embedding → 2 classes (premise / not-premise)
        self.head_evidence = TaskHead(head_in, cfg.head_hidden, 2)

        self.task_weights = cfg.task_weights   # [w_rel, w_ev]

    # ── forward ──────────────────────────────────────────────────────────────

    def encode(self, sample) -> torch.Tensor:
        """
        Encode one AMGraphSample and return R-GCN node embeddings (N, D).
        Call this inside the training loop per sample.
        """
        data       = sample.graph_data
        edu_texts  = data["edu"].edu_texts
        node_feats = self.encoder(edu_texts)          # (N, 768)

        if self.use_graph:
            edge_indices = self._extract_edge_indices(data)
            node_weights = data["edu"].node_weight.to(node_feats.device)
            h = node_feats
            for layer in self.rgcn:
                h = layer(h, edge_indices, node_weights)
        else:
            h = F.relu(self.proj(node_feats))

        return h                                       # (N, D)

    def forward_relation(
        self,
        h:          torch.Tensor,   # (N, D) — node embeddings for this doc
        span_pairs: torch.Tensor,   # (P, 2) — (src_idx, tgt_idx) pairs
    ) -> torch.Tensor:              # (P, 3)
        """Task 0: Argument relation identification."""
        src = h[span_pairs[:, 0]]
        tgt = h[span_pairs[:, 1]]
        return self.head_relation(torch.cat([src, tgt], dim=-1))

    def forward_evidence(self, h: torch.Tensor) -> torch.Tensor:
        """Task 1: Evidence detection — one logit pair per node."""
        return self.head_evidence(h)                   # (N, 2)

    # ── loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        logits_rel: Optional[torch.Tensor],  # (P, 3)
        labels_rel: Optional[torch.Tensor],  # (P,)
        logits_ev:  Optional[torch.Tensor],  # (N, 2)
        labels_ev:  Optional[torch.Tensor],  # (N,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Weighted sum of both task losses.
        Returns (total_loss, {"relation": float, "evidence": float}).
        """
        losses: Dict[str, float] = {}
        total = torch.tensor(0.0, requires_grad=True)

        if logits_rel is not None and labels_rel is not None:
            l = F.cross_entropy(logits_rel, labels_rel)
            losses["relation"] = l.item()
            total = total + self.task_weights[0] * l

        if logits_ev is not None and labels_ev is not None:
            l = F.cross_entropy(logits_ev, labels_ev)
            losses["evidence"] = l.item()
            total = total + self.task_weights[1] * l

        return total, losses

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract_edge_indices(self, data) -> Dict[int, torch.Tensor]:
        n_rel = (1 if getattr(self.cfg, "ablation_untyped_edges", False)
                 else self.cfg.num_edge_types)
        ei = {}
        for etype in range(n_rel):
            key = ("edu", f"rel_{etype}", "edu")
            if key in data.edge_index_dict:
                ei[etype] = data[key].edge_index
        return ei
