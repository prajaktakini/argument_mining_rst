"""
models/graph_builder.py

Converts an RSTGraph + EDU-span alignment into a PyTorch Geometric HeteroData
object. This is the bridge between the RST parser output and the R-GCN model.

Node features: RoBERTa [CLS] embeddings of each EDU's text (computed lazily).
Edge indices:  one edge_index tensor per RST relation type (NUM_EDGE_TYPES).
Node weights:  scalar nuclearity salience per node.
"""

import torch
import logging
from typing import Dict, Optional, List

from rst_am_config import NUM_EDGE_TYPES, TrainConfig, NUCLEUS_WEIGHT, SATELLITE_WEIGHT, MULTINUC_WEIGHT

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a PyTorch Geometric HeteroData object from an RSTGraph.

    The graph has one node type ("edu") and NUM_EDGE_TYPES edge relation types.
    This heterogeneous structure allows the R-GCN to learn a separate weight
    matrix per RST relation type.

    Node features are placeholder zeros at build time — they are replaced with
    RoBERTa embeddings during the forward pass (see encoder.py).
    """

    def __init__(self, ablation_untyped: bool = False):
        """
        ablation_untyped: if True, collapse all edge types to a single type.
        This is used for the ablation in Section 6.3 of the proposal.
        """
        self.ablation_untyped = ablation_untyped
        self._edge_type_count = 1 if ablation_untyped else NUM_EDGE_TYPES

    def build(
        self,
        rst_graph,           # RSTGraph from rst_pipeline.py
        edu_to_span: Dict[int, Optional[int]],
        doc_text: str,
    ):
        """
        Returns a torch_geometric.data.HeteroData object.

        Graph structure:
          - Nodes: one per EDU
          - Node attrs:
              x            (N, 1)   placeholder — replaced in forward pass
              edu_texts    list[str] for tokenisation in encoder
              node_weight  (N,)     nuclearity salience
              span_ids     (N,)     aligned AM span id (-1 = unaligned)
          - Edge attrs (per edge type):
              edge_index   (2, E_t)  src/tgt node indices
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError(
                "torch-geometric not installed. "
                "Run: pip install torch-geometric"
            )

        n = rst_graph.n_edus
        if n == 0:
            return self._empty_graph()

        data = HeteroData()

        # ── Node features (placeholder) ───────────────────────────────────
        data["edu"].x           = torch.zeros(n, 1)   # replaced by encoder
        data["edu"].edu_texts   = [e.text for e in rst_graph.edus]
        data["edu"].node_weight = self._compute_node_weights(rst_graph)
        data["edu"].span_ids    = torch.tensor(
            [edu_to_span.get(i, -1) if edu_to_span.get(i) is not None else -1
             for i in range(n)],
            dtype=torch.long,
        )

        # ── Edge indices per relation type ────────────────────────────────
        # Initialise empty edge lists for every type
        edge_lists = [[] for _ in range(self._edge_type_count)]

        for edge in rst_graph.edges:
            src, tgt = edge.src, edge.tgt
            if src >= n or tgt >= n:
                continue
            etype = 0 if self.ablation_untyped else edge.edge_type_idx
            edge_lists[etype].append((src, tgt))

        for etype_idx, pairs in enumerate(edge_lists):
            rel_name = f"rel_{etype_idx}"
            if pairs:
                idx = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            else:
                idx = torch.zeros((2, 0), dtype=torch.long)
            data["edu", rel_name, "edu"].edge_index = idx

        # ── Add self-loops for isolated nodes ─────────────────────────────
        # Unaligned EDUs that have no edges get a self-loop of type SEQUENCE (6)
        # so they still receive a message in the R-GCN aggregation.
        isolated = self._find_isolated_nodes(rst_graph, n)
        if isolated:
            sl_etype = 0 if self.ablation_untyped else (NUM_EDGE_TYPES - 1)
            rel_name  = f"rel_{sl_etype}"
            sl_idx    = torch.tensor([[i, i] for i in isolated],
                                     dtype=torch.long).t().contiguous()
            existing  = data["edu", rel_name, "edu"].edge_index
            data["edu", rel_name, "edu"].edge_index = torch.cat(
                [existing, sl_idx], dim=1
            )

        # ── Document-level metadata ───────────────────────────────────────
        data.doc_text  = doc_text
        data.num_nodes = n

        return data

    # ── helpers ──────────────────────────────────────────────────────────────

    def _compute_node_weights(self, rst_graph) -> torch.Tensor:
        """
        Assign nuclearity salience weights to each EDU node.
        Nucleus = 1.0, Satellite = 0.6, Multinuc = 0.8.
        Nodes that appear as nucleus in ANY relation in the tree get 1.0.
        """
        from rst_am_config import NUCLEUS_WEIGHT, SATELLITE_WEIGHT, MULTINUC_WEIGHT

        weights = torch.full((rst_graph.n_edus,), SATELLITE_WEIGHT)

        for edge in rst_graph.edges:
            if edge.is_multinuc:
                weights[edge.src] = max(weights[edge.src].item(), MULTINUC_WEIGHT)
                weights[edge.tgt] = max(weights[edge.tgt].item(), MULTINUC_WEIGHT)
            else:
                # tgt is nucleus in mononuclear relations
                weights[edge.tgt] = NUCLEUS_WEIGHT
                weights[edge.src] = min(weights[edge.src].item(), SATELLITE_WEIGHT)

        return weights

    def _find_isolated_nodes(self, rst_graph, n: int) -> List[int]:
        """Return indices of EDU nodes with no incoming or outgoing edges."""
        connected = set()
        for edge in rst_graph.edges:
            connected.add(edge.src)
            connected.add(edge.tgt)
        return [i for i in range(n) if i not in connected]

    def _empty_graph(self):
        """Return a minimal valid HeteroData for empty documents."""
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["edu"].x = torch.zeros(1, 1)
        data["edu"].edu_texts = [""]
        data["edu"].node_weight = torch.ones(1)
        data["edu"].span_ids = torch.tensor([-1])
        for i in range(self._edge_type_count):
            data["edu", f"rel_{i}", "edu"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.num_nodes = 1
        return data
