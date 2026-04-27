"""
HelixLM Graph Builder & Executor.

Implements the heterogeneous directed acyclic graph (DAG) that forms the
computational backbone of HelixLM.  Nodes are wired with vertical
(feed-forward) and lateral (skip) edges, then executed in topological order.
"""
import random
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn

from .config import HelixConfig
from .nodes import (
    HeteroNode,
    LinearAttnNode,
    FullAttnNode,
    SwiGLUNode,
    GateNode,
    Mamba2Node,
    TitansMemoryNode,
)


class HelixGraph(nn.Module):
    """
    Heterogeneous DAG of neural columns.

    Each ``column`` is a set of nodes that execute in parallel (within the
    column).  Edges connect earlier columns to later ones (vertical) or
    nodes within the same column (lateral).  The graph is rebuilt from a
    random seed on every ``__init__`` call, but the wiring is frozen after
    construction.

    Args:
        cfg: HelixConfig instance.
    """
    def __init__(self, cfg: HelixConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)

        # Build node specifications
        self.node_spec: List[Tuple[str, Dict[str, Any]]] = []
        self._build_node_spec()

        # Instantiate nodes
        self.nodes: Dict[str, nn.Module] = nn.ModuleDict()
        self.node_meta: Dict[str, Tuple[int, str]] = {}  # name -> (column_index, node_type)
        for name, (ntype, kwargs) in self.node_spec:
            node = self._create_node(ntype, kwargs)
            self.nodes[name] = node
            # Infer column index from name (col{i}_...)
            col_idx = int(name.split("_")[0][3:])
            self.node_meta[name] = (col_idx, ntype)

        # Build wiring (predecessor lists)
        self.preds: Dict[str, List[str]] = {}
        self._build_wiring()

        # Topological order for execution
        self.order = self._topological_sort()

    # ------------------------------------------------------------------
    # Node specification builder
    # ------------------------------------------------------------------
    def _build_node_spec(self):
        """Generate the list of (node_type, kwargs) per column."""
        cfg = self.cfg
        has_titans = False

        for ci in range(cfg.n_columns):
            n_nodes = cfg.nodes_per_column[ci] if ci < len(cfg.nodes_per_column) else cfg.nodes_per_column[-1]
            column: List[Tuple[str, Dict[str, Any]]] = []

            # Attention node
            use_full_attn = cfg.attention_mode == "full"
            if cfg.attention_mode == "hybrid" and (ci % cfg.hybrid_full_attention_interval == 0):
                use_full_attn = True

            if use_full_attn:
                column.append(("full_attn", {
                    "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads,
                    "use_rope": cfg.use_rope,
                    "dropout": cfg.dropout,
                }))
            else:
                column.append(("linear_attn", {
                    "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads,
                    "use_rope": cfg.use_rope,
                    "dropout": cfg.dropout,
                }))

            # Feedforward node
            column.append(("swiglu", {
                "d_model": cfg.d_model,
                "dropout": cfg.dropout,
            }))

            # Optional Mamba-2 SSM
            if cfg.use_mamba2:
                column.append(("mamba2", {
                    "d_model": cfg.d_model,
                    "d_state": cfg.mamba2_d_state,
                    "d_conv": cfg.mamba2_d_conv,
                    "expand": cfg.mamba2_expand,
                    "dt_rank": cfg.mamba2_dt_rank,
                    "dropout": cfg.dropout,
                }))

            # Optional Titans Neural Memory — guaranteed at least once
            if cfg.use_titans_memory:
                if cfg.titans_always_select and ci == 0:
                    # Force one Titans node into column 0 so it is *always* selected
                    column.append(("titans", {
                        "d_model": cfg.d_model,
                        "feature_dim": cfg.titans_feature_dim,
                        "eta_init": cfg.titans_eta_init,
                        "n_heads": cfg.titans_n_heads,
                        "dropout": cfg.titans_dropout,
                    }))
                    has_titans = True
                elif self.rng.random() < 0.3:
                    # 30% chance for additional Titans nodes in later columns
                    column.append(("titans", {
                        "d_model": cfg.d_model,
                        "feature_dim": cfg.titans_feature_dim,
                        "eta_init": cfg.titans_eta_init,
                        "n_heads": cfg.titans_n_heads,
                        "dropout": cfg.titans_dropout,
                    }))

            # Gate / merge node if column has >1 compute node
            compute_nodes = [n for n, _ in column if n not in ("gate",)]
            if len(compute_nodes) > 1:
                column.append(("gate", {
                    "d_model": cfg.d_model,
                    "n_inputs": len(compute_nodes),
                    "dropout": cfg.dropout,
                }))

            # Register with unique names
            for ni, (ntype, kwargs) in enumerate(column):
                name = f"col{ci}_node{ni}_{ntype}"
                self.node_spec.append((name, (ntype, kwargs)))

        # Fallback: if titans is enabled but random wiring somehow excluded it,
        # inject one into the very last column as a dead-end failsafe.
        if cfg.use_titans_memory and cfg.titans_always_select and not has_titans:
            ci = cfg.n_columns - 1
            name = f"col{ci}_node99_titans"
            self.node_spec.append((name, ("titans", {
                "d_model": cfg.d_model,
                "feature_dim": cfg.titans_feature_dim,
                "eta_init": cfg.titans_eta_init,
                "n_heads": cfg.titans_n_heads,
                "dropout": cfg.titans_dropout,
            })))

    # ------------------------------------------------------------------
    # Node factory
    # ------------------------------------------------------------------
    def _create_node(self, ntype: str, kwargs: Dict[str, Any]) -> HeteroNode:
        if ntype == "linear_attn":
            return LinearAttnNode(**kwargs)
        elif ntype == "full_attn":
            return FullAttnNode(**kwargs)
        elif ntype == "swiglu":
            return SwiGLUNode(**kwargs)
        elif ntype == "gate":
            return GateNode(**kwargs)
        elif ntype == "mamba2":
            return Mamba2Node(**kwargs)
        elif ntype == "titans":
            return TitansMemoryNode(**kwargs)
        else:
            raise ValueError(f"Unknown node type: {ntype}")

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _build_wiring(self):
        """Build predecessor lists for each node."""
        cfg = self.cfg
        self.preds = {name: [] for name in self.nodes.keys()}

        for name, (col_idx, ntype) in self.node_meta.items():
            if col_idx == 0:
                # Root nodes have no predecessors
                continue

            # Vertical edges: connect to previous columns
            for pc in range(max(0, col_idx - cfg.vertical_depth), col_idx):
                above = [k for k, v in self.node_meta.items() if v[0] == pc]
                if above and self.rng.random() < cfg.vertical_p:
                    chosen = self.rng.sample(above, k=min(len(above), self.rng.randint(1, 2)))
                    self.preds[name].extend(chosen)

            # Lateral edges: connect to same column (skip connections)
            # Only wire from earlier nodes -> later nodes to avoid cycles.
            same_col = [k for k, v in self.node_meta.items() if v[0] == col_idx and k != name]
            # Filter to only nodes that come AFTER this node in the column order
            this_idx = int(name.split("_")[1][4:])  # node{N}_...
            later_nodes = [k for k in same_col if int(k.split("_")[1][4:]) > this_idx]
            if later_nodes and self.rng.random() < cfg.lateral_p:
                chosen = self.rng.sample(later_nodes, k=1)
                self.preds[name].extend(chosen)

            # Deduplicate
            self.preds[name] = list(dict.fromkeys(self.preds[name]))

        # Dead-end failsafe: if a node has no preds (and is not column 0), wire to previous column
        for name, (col_idx, ntype) in self.node_meta.items():
            if col_idx > 0 and not self.preds[name]:
                prev_col = col_idx - 1
                prev_nodes = [k for k, v in self.node_meta.items() if v[0] == prev_col]
                if prev_nodes:
                    self.preds[name].append(self.rng.choice(prev_nodes))

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------
    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm."""
        in_degree = {name: len(self.preds[name]) for name in self.nodes.keys()}
        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for succ, preds in self.preds.items():
                if node in preds:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        if len(order) != len(self.nodes):
            raise RuntimeError("Cycle detected in HelixGraph wiring!")
        return order

    # ------------------------------------------------------------------
    # Forward execution
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Dict[str, Any]] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute the graph in topological order.

        Args:
            x: Input tensor (B, T, D).
            states: Optional dict of persistent node states (e.g. SSM state,
                Titans memory tensors) from a previous chunk.
            freqs_cis: Precomputed RoPE frequencies.

        Returns:
            (output, new_states) where new_states maps node names to their
            updated state tensors (for nodes that maintain state).
        """
        if states is None:
            states = {}

        cache = {"freqs_cis": freqs_cis} if freqs_cis is not None else None
        activations: Dict[str, torch.Tensor] = {}
        new_states: Dict[str, Any] = {}

        for name in self.order:
            node = self.nodes[name]
            col_idx, ntype = self.node_meta[name]

            # Gather inputs
            pred_names = self.preds[name]
            if not pred_names:
                # Root node: feed x directly
                node_input = x
            else:
                # Merge predecessor activations by averaging
                preds_acts = [activations[p] for p in pred_names if p in activations]
                if not preds_acts:
                    node_input = x
                else:
                    node_input = torch.stack(preds_acts, dim=0).mean(dim=0)

            # Retrieve persistent state for this node if available
            node_state = states.get(name, None)

            # Forward
            out, new_state = node(node_input, state=node_state, cache=cache)

            # Store activation and updated state
            activations[name] = out
            if new_state is not None:
                new_states[name] = new_state

        # Sink: average last-column activations
        last_col = self.cfg.n_columns - 1
        sink_nodes = [n for n, (ci, _) in self.node_meta.items() if ci == last_col]
        if sink_nodes:
            output = torch.stack([activations[n] for n in sink_nodes], dim=0).mean(dim=0)
        else:
            output = x

        return output, new_states
