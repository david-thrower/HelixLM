"""
HelixGraph: Biological brain-inspired heterogeneous graph executor.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HelixConfig
from .nodes import (
    HeteroNode, LinearAttnNode, FullAttnNode, DenseNode,
    SwiGLUNode, SSMNode, Mamba2Node, GateNode
)


class HelixGraph(nn.Module):
    """
    Randomly wired heterogeneous graph of neural nodes.
    
    - Topology: Biological-style neural columns with vertical and lateral connections
    - Aggregation: learned per-node merge (Linear bottleneck) or Gate
    - Stateful nodes (SSM/Mamba-2) expose state read/write across loops
    """
    def __init__(self, cfg: HelixConfig, seed: int = 42):
        super().__init__()
        self.cfg = cfg
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self.node_spec = self._build_node_spec()
        self.nodes = nn.ModuleDict()
        self.node_meta: Dict[str, Tuple[int, int, str]] = {}

        nid = 0
        for ci, column in enumerate(self.node_spec):
            for ni, (ntype, ncfg) in enumerate(column):
                name = f"n{nid}"
                self.node_meta[name] = (ci, ni, ntype)
                self.nodes[name] = self._create_node(ntype, ncfg)
                nid += 1

        # Build random wiring
        self.graph: Dict[str, List[str]] = {}
        names = list(self.nodes.keys())

        for name in names:
            ci, idx, ntype = self.node_meta[name]
            preds: List[str] = []

            # Vertical connections
            if ci > 0:
                for pc in range(max(0, ci - cfg.vertical_depth), ci):
                    above = [k for k, v in self.node_meta.items() if v[0] == pc]
                    if above and rng.rand() < cfg.vertical_p:
                        n_pick = rng.randint(1, len(above) + 1)
                        chosen = rng.choice(above, size=min(n_pick, len(above)), replace=False)
                        preds.extend(chosen.tolist())

            # Lateral connections
            same_column = [k for k, v in self.node_meta.items() if v[0] == ci and v[1] < idx]
            for s in same_column:
                if rng.rand() < cfg.lateral_p:
                    preds.append(s)

            # Dead-end failsafe
            if not preds and ci > 0:
                above = [k for k, v in self.node_meta.items() if v[0] == ci - 1]
                if above:
                    preds.append(rng.choice(above))

            self.graph[name] = preds

        # Merge layers for multi-predecessor non-gate nodes
        self.merges = nn.ModuleDict()
        for name, preds in self.graph.items():
            if len(preds) > 1 and self.node_meta[name][2] != "gate":
                self.merges[name] = nn.Linear(len(preds) * cfg.d_model, cfg.d_model)

        self.order = self._topsort()
        self.root_nodes = [n for n in names if len(self.graph[n]) == 0 or self.node_meta[n][0] == 0]
        last_col = max(v[0] for v in self.node_meta.values())
        self.sink_nodes = [k for k, v in self.node_meta.items() if v[0] == last_col]

    def _build_node_spec(self) -> List[List[Tuple[str, dict]]]:
        cfg = self.cfg
        spec = []
        for ci in range(cfg.n_columns):
            column = []
            use_full_attn = False
            if cfg.attention_mode == "full":
                use_full_attn = True
            elif cfg.attention_mode == "hybrid":
                use_full_attn = (ci % cfg.hybrid_full_attention_interval == 0)

            if use_full_attn:
                column.append(("full_attn", {
                    "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                    "dropout": cfg.dropout, "use_rope": cfg.use_rope,
                }))
            else:
                column.append(("linear_attn", {
                    "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                    "feature_dim": cfg.linear_feature_dim, "dropout": cfg.dropout,
                }))

            column.append(("swiglu", {
                "d_model": cfg.d_model, "expansion": cfg.ffn_expansion, "dropout": cfg.dropout,
            }))

            if cfg.use_ssm:
                if hasattr(cfg, 'ssm_d_state') and cfg.ssm_d_state >= 64:
                    # Use Mamba-2 for larger state dimensions
                    column.append(("mamba2", {
                        "d_model": cfg.d_model, "d_state": cfg.ssm_d_state,
                        "d_conv": cfg.ssm_d_conv, "expand": cfg.ssm_expand,
                        "dt_rank": cfg.ssm_dt_rank if hasattr(cfg, 'ssm_dt_rank') else "auto",
                        "conv_bias": cfg.ssm_conv_bias if hasattr(cfg, 'ssm_conv_bias') else True,
                        "bias": cfg.ssm_bias if hasattr(cfg, 'ssm_bias') else False,
                        "dropout": cfg.dropout,
                    }))
                else:
                    column.append(("ssm", {
                        "d_model": cfg.d_model, "d_state": cfg.ssm_d_state,
                        "d_conv": cfg.ssm_d_conv, "expand": cfg.ssm_expand, "dropout": cfg.dropout,
                    }))

            if len(column) > 1 or ci > 0:
                column.append(("gate", {
                    "d_model": cfg.d_model, "n_preds": len(column), "dropout": cfg.dropout,
                }))

            spec.append(column)
        return spec

    def _create_node(self, ntype: str, ncfg: dict) -> HeteroNode:
        if ntype == "linear_attn":
            return LinearAttnNode(**ncfg)
        elif ntype == "full_attn":
            return FullAttnNode(**ncfg)
        elif ntype == "dense":
            return DenseNode(**ncfg)
        elif ntype == "swiglu":
            return SwiGLUNode(**ncfg)
        elif ntype == "ssm":
            return SSMNode(**ncfg)
        elif ntype == "mamba2":
            return Mamba2Node(**ncfg)
        elif ntype == "gate":
            return GateNode(**ncfg)
        raise ValueError(f"Unknown node type: {ntype}")

    def _topsort(self) -> List[str]:
        in_deg = {n: 0 for n in self.nodes}
        adj = {n: [] for n in self.nodes}
        for n, preds in self.graph.items():
            for p in preds:
                adj[p].append(n)
                in_deg[n] += 1
        queue = [n for n, d in in_deg.items() if d == 0]
        out = []
        while queue:
            cur = queue.pop(0)
            out.append(cur)
            for nxt in adj[cur]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        if len(out) != len(self.nodes):
            remaining = [n for n in self.nodes if n not in out]
            raise ValueError(f"Cycle detected! Remaining: {remaining}")
        return out

    def forward(self, x: torch.Tensor, states: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if states is None:
            states = {}
        new_states = {}
        cache: Dict[str, torch.Tensor] = {}

        for name in self.nodes:
            if not self.graph[name]:
                cache[name] = x

        for name in self.order:
            if name in cache:
                continue
            preds = self.graph[name]
            feats = [cache[p] for p in preds]
            _, _, ntype = self.node_meta[name]

            if len(feats) == 1:
                merged = feats[0]
            elif ntype == "gate":
                merged = feats
            else:
                merged = self.merges[name](torch.cat(feats, dim=-1))

            node = self.nodes[name]
            if isinstance(node, (SSMNode, Mamba2Node)):
                out, s = node(merged, state=states.get(name))
                new_states[name] = s
            elif isinstance(node, GateNode):
                out, _ = node(merged)
            else:
                out, _ = node(merged)
            cache[name] = out

        if len(self.sink_nodes) == 1:
            out = cache[self.sink_nodes[0]]
        else:
            out = torch.stack([cache[s] for s in self.sink_nodes], dim=-1).mean(dim=-1)

        return out + x, new_states

    def get_graph_info(self) -> Dict[str, Any]:
        info = {
            "n_nodes": len(self.nodes),
            "n_columns": self.cfg.n_columns,
            "node_types": {},
            "n_edges": sum(len(p) for p in self.graph.values()),
            "roots": self.root_nodes,
            "sinks": self.sink_nodes,
        }
        for name, (ci, idx, ntype) in self.node_meta.items():
            info["node_types"][ntype] = info["node_types"].get(ntype, 0) + 1
        return info
