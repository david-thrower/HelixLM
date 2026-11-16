"""
Heterogeneous neural nodes for HelixLM.

Each node type implements a different computational mechanism inspired by
diverse neural structures in biological brains.

REMEDIATION (2026-08-22):
  ErrorCorrectingMultiScaleAttnNode rewritten to fix:
    - Issue 1: parameterized strict NaN detection (default non-raising)
    - Issue 2: causal prefix compression + causal compressed attention + causal expansion
    - Issue 3: mask-aware pooling (torch.where, valid-token sums/counts)
    - Issue 6: removed output_ffn (graph owns FFN via SwiGLU)
  Also: corrector_type="attn" now uses causal + padding masks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


def apply_rotary(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """RoPE for [N, H, L, Dh] using per-token absolute positions [N, L].

    Rotating q and k by their absolute position makes attention depend on the
    relative offset only, which is what the architecture's README claims and
    what was previously missing: freqs_cis was computed and never used.
    """
    _, _, _, dh = x.shape
    if dh % 2 != 0:
        return x
    half = dh // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    angles = positions.to(torch.float32).unsqueeze(-1) * freqs.view(1, 1, -1)
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    # Rotate in fp32 for accuracy, then return in the caller's dtype: returning
    # fp32 here promoted q/k (and the scores) to fp32 under pure-bf16 weights,
    # and the attn @ v matmul then failed on mixed dtypes.
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        norm = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = self.weight.to(x_f.dtype)
        return (x_f * norm * weight).to(dtype)


class HeteroNode(nn.Module):
    """Base class for all heterogeneous nodes."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


class ErrorCorrectingMultiScaleAttnNode(HeteroNode):
    """
    Three-path attention with learned consensus and logical error correction.

    Architecture (causal, mask-aware, multi-view):
      - Local causal window attention   -> [B, T, D]
      - Coarse causal window attention   -> [B, T, D]
      - Compressed branch:
          V independent views, each:
            down-proj -> causal prefix compress (K positions) ->
            causal compressed attention -> causal expand -> up-proj
          internal consensus across V views -> [B, T, D]
      - External tokenwise consensus across (local, coarse, compressed)
      - Gated causal correction
      - Gated residual merge with original input
      (NO output FFN -- the graph owns FFN via SwiGLU)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window: int = 64,
        coarse_window: int = 128,
        compressed_windows: int = 8,          # K: temporal compressed positions
        compressed_views: int = 8,            # V: independent compressed kernels
        compressed_dim: Optional[int] = None, # Dc: bottleneck width
        compressed_heads: Optional[int] = None,
        corrector_dim: Optional[int] = None,
        consensus_type: str = "cosine",
        corrector_type: str = "ffn",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        strict_nan_check: bool = False,       # Issue 1: raise on corrupt NaN
        sliding_windows: bool = False,        # C7: causal sliding local/coarse windows
        compressed_stride: Optional[int] = None,  # C7: fixed-stride compressed boundaries
        use_rope: bool = False,               # apply RoPE to q/k in every attention path
        rope_theta: float = 10000.0,
        # Deprecated: kept for config compatibility, ignored.
        output_ffn_dim: Optional[int] = None,
    ):
        super().__init__(d_model)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if local_window <= 0 or coarse_window <= 0:
            raise ValueError("local_window and coarse_window must be positive")
        if compressed_windows <= 0:
            raise ValueError("compressed_windows (K) must be positive")
        if compressed_views <= 0:
            raise ValueError("compressed_views (V) must be positive")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.local_window = local_window
        self.coarse_window = coarse_window
        self.compressed_windows = compressed_windows   # K
        self.compressed_views = compressed_views       # V

        Dc = compressed_dim if compressed_dim is not None else d_model // 2
        if Dc <= 0:
            raise ValueError("compressed_dim (Dc) must be positive")
        c_heads = compressed_heads if compressed_heads is not None else max(1, n_heads // 2)
        if Dc % c_heads != 0:
            c_heads = 1
        self.compressed_dim = Dc
        self.compressed_heads = c_heads
        self.compressed_head_dim = Dc // c_heads

        # Issue 1: strict NaN detection is opt-in (default: non-raising, training-stable)
        self.strict_nan_check = bool(strict_nan_check)
        # C7 (both default off: no new parameters, identical behaviour).
        self.sliding_windows = bool(sliding_windows)
        if compressed_stride is not None and int(compressed_stride) <= 0:
            raise ValueError("compressed_stride must be positive")
        self.compressed_stride = int(compressed_stride) if compressed_stride is not None else None
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)

        # Layer 1: shared QKV + output projection for local/coarse windowed attention
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Compressed branch: V independent views
        self.view_down = nn.ModuleList([nn.Linear(d_model, Dc) for _ in range(compressed_views)])
        self.view_qkv = nn.ModuleList([nn.Linear(Dc, 3 * Dc) for _ in range(compressed_views)])
        self.view_out = nn.ModuleList([nn.Linear(Dc, Dc) for _ in range(compressed_views)])
        self.view_up = nn.ModuleList([nn.Linear(Dc, d_model) for _ in range(compressed_views)])

        # Layer 2: external consensus across (local, coarse, compressed)
        self.consensus_type = consensus_type.lower()
        if self.consensus_type == "mha":
            self.cross_view_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=attn_dropout, batch_first=True
            )
            self.cross_view_norm = RMSNorm(d_model)
        elif self.consensus_type == "cosine":
            self.consensus_temp = nn.Parameter(torch.ones(1))
            self.consensus_norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown consensus_type: {consensus_type}")

        # Internal compressed-view consensus (tokenwise cosine across V views)
        self.view_consensus_temp = nn.Parameter(torch.ones(1))
        self.view_consensus_norm = RMSNorm(d_model)

        # Layer 3: correction
        self.corrector_type = corrector_type.lower()
        corrector_hidden = corrector_dim if corrector_dim is not None else d_model // 2

        if self.corrector_type == "ffn":
            self.corrector = nn.Sequential(
                nn.Linear(d_model * 2, corrector_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(corrector_hidden, d_model),
            )
        elif self.corrector_type == "attn":
            self.corrector_qkv = nn.Linear(d_model, 3 * d_model)
            self.corrector_out = nn.Linear(d_model, d_model)
            self.corrector_ffn = nn.Sequential(
                nn.Linear(d_model, corrector_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(corrector_hidden, d_model),
            )
            self.corrector_norm1 = RMSNorm(d_model)
            self.corrector_norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown corrector_type: {corrector_type}")

        self.corrector_gate = nn.Linear(d_model * 2, 1)

        # Issue 6: output FFN REMOVED. The graph appends SwiGLU after this node.
        # (output_ffn_dim is accepted for backward config compatibility but unused.)

        # Final residual gate
        self.out_gate = nn.Linear(d_model * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout > 0 else dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Safe causal masked attention (shared by windowed, compressed, corrector)
    # ------------------------------------------------------------------
    def _project_qkv(self, x: torch.Tensor, qkv_proj: nn.Linear, n_heads: int, head_dim: int,
                     positions: Optional[torch.Tensor] = None):
        N, L, _ = x.shape
        qkv = qkv_proj(x).view(N, L, 3, n_heads, head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2).contiguous()  # [N, H, L, Dh]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        if positions is not None:
            q = apply_rotary(q, positions, self.rope_theta)
            k = apply_rotary(k, positions, self.rope_theta)
        return q, k, v

    def _safe_attention(self, q, k, v, causal_allowed, key_valid, query_valid):
        """
        Causal masked attention with explicit all-masked-row handling.

        q, k, v:            [N, H, L, Dh]
        causal_allowed:     [L, L] bool (lower-triangular)
        key_valid:           [N, L] bool
        query_valid:         [N, L] bool

        Returns: [N, H, L, Dh]

        Issue 1 policy:
          - Structurally empty rows (no lawful keys) -> zero output (always).
          - NaN/Inf in rows that DO contain lawful keys:
              strict_nan_check=True  -> raise FloatingPointError
              strict_nan_check=False -> zero the non-finite weights (training-stable)
        """
        N, H, L, Dh = q.shape
        scale = Dh ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        allowed = (
            causal_allowed.unsqueeze(0).unsqueeze(0)
            & key_valid[:, None, None, :]
            & query_valid[:, None, :, None]
        )
        allowed = allowed.expand(N, H, L, L)
        row_has_key = allowed.any(dim=-1, keepdim=True)  # [N, H, L, 1]

        scores = scores.masked_fill(~allowed, float("-inf"))
        # Only structurally empty rows receive a finite placeholder before softmax.
        safe_scores = torch.where(row_has_key, scores, torch.zeros_like(scores))

        # Softmax in fp32 for AMP/BF16 stability.
        attn = torch.softmax(safe_scores.float(), dim=-1).to(scores.dtype)

        if self.strict_nan_check:
            valid_attn = attn.masked_select(row_has_key.expand_as(attn))
            if valid_attn.numel() > 0 and not torch.isfinite(valid_attn).all().item():
                raise FloatingPointError(
                    "non-finite attention on a row containing lawful keys")
        else:
            # Non-strict: zero any non-finite weights so training does not crash.
            attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))

        # Zero structurally empty rows.
        attn = torch.where(row_has_key, attn, torch.zeros_like(attn))
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn.to(v.dtype), v)  # [N, H, L, Dh]  dtype-safe: attn is fp32 by design (softmax stability), v follows the model dtype
        return out

    def _check_finite_output(self, out, valid_mask, label: str):
        """Issue 1: optional finite-output check on valid query positions."""
        if self.strict_nan_check:
            valid_out = out.masked_select(valid_mask.unsqueeze(-1).expand_as(out))
            if valid_out.numel() > 0 and not torch.isfinite(valid_out).all().item():
                raise FloatingPointError(f"non-finite output on a valid query position ({label})")
        else:
            return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

    # ------------------------------------------------------------------
    # Windowed attention (local + coarse)
    # ------------------------------------------------------------------
    def _pad_to_window(self, x: torch.Tensor, win: int):
        B, T, D = x.shape
        pad_len = (win - T % win) % win
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        return x, pad_len

    def _windowed_attn(self, x: torch.Tensor, win: int, attention_mask: Optional[torch.Tensor] = None):
        if self.sliding_windows:
            return self._sliding_windowed_attn(x, win, attention_mask=attention_mask)
        B, T, D = x.shape
        x_pad, pad_len = self._pad_to_window(x, win)
        T_pad = T + pad_len
        nwin = T_pad // win

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool, device=x.device)
        else:
            valid = torch.ones((B, T), dtype=torch.bool, device=x.device)
        if pad_len:
            valid = F.pad(valid, (0, pad_len), value=False)
        mask_win = valid.reshape(B, nwin, win).reshape(B * nwin, win)

        # NaN-safe: use torch.where, never multiplication, for masked positions.
        xw = x_pad.reshape(B, nwin, win, D).reshape(B * nwin, win, D)
        xw = torch.where(mask_win.unsqueeze(-1), xw, torch.zeros_like(xw))

        if self.use_rope:
            starts = torch.arange(nwin, device=x.device) * win
            pos = starts.view(1, nwin, 1) + torch.arange(win, device=x.device).view(1, 1, win)
            pos = pos.expand(B, nwin, win).reshape(B * nwin, win)
        else:
            pos = None
        q, k, v = self._project_qkv(xw, self.qkv_proj, self.n_heads, self.head_dim,
                                    positions=pos)
        causal = torch.tril(torch.ones(win, win, device=x.device, dtype=torch.bool))

        out = self._safe_attention(q, k, v, causal, mask_win, mask_win)  # [B*nwin, H, win, Dh]
        out = out.transpose(1, 2).reshape(B * nwin, win, D)
        out = self.out_proj(out)

        # Suppress projection bias at padded queries.
        out = torch.where(mask_win.unsqueeze(-1), out, torch.zeros_like(out))
        out = self._check_finite_output(out, mask_win, "windowed")

        out = out.view(B, nwin, win, D).reshape(B, T_pad, D)
        return out[:, :T, :]

    def _sliding_windowed_attn(self, x: torch.Tensor, win: int,
                               attention_mask: Optional[torch.Tensor] = None):
        """C7: exact causal sliding-window attention.

        Query t attends to keys max(0, t-win+1)..t (never a block-aligned
        window), so every token sees its previous win-1 tokens and the output
        at t does not depend on where t falls relative to a window boundary or
        on the total length T.

        Blocked form: pad T to a multiple of win; block i's queries attend to
        the 2*win keys of blocks i-1 and i through a fixed band mask. The
        existing _safe_attention is reused (same NaN/empty-row policy), with the
        previous block's rows passed as invalid queries and sliced off, so the
        score tensor is [2*win, 2*win] per block (4x the block-window cost,
        still O(T*win)). RoPE uses absolute positions.
        """
        B, T, D = x.shape
        x_pad, pad_len = self._pad_to_window(x, win)
        T_pad = T + pad_len
        nwin = T_pad // win

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool, device=x.device)
        else:
            valid = torch.ones((B, T), dtype=torch.bool, device=x.device)
        if pad_len:
            valid = F.pad(valid, (0, pad_len), value=False)
        x_pad = torch.where(valid.unsqueeze(-1), x_pad, torch.zeros_like(x_pad))

        pos = (torch.arange(T_pad, device=x.device).unsqueeze(0).expand(B, T_pad)
               if self.use_rope else None)
        q, k, v = self._project_qkv(x_pad, self.qkv_proj, self.n_heads, self.head_dim,
                                    positions=pos)              # [B, H, T_pad, Dh]
        H, Dh = self.n_heads, self.head_dim

        def blocks(t):  # [B, H, T_pad, Dh] -> [B*nwin, H, 2*win, Dh] (prev block ++ block)
            t = t.view(B, H, nwin, win, Dh).permute(0, 2, 1, 3, 4)       # [B, nwin, H, win, Dh]
            prev = torch.cat([torch.zeros_like(t[:, :1]), t[:, :-1]], dim=1)
            return torch.cat([prev, t], dim=3).reshape(B * nwin, H, 2 * win, Dh)

        qb, kb, vb = blocks(q), blocks(k), blocks(v)
        vw = valid.view(B, nwin, win)
        vprev = torch.cat([torch.zeros_like(vw[:, :1]), vw[:, :-1]], dim=1)
        key_valid = torch.cat([vprev, vw], dim=2).reshape(B * nwin, 2 * win)
        query_valid = torch.cat([torch.zeros_like(vw), vw], dim=2).reshape(B * nwin, 2 * win)

        r = torch.arange(2 * win, device=x.device)
        band = (r.view(-1, 1) >= r.view(1, -1)) & (r.view(-1, 1) - r.view(1, -1) < win)

        out = self._safe_attention(qb, kb, vb, band, key_valid, query_valid)
        out = out[:, :, win:, :]                                      # this block's queries
        out = out.transpose(1, 2).reshape(B * nwin, win, D)
        out = self.out_proj(out)
        mask_win = vw.reshape(B * nwin, win)
        out = torch.where(mask_win.unsqueeze(-1), out, torch.zeros_like(out))
        out = self._check_finite_output(out, mask_win, "windowed")
        return out.view(B, nwin, win, D).reshape(B, T_pad, D)[:, :T, :]

    def _fixed_stride_boundaries(self, T: int, device) -> torch.Tensor:
        """C7: boundaries 0, S, 2S, ... (< T) with S = compressed_stride.

        Independent of T, so a prefix sees exactly the first boundaries of a
        longer input. Starting at 0 means every token has a boundary <= t, so
        no token is ever mapped to a summary that contains future tokens. The
        number of compressed positions is ceil(T / S) (= K at T = seq_len).
        """
        return torch.arange(0, T, self.compressed_stride, device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Causal compressed branch (Issues 2 + 3)
    # ------------------------------------------------------------------
    def _causal_boundaries(self, T: int, K: int, device) -> torch.Tensor:
        """Monotonic source boundaries in [0, T-1], inclusive of both ends."""
        if K <= 1:
            return torch.tensor([T - 1], device=device, dtype=torch.long)
        boundaries = torch.linspace(0, T - 1, K, device=device).long()
        return boundaries.clamp(0, T - 1)

    def _causal_prefix_compress(self, z: torch.Tensor, valid: torch.Tensor, boundaries: torch.Tensor):
        """
        Mask-aware causal prefix compression.

        Z_k = sum(valid[0:m_k] * z[0:m_k]) / max(1, sum(valid[0:m_k]))

        z:        [B, T, Dc]
        valid:    [B, T] bool
        boundaries: [K] long (sorted, in [0, T-1])

        Returns: (z_comp [B, K, Dc], comp_valid [B, K] bool)
        """
        B, T, Dc = z.shape
        K = boundaries.shape[0]

        # Issue 3: torch.where (not multiply) so NaN*0 cannot survive.
        z_clean = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))
        valid_f = valid.unsqueeze(-1).float()

        # FP32 accumulation and division for stability.
        cum_z = torch.cumsum(z_clean.float(), dim=1)       # [B, T, Dc]
        cum_count = torch.cumsum(valid_f, dim=1)            # [B, T, 1]

        idx = boundaries.view(1, K, 1).expand(B, K, Dc)
        z_comp_fp32 = cum_z.gather(1, idx)                 # [B, K, Dc]

        count_idx = boundaries.view(1, K, 1).expand(B, K, 1)
        counts = cum_count.gather(1, count_idx).clamp(min=1.0)  # [B, K, 1]
        z_comp = (z_comp_fp32 / counts).to(z.dtype)

        comp_valid = (cum_count.gather(1, count_idx).squeeze(-1) > 0)  # [B, K]
        return z_comp, comp_valid

    def _causal_expand(self, z_comp: torch.Tensor, boundaries: torch.Tensor, T: int):
        """
        Causal expansion: token i uses the most recent compressed position k
        with boundaries[k] <= i. Never maps a token to a future-containing summary.
        """
        B, K, Dc = z_comp.shape
        positions = torch.arange(T, device=z_comp.device, dtype=boundaries.dtype)
        idx = torch.searchsorted(boundaries, positions, right=True) - 1
        idx = idx.clamp(0, K - 1)  # [T]
        return z_comp[:, idx, :]   # [B, T, Dc]

    def _compressed_attn(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        K = self.compressed_windows
        V = self.compressed_views

        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool, device=x.device)
        else:
            valid = torch.ones((B, T), dtype=torch.bool, device=x.device)

        if self.compressed_stride is not None:
            boundaries = self._fixed_stride_boundaries(T, x.device)
            K = int(boundaries.shape[0])
        else:
            boundaries = self._causal_boundaries(T, K, x.device)
        causal_K = torch.tril(torch.ones(K, K, device=x.device, dtype=torch.bool))

        view_outputs = []
        for v in range(V):
            # Independent down-projection -> non-identical inputs per view.
            z = self.view_down[v](x)                              # [B, T, Dc]
            z_comp, comp_valid = self._causal_prefix_compress(z, valid, boundaries)  # [B, K, Dc]

            # Causal attention over K compressed positions.
            pos = boundaries.unsqueeze(0).expand(B, K) if self.use_rope else None
            q, k, vv = self._project_qkv(z_comp, self.view_qkv[v],
                                        self.compressed_heads, self.compressed_head_dim,
                                        positions=pos)
            out_c = self._safe_attention(q, k, vv, causal_K, comp_valid, comp_valid)
            out_c = out_c.transpose(1, 2).reshape(B, K, self.compressed_dim)
            out_c = self.view_out[v](out_c)                       # [B, K, Dc]

            # Zero invalid compressed positions (no bias-generated pseudo-tokens).
            out_c = torch.where(comp_valid.unsqueeze(-1), out_c, torch.zeros_like(out_c))
            out_c = self._check_finite_output(out_c, comp_valid, "compressed")

            # Causal expansion back to full resolution.
            h = self._causal_expand(out_c, boundaries, T)        # [B, T, Dc]
            h = torch.where(valid.unsqueeze(-1), h, torch.zeros_like(h))

            view_out = self.view_up[v](h)                         # [B, T, D]
            view_out = torch.where(valid.unsqueeze(-1), view_out, torch.zeros_like(view_out))
            view_outputs.append(view_out)

        # Internal consensus across V views (tokenwise, across view axis -> causal).
        stacked_views = torch.stack(view_outputs, dim=2)         # [B, T, V, D]
        comp = self._tokenwise_consensus(
            stacked_views, self.view_consensus_temp, self.view_consensus_norm)
        return comp

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------
    def _tokenwise_consensus(self, stacked: torch.Tensor, temp: nn.Parameter, norm: RMSNorm):
        """
        Tokenwise cosine consensus across the view axis.
        stacked: [B, T, V, D] -> [B, T, D]
        Causal by construction: only same-position views are compared.
        """
        stacked_norm = F.normalize(stacked, dim=-1, p=2)
        sim = torch.matmul(stacked_norm, stacked_norm.transpose(-2, -1))  # [B, T, V, V]
        confidence = sim.mean(dim=-1)                                      # [B, T, V]
        vote_weights = F.softmax(confidence * temp, dim=-1).unsqueeze(-1)  # [B, T, V, 1]
        consensus = (stacked * vote_weights).sum(dim=2)                    # [B, T, D]
        return norm(consensus)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, state=None, cache=None, attention_mask=None, **kwargs):
        x = self.norm(x)
        B, T, D = x.shape

        # ===== Layer 1: Ensemble (three causal pathways) =====
        local = self._windowed_attn(x, self.local_window, attention_mask=attention_mask)
        coarse = self._windowed_attn(x, self.coarse_window, attention_mask=attention_mask)
        comp = self._compressed_attn(x, attention_mask=attention_mask)

        # ===== Layer 2: External consensus across (local, coarse, compressed) =====
        stacked = torch.stack([local, coarse, comp], dim=2)  # [B, T, 3, D]

        if self.consensus_type == "mha":
            B2, T2, V2, D2 = stacked.shape
            flat = stacked.reshape(B2 * T2, V2, D2)
            cv_out, attn_weights = self.cross_view_attn(flat, flat, flat)
            cv_out = self.cross_view_norm(flat + cv_out)
            self_confidence = attn_weights.diagonal(dim1=1, dim2=2)  # [B*T, 3]
            vote_weights = F.softmax(self_confidence, dim=-1).unsqueeze(-1)
            consensus = (cv_out * vote_weights).sum(dim=1).reshape(B, T, D)
        else:  # cosine (default)
            consensus = self._tokenwise_consensus(
                stacked, self.consensus_temp, self.consensus_norm)

        # ===== Layer 3: Gated causal correction =====
        corrector_input = torch.cat([consensus, x], dim=-1)

        if self.corrector_type == "ffn":
            delta = self.corrector(corrector_input)
        else:  # attn (now causal + mask-aware)
            cpos = (torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                    if self.use_rope else None)
            q, k, v = self._project_qkv(consensus, self.corrector_qkv,
                                       self.n_heads, self.head_dim, positions=cpos)
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            if attention_mask is not None:
                cvalid = attention_mask.to(dtype=torch.bool, device=x.device)
            else:
                cvalid = torch.ones((B, T), dtype=torch.bool, device=x.device)
            c_out = self._safe_attention(q, k, v, causal, cvalid, cvalid)
            c_out = c_out.transpose(1, 2).reshape(B, T, D)
            c_out = self.corrector_out(c_out)
            c_att = self.corrector_norm1(consensus + c_out)
            delta = self.corrector_ffn(c_att)
            delta = self.corrector_norm2(c_att + delta)

        fix_gate = torch.sigmoid(self.corrector_gate(corrector_input))
        corrected = consensus + fix_gate * delta

        # ===== Gated residual merge with original input =====
        out_gate = torch.sigmoid(self.out_gate(torch.cat([corrected, x], dim=-1)))
        out = x + out_gate * self.dropout(corrected)

        # Issue 6: NO output FFN here. The graph appends SwiGLU.

        # Zero padded positions so the node never injects bias into pad slots.
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool, device=x.device).unsqueeze(-1)
            out = torch.where(valid, out, torch.zeros_like(out))

        return out, None


class LinearAttnNode(HeteroNode):
    """
    Causal linear attention node using feature maps.
    O(n) in sequence length during training via prefix sums.
    """
    def __init__(self, d_model: int, n_heads: int = 4, feature_dim: int = 64, dropout: float = 0.0,
                 attn_dropout: float = 0.0):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_feat = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.k_feat = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout > 0 else dropout)

        self._reset_parameters()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # AMP-safe: compute feature maps and cumulatives in fp32 to prevent
        # float16 overflow. The einsum over feature_dim can exceed 65504.
        # Cast weights to fp32 so matmul dtype matches even when weights are bf16.
        q_fp32 = self._feature_map(F.linear(q.float(), self.q_feat.weight.float(),
                                            self.q_feat.bias.float() if self.q_feat.bias is not None else None))
        k_fp32 = self._feature_map(F.linear(k.float(), self.k_feat.weight.float(),
                                            self.k_feat.bias.float() if self.k_feat.bias is not None else None))
        v_fp32 = v.float()

        # NaN/Inf isolation: a single contaminated position would otherwise
        # spread through the prefix scans (kv_cum and z below) to EVERY later
        # position in the sequence. Zero the bad position's contribution
        # instead of poisoning the suffix. Positions before it stay exact.
        q_fp32 = torch.nan_to_num(q_fp32, nan=0.0, posinf=0.0, neginf=0.0)
        k_fp32 = torch.nan_to_num(k_fp32, nan=0.0, posinf=0.0, neginf=0.0)
        v_fp32 = torch.nan_to_num(v_fp32, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply attention mask: zero out k,v contributions from pad positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, T, 1)
            k_fp32 = k_fp32 * mask
            v_fp32 = v_fp32 * mask

        kv = torch.einsum('bhTf,bhTd->bhTfd', k_fp32, v_fp32)

        # Move T to the LAST dimension so cumsum scans contiguous memory (stride = 1)
        kv = kv.permute(0, 1, 3, 4, 2).contiguous()   # [B, H, F, D, T]
        kv_cum = torch.cumsum(kv, dim=-1)              # one contiguous scan per row
        kv_cum = kv_cum.permute(0, 1, 4, 2, 3)        # [B, H, T, F, D]

        # Normaliser of causal linear attention: z_t = phi(q_t) . sum_{s<=t} phi(k_s),
        # so the weights phi(q_t).phi(k_s) / z_t sum to 1 over s <= t. The old
        # sum over features of cumsum(phi(k)) left q out of the denominator, so
        # the output was not a convex combination of the values.
        z = torch.einsum('bhTf,bhTf->bhT', q_fp32, torch.cumsum(k_fp32, dim=2)).unsqueeze(-1).clamp(min=1e-6)

        out = torch.einsum('bhTf,bhTfd->bhTd', q_fp32, kv_cum) / z
        out = out.to(x.dtype)  # cast back to fp16/bf16
        # --------------------------------------------------------------

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.attn_dropout(self.out_proj(out))
        return out, None


class FullAttnNode(HeteroNode):
    """Standard causal softmax attention with multi-head support and optional RoPE."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0, use_rope: bool = True,
                 attn_dropout: float = 0.0, rope_theta: float = 10000.0):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.rope_theta = float(rope_theta)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout > 0 else dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # use_rope was accepted and ignored: the node had no positional signal
        # beyond the causal mask. Rotate q/k by absolute position (relative
        # offsets only enter the scores).
        if self.use_rope:
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            q = apply_rotary(q, pos, self.rope_theta)
            k = apply_rotary(k, pos, self.rope_theta)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask
        allowed = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        # Padding mask: prevent attending to padded positions
        valid = None
        if attention_mask is not None:
            # attention_mask: (B, T) with 0 for pad, 1 for real
            valid = attention_mask.to(device=x.device).bool()
            allowed = allowed & valid.view(B, 1, 1, T)
        allowed = allowed.expand(B, self.n_heads, T, T)
        # A query with no lawful key (e.g. a left-padded position) softmaxed over
        # all -inf and produced NaN; give such rows a finite placeholder and zero
        # their weights instead.
        row_has_key = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, float('-inf'))
        scores = torch.where(row_has_key, scores, torch.zeros_like(scores))

        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        attn = torch.where(row_has_key, attn, torch.zeros_like(attn))
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.resid_dropout(self.out_proj(out))
        if valid is not None:
            # padded query positions carry no output (no out_proj bias leak)
            out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        return out, None


class DenseNode(HeteroNode):
    """Dense processing node with GELU activation."""
    def __init__(self, d_model: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__(d_model)
        h = int(d_model * expansion)
        self.w1 = nn.Linear(d_model, h)
        self.w2 = nn.Linear(h, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        h = F.gelu(self.w1(x))
        out = self.w2(h)
        return self.dropout(out), None


class SwiGLUNode(HeteroNode):
    """SwiGLU activation node as used in modern LLMs."""
    def __init__(self, d_model: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__(d_model)
        h = int(d_model * expansion)
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        h = F.silu(self.gate(x)) * self.up(x)
        out = self.down(h)
        return self.dropout(out), None


class SSMNode(HeteroNode):
    """
    Simplified SSM node (Mamba-style) with efficient batched sequential scan.
    For production, replace with Mamba2SSD from mamba2.py.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__(d_model)
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=False
        )
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log.data = torch.log(A)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        x_and_gate = self.in_proj(x)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)
        x_inner = x_inner * F.silu(gate)

        x_conv = self.conv(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log.float())
        # (named B_sel, not B: `B` is the batch size from x.shape above, and
        # shadowing it made torch.zeros(B, ...) below crash with a tensor arg)
        B_sel = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        dt = dt.float()
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B_sel.unsqueeze(2)

        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            y = y + self.D * x_conv[:, t]
            ys.append(y)

        out = torch.stack(ys, dim=1)
        out = self.dropout(self.out_proj(out.to(x.dtype)))
        return out, h.detach()


class Mamba2Node(HeteroNode):
    """Mamba-2 SSD node wrapper."""
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2,
                 dt_rank: str = "auto", conv_bias: bool = True, bias: bool = False, dropout: float = 0.0):
        super().__init__(d_model)
        from .mamba2 import Mamba2SSD
        self.mamba = Mamba2SSD(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dt_rank=dt_rank, conv_bias=conv_bias, bias=bias,
            use_fast_path=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        out, new_state = self.mamba(x, state=state)
        return self.dropout(out), new_state


class GateNode(HeteroNode):
    """Aggregation node with learned softmax weighted sum.

    Dead-parameter fixes:
      - forward() now applies self.norm (previously built by HeteroNode but
        never used, so every gate's norm.weight received grad=None).
      - A single-predecessor gate is a pure pass-through: softmax over one
        logit is identically 1, so its mixing weights could never receive
        gradient. HelixGraph calls set_n_preds() with the ACTUAL predecessor
        count after wiring; with <= 1 predecessor the weights parameter is
        removed instead of left dead. This also closes the opposite hazard:
        when actual predecessors exceeded n_preds, weights[:n] truncated and
        zip() silently dropped the extra predecessors.
    """
    def __init__(self, d_model: int, n_preds: int = 2, dropout: float = 0.0):
        super().__init__(d_model)
        self.n_preds = n_preds
        self.weights = nn.Parameter(torch.ones(n_preds) / n_preds) if n_preds > 1 else None
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def set_n_preds(self, n_preds: int):
        """Re-size (or remove) mixing weights to match the actual wiring."""
        self.n_preds = n_preds
        if n_preds > 1:
            self.weights = nn.Parameter(
                torch.ones(n_preds, device=self.out_proj.weight.device) / n_preds
            )
        else:
            self.weights = None  # single-predecessor pass-through, no dead param

    def forward(self, x_list, state: Any = None, cache: Any = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        if not isinstance(x_list, list):
            raise TypeError("GateNode expects a list of predecessor tensors")
        n = len(x_list)
        if n == 0:
            raise ValueError("GateNode received no predecessors")
        if n == 1 or self.weights is None:
            if n > 1:
                # weights removed (single-pred at build time) but more inputs
                # arrived: fall back to a uniform mean rather than dropping any.
                out = torch.stack(x_list, dim=0).mean(dim=0)
            else:
                out = x_list[0]
        else:
            weights = F.softmax(self.weights[:n], dim=0)
            out = sum(w * x for w, x in zip(weights, x_list))
        out = self.out_proj(self.norm(out))
        return self.dropout(out), None



class TitansMemoryNode(HeteroNode):
    """
    Titans-style neural long-term memory node for HelixLM.

    Maintains persistent memory across forward passes using a surprise-gated
    delta rule. Compatible with the existing heterogeneous graph interface.

    Architecture (based on Behrouz et al. 2025 "Titans: Learning to Memorize
    at Test Time" MAC variant):
        - Keys and values are projected from the input hidden states.
        - A persistent memory tensor M (batch, feature_dim, d_model) stores
          the long-term key->value mapping via outer-product updates.
        - Surprise metric = ||v_pred - v_true|| drives update magnitude.
        - Retrieval uses query projection + ELU feature map.

    The memory tensor is returned as ``state`` and can be persisted across
    chunks by the caller (HelixRecurrentBlock / HelixLMCore).
    """
    def __init__(
        self,
        d_model: int,
        feature_dim: int = 64,
        eta_init: float = 0.01,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections for memory keys, values, and retrieval queries
        self.k_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, feature_dim, bias=False)

        # Learnable per-coordinate learning rate for memory updates
        self.eta = nn.Parameter(torch.ones(feature_dim) * eta_init)

        # Feature map: ELU + 1 (standard in Titans / linear attention literature)
        self.phi = lambda x: F.elu(x, alpha=1.0) + 1.0

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize a zero memory tensor for a new batch."""
        return torch.zeros(batch_size, self.feature_dim, self.d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, T, D) input hidden states.
            state: Previous persistent memory tensor (B, feature_dim, D) or None.
            cache: Unused (for API compatibility).

        Returns:
            (output, updated_memory) where updated_memory has shape
            (B, feature_dim, D) and should be passed back on the next chunk.
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # 1. Retrieve or initialize persistent memory
        if state is not None:
            M = state  # (B, feature_dim, D)
        else:
            M = self._init_memory(B, device, dtype)

        # 2. Pre-norm and project to keys / values
        x_norm = self.norm(x)

        k = self.phi(self.k_proj(x_norm))  # (B, T, feature_dim)
        v = self.v_proj(x_norm)             # (B, T, d_model)

        # 3. Memory read + update loop (test-time learning).
        # Causal: token t reads the memory BEFORE it is written with token t
        # (it holds tokens < t only). Previously every position read the memory
        # after the whole sequence had been written, so position 0 saw the future.
        eta = self.eta.abs().clamp(min=1e-6)  # (feature_dim,)
        q = self.phi(self.q_proj(x_norm))  # (B, T, feature_dim)
        valid = (attention_mask.to(device=device).bool()
                 if attention_mask is not None else None)

        reads = []
        for t in range(T):
            k_t = k[:, t, :]          # (B, feature_dim)
            v_t = v[:, t, :]          # (B, d_model)

            # 4. Retrieval for position t from memory over tokens < t
            reads.append(torch.einsum('bf,bfd->bd', q[:, t, :], M))

            # Surprise metric: deviation of memory prediction from true value
            v_pred = torch.einsum('bf,bfd->bd', k_t, M)
            surprise = torch.norm(v_t - v_pred, dim=-1, keepdim=True)  # (B, 1)

            # Delta rule update: outer product of key and value
            delta = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(1))

            # Surprise-gated update with learnable per-coordinate LR
            M_new = M + eta.view(1, -1, 1) * surprise.unsqueeze(-1) * delta

            # Layer norm over the feature_dim dimension to prevent explosion
            M_new = F.layer_norm(M_new, M_new.shape[-2:])
            # padded positions do not write into the memory
            M = M_new if valid is None else torch.where(valid[:, t].view(B, 1, 1), M_new, M)

        mem_out = torch.stack(reads, dim=1)  # (B, T, d_model)

        # 5. Output projection + residual
        out = self.dropout(self.out_proj(mem_out))
        output = x + out

        return output, M
