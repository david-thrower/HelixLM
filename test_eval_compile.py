"""Eval/generation forward compiles as one graph and matches eager.

Guards two host syncs that broke the graph on every eval call:
  * hf_model._canonical_eval_hidden: int(n.max()) length check
  * recurrent: `halted.all()` ACT early exit
"""
from __future__ import annotations

import contextlib
import io
import unittest

import torch

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM


def _tiny_model() -> HelixForCausalLM:
    cfg = HelixConfig(
        attention_mode="multi_scale_windowed", d_model=64, n_heads=4, n_loops=2,
        n_columns=2, nodes_per_column=(2, 2), ffn_expansion=2.0, use_ssm=False,
        use_titans_memory=False, use_rope=True, seq_len=256, vocab_size=257,
        dropout=0.0, attn_dropout=0.0, local_window=64, coarse_window=128,
        compressed_windows=8, compressed_views=2, grad_buffer_ratio=0.0, seed=42,
        pad_token_id=0, eos_token_id=0, bos_token_id=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return HelixForCausalLM(cfg).eval()


class EvalCompileTest(unittest.TestCase):
    def test_eval_forward_has_no_graph_break(self):
        torch._dynamo.reset()
        torch.manual_seed(1)
        model = _tiny_model()
        ids = torch.randint(1, 257, (2, 40))
        mask = torch.ones_like(ids)
        mask[1, :10] = 0  # left padded row
        with torch.no_grad():
            eager = model(input_ids=ids, attention_mask=mask).logits
            compiled = torch.compile(model, fullgraph=True)(
                input_ids=ids, attention_mask=mask).logits
        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)

    def test_rows_longer_than_seq_len_use_raw_layout(self):
        model = _tiny_model()
        ids = torch.randint(1, 257, (1, 300))  # 300 valid tokens > seq_len 256
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        self.assertEqual(tuple(out.shape), (1, 300, 257))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
