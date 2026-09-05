from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


LAUNCHER_PATH = Path(__file__).with_name("113M_param_train.py")
SPEC = importlib.util.spec_from_file_location("branch62_launcher", LAUNCHER_PATH)
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)


class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 1

    def __len__(self):
        return 50_257


class Branch62LauncherTest(unittest.TestCase):
    def test_rtx5080_profile_preserves_relative_comparison_shape(self):
        settings = LAUNCHER.resolve_settings({})

        self.assertEqual(settings.profile.name, "rtx5080-relative")
        self.assertEqual(settings.profile.d_model, 768)
        self.assertEqual(settings.profile.n_heads, 12)
        self.assertEqual(settings.profile.batch_size, 3)
        self.assertEqual(settings.profile.grad_accum, 28)
        self.assertEqual(settings.profile.effective_batch, 84)
        self.assertEqual(settings.compile_store_dir, Path("pretrain_store"))
        self.assertEqual(settings.dataset, LAUNCHER.SUTRA_DATASET)
        self.assertEqual(settings.dataset_revision, LAUNCHER.SUTRA_REVISION)
        cfg = LAUNCHER.build_config(settings, TokenizerStub())
        self.assertEqual(cfg.d_model, 768)
        self.assertEqual(cfg.seq_len, 1024)
        self.assertEqual(cfg.ffn_expansion, 3.0)
        self.assertEqual(cfg.lateral_p, 0.8)
        self.assertEqual(cfg.vertical_p, 0.9)
        self.assertEqual(cfg.vertical_depth, 2)

    def test_exact_profile_preserves_branch60_overnight_shape(self):
        settings = LAUNCHER.resolve_settings(
            {"HELIX_PROFILE": "branch60-exact-shape"}
        )

        self.assertEqual(settings.profile.d_model, 1024)
        self.assertEqual(settings.profile.n_heads, 16)
        self.assertEqual(settings.profile.batch_size, 2)
        self.assertEqual(settings.profile.grad_accum, 42)
        self.assertEqual(settings.profile.effective_batch, 84)
        self.assertEqual(settings.max_optimizer_steps, 3082)
        self.assertEqual(
            settings.profile.reference_run_id,
            "6ad46206ff1d49a3a96d71fd7723f16b",
        )

    def test_existing_store_disables_the_default_compile_target(self):
        settings = LAUNCHER.resolve_settings(
            {"HELIX_PRETRAIN_STORE_DIR": "/data/sutra-store"}
        )

        self.assertEqual(settings.train_store_dir, Path("/data/sutra-store"))
        self.assertIsNone(settings.compile_store_dir)

    def test_explicit_dual_store_modes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LAUNCHER.resolve_settings(
                {
                    "HELIX_PRETRAIN_STORE_DIR": "/data/existing",
                    "HELIX_PRETRAIN_COMPILE_DIR": "/data/new",
                }
            )

    def test_hugging_face_name_is_bounded_and_descriptive(self):
        settings = LAUNCHER.resolve_settings({"HELIX_EPOCHS": "3"})
        name = LAUNCHER.model_name(settings, "260904-2300")

        self.assertLessEqual(len(name), 96)
        self.assertIn("d768-c3-n333-l4-f30-s1024-e3", name)


if __name__ == "__main__":
    unittest.main()
