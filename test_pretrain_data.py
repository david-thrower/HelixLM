#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from helix_lm.pretrain_data import (
    PretrainIndexedDataset,
    PretrainPermutation,
    PretrainPermutationSampler,
    PretrainSampleCompiler,
    create_pretrain_indexed_loader,
)
from helix_lm.dataset import ContinuousWindowDataset
from helix_lm.trainer import PretrainTrainer, get_cosine_schedule_with_warmup


class IntegerTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        self.last_add_special_tokens = add_special_tokens
        return [int(value) for value in text.split()]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None, attention_mask=None, cca_step=None):
        return {"loss": self.weight.square() + input_ids.float().mean() * 0}

    def count_parameters(self):
        return {"total": 1}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


class StreamingTexts:
    def __init__(self, values):
        self.values = tuple(values)

    def __iter__(self):
        return iter(self.values)


class PretrainDataTest(unittest.TestCase):
    def compile_fixture(self, root: Path, *, samples_per_shard=2):
        return PretrainSampleCompiler(
            IntegerTokenizer(),
            4,
            root,
            samples_per_shard=samples_per_shard,
            source={"fixture": "integer-v1"},
        ).compile(["1 2", "3 4 5", "6 7 8 9 10"])

    def test_compiler_preserves_continuous_eos_joined_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            manifest = self.compile_fixture(root)
            dataset = PretrainIndexedDataset(root, verify=True)

            self.assertEqual(manifest.sample_count, 3)
            self.assertEqual(manifest.value["causal_target_count"], 9)
            self.assertEqual(manifest.value["dropped_tail_tokens"], 1)
            self.assertEqual(dataset[0]["input_ids"].tolist(), [1, 2, 99, 3])
            self.assertEqual(dataset[1]["input_ids"].tolist(), [4, 5, 99, 6])
            self.assertEqual(dataset[2]["input_ids"].tolist(), [7, 8, 9, 10])

    def test_persisted_permutation_is_unique_and_resumeable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch-0000.u32"
            created = PretrainPermutation.create(path, 32, 42, epoch=0)
            loaded = PretrainPermutation.load(path)
            values = [int(value) for value in loaded.values()]

            self.assertEqual(created.metadata, loaded.metadata)
            self.assertEqual(sorted(values), list(range(32)))
            self.assertEqual(
                list(PretrainPermutationSampler(loaded, cursor=7)),
                values[7:],
            )

    def test_permutation_cursor_counts_only_non_holdout_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch-0000.u32"
            permutation = PretrainPermutation.create(path, 12, 42, epoch=0)
            values = [int(value) for value in permutation.values()]
            holdout = values[-3:]
            admitted = [value for value in values if value not in holdout]

            sampler = PretrainPermutationSampler(
                permutation,
                cursor=4,
                excluded_sample_ids=holdout,
            )

            self.assertEqual(len(sampler), len(admitted) - 4)
            self.assertEqual(list(sampler), admitted[4:])

    def test_pretrain_permutation_load_refuses_metadata_path_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch-0000.u32"
            PretrainPermutation.create(path, 8, 42, epoch=0)
            metadata_path = path.with_suffix(path.suffix + ".json")
            metadata = __import__("json").loads(metadata_path.read_text())
            metadata["file"] = "another-permutation.u32"
            metadata_path.write_text(__import__("json").dumps(metadata))

            with self.assertRaisesRegex(ValueError, "metadata path mismatch"):
                PretrainPermutation.load(path)

    def test_disk_loader_replays_the_exact_persisted_batch_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            self.compile_fixture(root, samples_per_shard=1)
            dataset = PretrainIndexedDataset(root, verify=True)
            permutation = PretrainPermutation.create(
                root / "permutations" / "epoch-0000.u32",
                len(dataset),
                7,
                epoch=0,
            )
            expected_ids = [int(value) for value in permutation.values()]
            loader = create_pretrain_indexed_loader(
                dataset,
                permutation,
                batch_size=2,
                drop_last=False,
                pin_memory=False,
            )

            observed_ids = []
            observed_tokens = []
            for batch in loader:
                observed_ids.extend(batch["sample_id"].tolist())
                observed_tokens.extend(batch["input_ids"].tolist())

            self.assertEqual(observed_ids, expected_ids)
            self.assertEqual(
                observed_tokens,
                [dataset[sample_id]["input_ids"].tolist() for sample_id in expected_ids],
            )
            self.assertTrue(all(label.dtype == torch.int64 for label in [batch["labels"]]))

    def test_compiled_and_live_pretraining_paths_have_identical_samples(self):
        texts = ["1 2", "3 4 5", "6 7 8 9 10"]
        tokenizer = IntegerTokenizer()
        live = list(ContinuousWindowDataset(texts, tokenizer, 4, shuffle=False))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            PretrainSampleCompiler(tokenizer, 4, root).compile(texts)
            compiled = PretrainIndexedDataset(root, verify=True)

            self.assertEqual(len(live), len(compiled))
            for sample_id, live_sample in enumerate(live):
                compiled_sample = compiled[sample_id]
                self.assertTrue(torch.equal(live_sample["input_ids"], compiled_sample["input_ids"]))
                self.assertTrue(torch.equal(live_sample["labels"], compiled_sample["labels"]))

    def test_streaming_input_auto_compiles_without_an_explicit_store_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )

            with mock.patch("tempfile.gettempdir", return_value=str(temporary)):
                trainer = PretrainTrainer(
                    model=TinyModel(), cfg=cfg,
                    train_texts=StreamingTexts(
                        ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
                    ),
                    tokenizer=IntegerTokenizer(),
                    output_dir=temporary / "checkpoints",
                    seed=42, num_workers=0, verbose=False,
                )

            self.assertTrue(trainer._indexed_train)
            self.assertIsNotNone(trainer._auto_compiled_store_dir)
            self.assertTrue(
                Path(trainer._auto_compiled_store_dir, "manifest.json").is_file()
            )

    def test_auto_compile_refuses_an_existing_empty_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            empty_store = temporary / "existing-empty"
            empty_store.mkdir()
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )

            with self.assertRaises(FileNotFoundError):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg,
                    train_texts=StreamingTexts(["1 2 3", "4 5 6"]),
                    pretrain_store_dir=empty_store,
                    tokenizer=IntegerTokenizer(),
                    output_dir=temporary / "checkpoints",
                    seed=42, num_workers=0, verbose=False,
                )

    def test_manifest_verification_rejects_mutated_sample_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "samples"
            manifest = self.compile_fixture(root)
            shard_path = root / manifest.value["shards"][0]["file"]
            payload = bytearray(shard_path.read_bytes())
            payload[0] ^= 1
            shard_path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                PretrainIndexedDataset(root, verify=True)

    def test_pretrain_trainer_writes_exact_data_resume_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            self.compile_fixture(root)
            cfg = SimpleNamespace(
                seq_len=4,
                batch_size=2,
                lr=0.01,
                weight_decay=0.0,
                warmup_steps=1,
                grad_clip=1.0,
                device="cpu",
                epochs=1,
                use_titans_memory=False,
                use_cca=False,
                max_new_tokens=1,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(),
                cfg=cfg,
                train_store_dir=root,
                tokenizer=IntegerTokenizer(),
                output_dir=temporary / "checkpoints",
                seed=42,
                num_workers=0,
                verbose=False,
            )

            trainer.train_epoch(1)
            trainer.save_checkpoint(1, "resume-court")
            state_path = temporary / "checkpoints" / "resume-court" / "pretrain_data_state.json"
            state = __import__("json").loads(state_path.read_text())

            self.assertEqual(state["sample_cursor"], 2)
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(
                state["dataset_manifest_sha256"],
                trainer._train_dataset.manifest.manifest_sha256,
            )
            self.assertEqual(
                state["permutation_sha256"],
                trainer._train_permutation.metadata["sha256"],
            )
            self.assertTrue(
                (
                    temporary / "checkpoints" / "resume-court" /
                    "pretrain_training_state.pt"
                ).is_file()
            )

    def test_pretrain_trainer_restores_model_optimizer_rng_and_data_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12", "13 14 15", "16 17 18"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            original = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False, total_optimizer_steps=20,
            )
            original.model.weight.data.fill_(3.0)
            original.scheduler = get_cosine_schedule_with_warmup(
                original.optimizer,
                num_warmup_steps=original._scheduler_warmup,
                num_training_steps=20,
                num_cycles=original._scheduler_cycles,
                min_lr_ratio=original._scheduler_min_lr,
            )
            original.optimizer.zero_grad()
            original.model.weight.square().backward()
            original.optimizer.step()
            original.scheduler.step()
            original.global_step = 1
            original._train_cursor = 2
            expected_weight = original.model.weight.detach().clone()
            expected_next_id = int(original._train_permutation.values()[2])
            expected_rng = torch.get_rng_state().clone()
            expected_scheduler_step = original.scheduler.last_epoch
            original.save_checkpoint(1, "resume-court")
            torch.manual_seed(999)

            restored = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "restored",
                seed=42, num_workers=0, verbose=False,
                total_optimizer_steps=20,
                resume_training_state=(
                    temporary / "checkpoints" / "resume-court" /
                    "pretrain_training_state.pt"
                ),
            )

            self.assertEqual(restored.global_step, 1)
            self.assertEqual(restored._train_cursor, 2)
            self.assertTrue(torch.equal(restored.model.weight, expected_weight))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            first_batch = next(iter(restored.train_loader))
            self.assertEqual(int(first_batch["sample_id"][0]), expected_next_id)
            self.assertTrue(restored.optimizer.state_dict()["state"])
            restored.train_epoch(1)
            self.assertEqual(restored.global_step, 3)
            self.assertEqual(restored.scheduler.last_epoch, expected_scheduler_step + 2)
            self.assertIsNone(restored._resume_scheduler_state)

            with self.assertRaisesRegex(ValueError, "scheduler configuration mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "mismatch",
                    seed=42, num_workers=0, verbose=False,
                    total_optimizer_steps=40,
                    resume_training_state=(
                        temporary / "checkpoints" / "resume-court" /
                        "pretrain_training_state.pt"
                    ),
                )

            changed_lr_cfg = SimpleNamespace(**vars(cfg))
            changed_lr_cfg.lr = 0.02
            with self.assertRaisesRegex(ValueError, "training configuration mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=changed_lr_cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "lr-mismatch",
                    seed=42, num_workers=0, verbose=False,
                    total_optimizer_steps=20,
                    resume_training_state=(
                        temporary / "checkpoints" / "resume-court" /
                        "pretrain_training_state.pt"
                    ),
                )

    def test_pretrain_trainer_refuses_incompatible_exact_runtime_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            self.compile_fixture(root)
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
            )
            trainer.save_checkpoint(1, "resume-court")
            state_path = (
                temporary / "checkpoints" / "resume-court" /
                "pretrain_training_state.pt"
            )

            state = torch.load(state_path, map_location="cpu", weights_only=True)
            state["cuda_rng_state_all"] = [torch.get_rng_state()]
            torch.save(state, state_path)
            with self.assertRaisesRegex(ValueError, "CUDA RNG configuration mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "cuda-mismatch",
                    seed=42, num_workers=0, verbose=False,
                    resume_training_state=state_path,
                )

            state["cuda_rng_state_all"] = None
            state["scaler"] = {"scale": torch.tensor(1.0)}
            torch.save(state, state_path)
            with self.assertRaisesRegex(ValueError, "AMP scaler configuration mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "scaler-mismatch",
                    seed=42, num_workers=0, verbose=False,
                    resume_training_state=state_path,
                )

    def test_resume_stage_plan_skips_completed_stage_without_duplicate_result(self):
        partial = {
            "permutation_epoch": 1,
            "sample_cursor": 2,
            "usable_sample_count": 4,
        }
        complete = {**partial, "sample_cursor": 4}

        self.assertEqual(PretrainTrainer.resume_stage_plan(partial, 3), (1, True))
        self.assertEqual(PretrainTrainer.resume_stage_plan(complete, 3), (2, False))
        self.assertEqual(
            PretrainTrainer.resume_stage_plan(
                {**complete, "permutation_epoch": 2},
                3,
            ),
            (3, False),
        )

    def test_pretrain_trainer_refuses_resume_from_another_sample_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
            )
            trainer.save_checkpoint(1, "resume-court")
            state_path = (
                temporary / "checkpoints" / "resume-court" /
                "pretrain_training_state.pt"
            )
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            state["dataset_manifest_sha256"] = "0" * 64
            torch.save(state, state_path)

            with self.assertRaisesRegex(ValueError, "dataset manifest mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "restored",
                    seed=42, num_workers=0, verbose=False,
                    resume_training_state=state_path,
                )

    def test_pretrain_trainer_refuses_resume_from_another_permutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
            )
            trainer.save_checkpoint(1, "resume-court")
            state_path = (
                temporary / "checkpoints" / "resume-court" /
                "pretrain_training_state.pt"
            )
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            state["permutation_sha256"] = "0" * 64
            torch.save(state, state_path)

            with self.assertRaisesRegex(ValueError, "permutation mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "restored",
                    seed=42, num_workers=0, verbose=False,
                    resume_training_state=state_path,
                )

    def test_pretrain_trainer_refuses_misfiled_epoch_permutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
            )
            misfiled = root / "permutations" / "epoch-0001-seed-42.u32"
            PretrainPermutation.create(misfiled, 4, 42, epoch=0)
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )

            with self.assertRaisesRegex(ValueError, "permutation epoch mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "output",
                    seed=42, num_workers=0, verbose=False,
                    train_permutation_epoch=1,
                )

    def test_pretrain_train_runs_when_initial_permutation_epoch_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "output",
                seed=42, num_workers=0, verbose=False,
                train_permutation_epoch=1,
            )

            history = trainer.train(num_epochs=1)

            self.assertEqual(len(history["train_loss"]), 1)
            self.assertGreater(trainer.global_step, 0)
            self.assertEqual(trainer._train_permutation_epoch, 1)

    def test_pretrain_trainer_uses_a_distinct_persisted_order_per_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12", "13 14 15", "16 17 18"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=2,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
            )

            trainer.train_epoch(1)
            first_root = trainer._train_permutation.metadata["sha256"]
            trainer.train_epoch(2)
            second_root = trainer._train_permutation.metadata["sha256"]

            self.assertEqual(trainer._train_permutation_epoch, 1)
            self.assertNotEqual(first_root, second_root)
            self.assertEqual(trainer._train_cursor, len(trainer._train_dataset))
            self.assertTrue((root / "permutations" / "epoch-0001-seed-42.u32").is_file())

    def test_indexed_validation_ids_are_fixed_and_excluded_from_every_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12", "13 14 15", "16 17 18"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=2,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
                validation_sample_count=2,
            )

            validation_ids = tuple(
                int(sample_id)
                for batch in trainer.val_loader
                for sample_id in batch["sample_id"]
            )
            epoch_zero_train_ids = tuple(
                int(sample_id)
                for batch in trainer.train_loader
                for sample_id in batch["sample_id"]
            )
            trainer._activate_indexed_epoch(1)
            epoch_one_train_ids = tuple(
                int(sample_id)
                for batch in trainer.train_loader
                for sample_id in batch["sample_id"]
            )

            self.assertEqual(validation_ids, trainer._validation_sample_ids)
            self.assertEqual(len(validation_ids), 2)
            self.assertTrue(set(validation_ids).isdisjoint(epoch_zero_train_ids))
            self.assertTrue(set(validation_ids).isdisjoint(epoch_one_train_ids))
            self.assertEqual(set(epoch_zero_train_ids), set(epoch_one_train_ids))

            trainer.save_checkpoint(1, "validation-root")
            state_path = (
                temporary / "checkpoints" / "validation-root" /
                "pretrain_training_state.pt"
            )
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            state["validation_sample_ids_sha256"] = "0" * 64
            torch.save(state, state_path)
            with self.assertRaisesRegex(ValueError, "validation identity mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "restored",
                    seed=42, num_workers=0, verbose=False,
                    validation_sample_count=2,
                    resume_training_state=state_path,
                )

    def test_pretrain_trainer_resumes_completed_epoch_at_next_persisted_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12", "13 14 15", "16 17 18"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=2,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            original = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
            )
            original.train_epoch(1)
            original.save_checkpoint(1, "epoch-one")

            restored = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "restored",
                seed=42, num_workers=0, verbose=False,
                resume_training_state=(
                    temporary / "checkpoints" / "epoch-one" /
                    "pretrain_training_state.pt"
                ),
            )

            self.assertEqual(restored._train_permutation_epoch, 1)
            self.assertEqual(restored._train_cursor, 0)
            restored.train(num_epochs=2)
            self.assertEqual(restored._train_permutation_epoch, 1)
            self.assertEqual(restored._train_cursor, len(restored._train_dataset))

    def test_pretrain_trainer_caps_steps_emits_metrics_and_rotates_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12", "13 14 15", "16 17 18"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            observed = []
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                seed=42, num_workers=0, verbose=False,
                grad_accum_steps=1,
                total_optimizer_steps=2,
                max_optimizer_steps=2,
                checkpoint_every_steps=1,
                checkpoint_slots=2,
                step_callback=observed.append,
            )

            metrics = trainer.train_epoch(1)

            self.assertEqual(trainer.global_step, 2)
            self.assertTrue(metrics["step_limit_reached"])
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[-1]["causal_targets_total"], 12.0)
            self.assertEqual(observed[-1]["sample_cursor"], 4.0)
            self.assertTrue((temporary / "checkpoints" / "latest-0").is_dir())
            self.assertTrue((temporary / "checkpoints" / "latest-1").is_dir())

    def test_indexed_evaluate_reports_causal_targets_and_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            PretrainSampleCompiler(IntegerTokenizer(), 4, root).compile(
                ["1 2 3", "4 5 6", "7 8 9", "10 11 12"]
            )
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            trainer = PretrainTrainer(
                model=TinyModel(), cfg=cfg, train_store_dir=root,
                tokenizer=IntegerTokenizer(), output_dir=temporary / "output",
                seed=42, num_workers=0, verbose=False,
                validation_sample_count=2,
            )

            metrics = trainer.evaluate(max_batches=1)

            self.assertEqual(metrics["sample_count"], 2)
            self.assertEqual(metrics["causal_targets"], 6)
            self.assertEqual(metrics["loss"], 1.0)

    def test_indexed_pretrain_source_identity_is_checked_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "samples"
            self.compile_fixture(root)
            cfg = SimpleNamespace(
                seq_len=4, batch_size=2, lr=0.01, weight_decay=0.0,
                warmup_steps=1, grad_clip=1.0, device="cpu", epochs=1,
                use_titans_memory=False, use_cca=False, max_new_tokens=1,
                temperature=1.0, top_k=0, top_p=1.0,
            )

            with self.assertRaisesRegex(ValueError, "Source identity mismatch"):
                PretrainTrainer(
                    model=TinyModel(), cfg=cfg, train_store_dir=root,
                    tokenizer=IntegerTokenizer(), output_dir=temporary / "checkpoints",
                    seed=42, num_workers=0, verbose=False,
                    pretrain_source={"fixture": "another-corpus"},
                )


if __name__ == "__main__":
    unittest.main()
