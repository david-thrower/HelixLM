"""
HelixLM: Recurrent Heterogeneous Graph Neural Language Model

An optimized hybrid architecture combining biological brain-inspired random graph wiring,
heterogeneous neural columns, hybrid linear/full attention, Mamba-2 SSD, and recurrent depth.

Designed for hyperpersonalization and on-device AI.
"""
from .config import HelixConfig
from .tokenizer import HelixTokenizer
from .model import HelixLMCore
from .hf_model import HelixForCausalLM
from .trainer import Trainer
from .dataset import (
    HelixDataset, HelixDatasetFromTokens, HelixHFDataset,
    DocumentAwareDataset, create_helix_dataloader, create_document_loader,
)

__version__ = "0.1.0"
__all__ = [
    "HelixConfig",
    "HelixTokenizer",
    "HelixLMCore",
    "HelixForCausalLM",
    "Trainer",
    "HelixDataset",
    "HelixDatasetFromTokens",
    "HelixHFDataset",
    "DocumentAwareDataset",
    "create_helix_dataloader",
    "create_document_loader",
]
