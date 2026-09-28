"""Fused layers' model check, on configs alone (CPU): Qwen3 as released passes, anything else is refused by name."""
import pytest

pytest.importorskip("transformers")

from transformers import LlamaConfig, Qwen3Config  # noqa: E402

from batchinfer.qwen3_fused import check_fusable  # noqa: E402


def test_qwen3_passes():
    check_fusable(Qwen3Config())


def test_other_models_and_variants_are_refused_by_name():
    with pytest.raises(ValueError, match="model_type 'llama'"):
        check_fusable(LlamaConfig())
    with pytest.raises(ValueError, match="sliding-window"):
        check_fusable(Qwen3Config(use_sliding_window=True, sliding_window=4096, max_window_layers=0))
    with pytest.raises(ValueError, match="attention_bias"):
        check_fusable(Qwen3Config(attention_bias=True))
    with pytest.raises(ValueError, match="head_dim 96"):
        check_fusable(Qwen3Config(head_dim=96))
