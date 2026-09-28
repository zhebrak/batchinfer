"""Sizing on a real card (Qwen3-0.6B, bf16, CUDA): the naive engine's fit and the batchinfer engine's auto pool are
measured on this card, verified by their probes, and stay within sizing's shares. Its own module, so its
near-full-card allocations are freed before anything else loads a model."""
import gc

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from batchinfer.naive import NaiveEngine  # noqa: E402
from batchinfer.executor import Executor  # noqa: E402
from batchinfer.kv import BLOCK_SIZE  # noqa: E402
from batchinfer.sizing import BUDGET_STEP, MAX_PEAK_SHARE, MIN_POOL_BLOCKS  # noqa: E402
from batchinfer.step_engine import StepEngine  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
GiB = 2**30


def free_all():
    gc.collect()
    torch.cuda.empty_cache()


def kv_bytes_per_token(config):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2  # K and V, bf16


def test_naive_fit_is_measured_and_verified():
    from transformers import AutoTokenizer
    engine = NaiveEngine(MODEL, AutoTokenizer.from_pretrained(MODEL))
    try:
        total = torch.cuda.mem_get_info()[1] / GiB
        fit = engine.fit()
        assert fit % BUDGET_STEP == 0 and engine.probed_tokens == fit
        facts = engine.memory_facts
        assert facts["max_batch_tokens_fit"] == fit and facts["in_use_after_load_gb"] > 1
        assert facts["padded_token_kib"] * 2**10 >= kv_bytes_per_token(engine.model.config)  # KV plus transients
        assert facts["row_length_bytes"] >= 0  # what a longer row adds per token: the padded mask
        assert facts["probe_peak_device_gb"] <= MAX_PEAK_SHARE * total  # the probe refuses a device peak above it
        hw = engine.hardware
        assert (hw.gpu_name, hw.max_batch_tokens_fit, hw.kv_pool_tokens) == (torch.cuda.get_device_name(), fit, None)
        engine.probe(BUDGET_STEP)  # an explicit budget is probed, not measured: the fit stays the fit
        assert engine.probed_tokens == BUDGET_STEP and engine.hardware.max_batch_tokens_fit == fit
    finally:
        del engine
        free_all()


def test_step_pool_is_sized_to_the_card():
    ex = Executor(MODEL, seed=3)  # reserve_gb="auto": measure the largest step, then size the pool
    try:
        total = torch.cuda.mem_get_info()[1] / GiB
        facts = ex.memory_facts
        assert ex.num_blocks >= MIN_POOL_BLOCKS and facts["step_activation_gb"] > 0 and facts["in_use_after_load_gb"] > 1
        # the verify probe is the run's own widest step and the pool is sized to put it at the limit: its device peak
        # (torch's reserved plus what the process holds outside torch) sits there, give or take the allocator's pages
        assert MAX_PEAK_SHARE * total - 1 < facts["probe_peak_device_gb"] <= MAX_PEAK_SHARE * total + 0.1
        hw = StepEngine(ex, None).hardware
        assert hw.kv_pool_tokens == (ex.num_blocks - 1) * BLOCK_SIZE and hw.max_batch_tokens_fit is None
    finally:
        del ex
        free_all()


@pytest.mark.parametrize("fused_layers", [False, True])
def test_step_pool_holds_the_graphs(fused_layers):
    """With cuda_graphs the sizing measures the largest graph's private pool and leaves room for it: the verify probe,
    run with every graph captured and alive, still lands within the limit."""
    ex = Executor(MODEL, seed=3, fused_layers=fused_layers, cuda_graphs=True)
    try:
        total = torch.cuda.mem_get_info()[1] / GiB
        facts = ex.memory_facts
        assert 0 < facts["graph_held_gb"] <= facts["graph_sized_gb"] + 0.1 and facts["graph_capture_s"] > 0 and len(ex.graphs) == len(ex.graph_sizes)
        assert MAX_PEAK_SHARE * total - 1 < facts["probe_peak_device_gb"] <= MAX_PEAK_SHARE * total + 0.1
    finally:
        del ex
        free_all()
