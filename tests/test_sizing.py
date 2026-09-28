"""sizing: what the engines measured at load becomes a size under the stated shares, and a card with no room is refused."""
import pytest

from batchinfer.sizing import (BUDGET_STEP, MAX_PEAK_SHARE, MIN_POOL_BLOCKS, PROBE_ROW_TOKENS, NAIVE_PEAK_SHARE,
                               pool_blocks, naive_fit_tokens, naive_group_peak_gb)

# Qwen3-8B as measured (fit probe on the A100 and the H100): about 16 GiB in use after load and 263-270 KiB per
# padded token, whichever card.
IN_USE, TOKEN = 16.0, 262 / 2**20
A100, H100 = 39.39, 79.18
BLOCK = 2.25 / 1024  # Qwen3-8B: a 16-token block of bf16 KV is 2.25 MiB


def test_naive_fit_is_the_largest_step_under_the_target():
    for total in (A100, H100):
        t = naive_fit_tokens(total, IN_USE, TOKEN)
        assert t % BUDGET_STEP == 0
        assert IN_USE + t * TOKEN <= NAIVE_PEAK_SHARE * total < IN_USE + (t + BUDGET_STEP) * TOKEN
    # nothing names a card: the same model gets more than twice the budget on twice the memory
    assert naive_fit_tokens(H100, IN_USE, TOKEN) > 2 * naive_fit_tokens(A100, IN_USE, TOKEN)
    # and a smaller model (less in use, a cheaper token) gets more on the same card
    assert naive_fit_tokens(A100, 4.0, 170 / 2**20) > naive_fit_tokens(A100, IN_USE, TOKEN)


@pytest.mark.parametrize("total,in_use", [(24.0, 21.5), (16.0, 16.5)])
def test_naive_fit_refuses_when_the_model_leaves_no_room(total, in_use):
    with pytest.raises(RuntimeError, match="no room for fixed groups"):
        naive_fit_tokens(total, in_use, TOKEN)


def test_a_group_with_longer_rows_costs_more_than_the_fit_priced():
    row_length = 1 / 2**30  # one byte per padded token per token of row length past the probe's: HF's padded bool mask
    short = naive_group_peak_gb(IN_USE, TOKEN, row_length, rows=16, row_len=PROBE_ROW_TOKENS, footprint=70_000)
    assert short == pytest.approx(IN_USE + 70_000 * TOKEN)  # rows no longer than the probe's cost what the fit priced
    long = naive_group_peak_gb(IN_USE, TOKEN, row_length, rows=2, row_len=32_768, footprint=70_000)
    assert long == pytest.approx(short + 2 * 32_768 * (32_768 - PROBE_ROW_TOKENS) / 2**30)  # +1.75 GiB
    # 1.7B on the H100, the measured case: fit() measured 1.25 bytes, so 49 rows of 8,026 tokens are predicted to add
    # 1.8 GiB; the run peaked 1.29 GiB past the probe, so the check errs on the safe side
    predicted = naive_group_peak_gb(0, 0, 1.25 * row_length, 49, 8026, 0)
    assert predicted == pytest.approx(1.80, abs=0.01) and predicted > 1.29


def test_pool_blocks_stop_at_the_limit():
    in_use, activation = 16.4, 1.9
    n = pool_blocks(H100, in_use, activation, BLOCK)
    assert in_use + n * BLOCK + activation <= MAX_PEAK_SHARE * H100 < in_use + (n + 1) * BLOCK + activation


def test_a_small_card_gets_a_small_pool_not_a_refusal():
    """8B on a 24 GB card (22.49 GiB): what fits after load and the widest step's activations is ~1,600 blocks, fewer
    than the widest step's probe uses; the activations were measured at the widest step, so any pool runs."""
    n = pool_blocks(22.49, 15.8, 2.0, BLOCK)
    assert MIN_POOL_BLOCKS <= n < 2049
    with pytest.raises(RuntimeError, match="num_blocks= or reserve_gb="):
        pool_blocks(16.0, 15.3, 1.0, BLOCK)  # the model barely fits: no room for even the pad block and one more
