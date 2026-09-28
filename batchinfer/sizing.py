"""Sizing: turns what an engine measured on its card at load into a size. Pure Python, no torch: the engines
measure (naive.NaiveEngine.fit, executor.Executor), this module only computes, so the CPU tests cover it.

Nothing here names a model or a GPU. Every input is measured on the card the engine runs on:
- total_gb: the card's memory (torch's total), in GiB like every _gb here.
- in_use_gb: device memory held after the model loaded, before any probe: weights, the allocator, and what the
  process holds outside torch (CUDA context, library handles).
- the cost of one unit of what is sized: a padded token for the naive engine (measured by its fit probe), a KV
  block plus the largest step's activations for the batchinfer engine (measured by its widest-step probe).
A device peak is torch's peak reserved memory plus what the process holds outside torch.
"""
import math

# The most of the card a run may use, as a device peak. The naive engine's probe refuses above it, and so does its
# check of every group before the first forward. The batchinfer engine sizes its KV pool up to it: its probe runs the
# largest step the scheduler can build, which is the run's worst step (probe and run peaks agree to 0.01 GiB in
# every measured row), so that probe records its peak and fails only on OOM.
MAX_PEAK_SHARE = 0.95
# The naive engine's budget target. Its fit prices a padded token in rows of PROBE_ROW_TOKENS; a group's longer
# rows (the padded attention mask grows with row length, which fit() also measures) and decode's cache
# concatenation cost more, and the gap up to MAX_PEAK_SHARE is left for them.
NAIVE_PEAK_SHARE = 0.90
PROBE_ROW_TOKENS = 4096  # the naive probes' row length
BUDGET_STEP = PROBE_ROW_TOKENS  # the naive budget's granularity: one probe row
MIN_POOL_BLOCKS = 2  # the pad block and one block to write: the smallest pool a step can run on


def naive_fit_tokens(total_gb, in_use_gb, padded_token_gb):
    """The largest multiple of BUDGET_STEP padded tokens whose predicted device peak,
    in_use_gb + tokens x padded_token_gb, is at most NAIVE_PEAK_SHARE of total_gb."""
    room = NAIVE_PEAK_SHARE * total_gb - in_use_gb
    tokens = int(room / padded_token_gb) // BUDGET_STEP * BUDGET_STEP if room > 0 else 0
    if tokens < BUDGET_STEP:
        raise RuntimeError(f"no room for fixed groups: {in_use_gb:.1f} of {total_gb:.1f} GiB is in use after load, and "
                           f"{BUDGET_STEP} padded tokens need {BUDGET_STEP * padded_token_gb:.2f} GiB more under the "
                           f"{NAIVE_PEAK_SHARE:.0%} target")
    return tokens


def naive_group_peak_gb(in_use_gb, padded_token_gb, row_length_gb, rows, row_len, footprint):
    """The predicted device peak of one naive group: what is in use after load, its padded end-state footprint at
    the fitted cost per padded token, and, for rows longer than PROBE_ROW_TOKENS, what each of its rows x row_len
    prefill tokens adds per token of row length past the probe's (row_length_gb, measured by fit())."""
    return in_use_gb + footprint * padded_token_gb + rows * row_len * max(0, row_len - PROBE_ROW_TOKENS) * row_length_gb


def pool_blocks(total_gb, in_use_gb, activation_gb, block_gb, graph_gb=0.0):
    """KV pool blocks whose predicted device peak, in_use_gb + blocks x block_gb + activation_gb + graph_gb, is at most
    MAX_PEAK_SHARE of total_gb. activation_gb was measured at the widest step, an upper bound for every step a
    smaller pool allows, so any pool of at least MIN_POOL_BLOCKS runs. graph_gb: the captured CUDA graphs' private
    memory pool, held for the whole run on top of the eager steps' activations."""
    blocks = math.floor((MAX_PEAK_SHARE * total_gb - in_use_gb - activation_gb - graph_gb) / block_gb)
    if blocks < MIN_POOL_BLOCKS:
        raise RuntimeError(f"no room for a KV pool: {in_use_gb:.1f} of {total_gb:.1f} GiB is in use after load and the "
                           f"widest step needs {activation_gb + graph_gb:.2f} GiB more, leaving {max(blocks, 0)} blocks under the "
                           f"{MAX_PEAK_SHARE:.0%} limit; size it by hand with num_blocks= or reserve_gb=")
    return blocks
