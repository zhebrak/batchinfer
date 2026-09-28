"""The naive baseline: an HF causal LM forward over fixed, left-padded batches with HF's DynamicCache, and a plain
prefill + decode loop. By default in arrival order (order=input), which is what a first implementation does; each
group decodes until its longest request finishes. It runs a Policy's fixed groups in order and never reorders,
regroups or re-splits; it checks that each group fits the policy's budget and raises otherwise. It was this project's
v0 and stays as the "before" row for every number.

Known costs, recorded rather than fixed:
- when any row in a group is padded, HF's SDPA path gets a materialised mask, drops the GQA fast
  path and calls repeat_kv: each decode step copies and reads num_heads/num_kv_heads times the
  live KV (4x on Qwen3). `padded_groups` in the stats says how many groups pay this;
- DynamicCache appends with torch.cat per layer per step, one more copy of the live KV.
Both are HF artefacts the batchinfer engine's paged KV removes.
"""
from collections import Counter

from .model import load_model, stop_token_ids  # before torch: sets the CUDA allocator config

import torch  # noqa: E402

from .metrics import FLAT_KEYS  # noqa: E402
from .policy import padded_tokens  # noqa: E402
from .schema import Hardware, Result  # noqa: E402
from .sizing import MAX_PEAK_SHARE, PROBE_ROW_TOKENS, naive_fit_tokens, naive_group_peak_gb  # noqa: E402

GiB = 2**30
# fit() prefills this many padded tokens twice: in rows of PROBE_ROW_TOKENS, for the cost of a padded token, and in
# one row, for what a longer row adds per token (the padded attention mask grows with row length).
FIT_PROBE_TOKENS = 2 * PROBE_ROW_TOKENS


class NaiveEngine:
    flat_keys = FLAT_KEYS  # the report columns this engine fills

    def __init__(self, model_name, tokenizer, *, dtype="bfloat16", attn="sdpa", device=None, model=None):
        """The tokenizer is only used to decode finished sequences; analysis owns encoding."""
        self.model_name, self.tokenizer, self.dtype, self.attn = model_name, tokenizer, dtype, attn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cuda = self.device.type == "cuda"
        if model is None:
            model = load_model(model_name, getattr(torch, dtype), attn)
        self.model = model.to(self.device).eval()
        self.eos = torch.tensor(stop_token_ids(self.model), dtype=torch.long, device=self.device)
        self.max_positions = getattr(self.model.config, "max_position_embeddings", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = int(pad) if pad is not None else int(self.eos[0])
        self.probed_tokens = None  # the largest budget probe() accepted; run() refuses a policy above it
        self.probe_peak_reserved_gb = None  # run() records it, as the batchinfer engine records its executor's
        self.max_batch_tokens_fit = None  # what fit() measured on this card (Hardware); None without a fit
        self.memory_facts = {}  # what fit() and probe() measured; run() records them in metrics.memory
        # fit()'s costs in GiB: (in use after load, per padded token, per padded token and row token past the probe's,
        # the card's total); None without a fit
        self._cost = None

    # -------------------------------------------------------------------------------------------------

    @property
    def hardware(self):
        """What this engine measured on its card (schema.Hardware), for policy.decide. All None off the GPU."""
        if not self.cuda:
            return Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=None)
        return Hardware(gpu_name=torch.cuda.get_device_name(), total_gb=round(torch.cuda.mem_get_info()[1] / GiB, 2),
                        max_batch_tokens_fit=self.max_batch_tokens_fit, kv_pool_tokens=None)

    def _prefill_peak(self, tokens, row_tokens=PROBE_ROW_TOKENS):
        """One padded prefill of `tokens` tokens in rows of row_tokens. One pad in the mask sends HF down the same
        masked (repeat_kv) path real padded groups take. Returns (torch's peak reserved, what the process holds
        outside torch, the card's total), in bytes."""
        L = min(row_tokens, tokens)
        B = max(1, tokens // L)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            ids = torch.full((B, L), self.pad_id, dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            mask[0, 0] = 0
            self.model(input_ids=ids, attention_mask=mask, past_key_values=self._new_cache(), use_cache=True,
                       logits_to_keep=1)
            del ids, mask
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        outside = total - free - torch.cuda.memory_reserved()  # CUDA context and library handles
        torch.cuda.empty_cache()  # release the probe's blocks, or every later peak reads as the probe
        torch.cuda.reset_peak_memory_stats()
        return peak, outside, total

    def fit(self):
        """max_batch_tokens="auto": measure how many padded tokens fit on this card for this model, then verify it.
        A prefill of FIT_PROBE_TOKENS in rows of PROBE_ROW_TOKENS gives the cost of one padded token (its KV plus
        prefill transients) on top of what is in use after load; the same tokens in one row give what each token of
        row length past PROBE_ROW_TOKENS adds per padded token, which run() uses to check groups with longer rows.
        sizing.naive_fit_tokens turns the first into a budget and probe() runs at it. Returns the budget (None off
        the GPU, where nothing is measured and policy uses its no-card default)."""
        if not self.cuda:
            return None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        loaded = torch.cuda.memory_reserved()
        peak, outside, total = self._prefill_peak(FIT_PROBE_TOKENS)
        long_peak, _, _ = self._prefill_peak(FIT_PROBE_TOKENS, row_tokens=FIT_PROBE_TOKENS)
        token_bytes = (peak - loaded) / FIT_PROBE_TOKENS
        row_length_bytes = max(0, long_peak - peak) / (FIT_PROBE_TOKENS * (FIT_PROBE_TOKENS - PROBE_ROW_TOKENS))
        in_use = loaded + outside
        tokens = naive_fit_tokens(total / GiB, in_use / GiB, token_bytes / GiB)
        self.probe(tokens)
        self.max_batch_tokens_fit = tokens
        self._cost = (in_use / GiB, token_bytes / GiB, row_length_bytes / GiB, total / GiB)
        self.memory_facts.update(in_use_after_load_gb=round(in_use / GiB, 2), padded_token_kib=round(token_bytes / 2**10, 1),
                                 row_length_bytes=round(row_length_bytes, 3), max_batch_tokens_fit=tokens)
        return tokens

    def probe(self, max_batch_tokens):
        """One prefill at the full token budget before any real work, so the prefill transient at that budget is
        exercised against this card. Refuses a device peak above sizing.MAX_PEAK_SHARE of the card. Keeps and
        returns torch's peak reserved GiB (None off the GPU, where it only records the budget for run's check)."""
        if self.cuda:
            peak, outside, total = self._prefill_peak(max_batch_tokens)
            if peak + outside > MAX_PEAK_SHARE * total:
                raise RuntimeError(f"memory probe at {max_batch_tokens} padded tokens peaked at "
                                   f"{(peak + outside) / GiB:.1f} of {total / GiB:.1f} GiB, over the "
                                   f"{MAX_PEAK_SHARE:.0%} limit; lower --max-batch-tokens")
            self.probe_peak_reserved_gb = round(peak / GiB, 2)
            self.memory_facts["probe_peak_device_gb"] = round((peak + outside) / GiB, 2)
        self.probed_tokens = max_batch_tokens
        return self.probe_peak_reserved_gb

    def run(self, job, policy, on_result, metrics):
        """Runs policy.groups in order, calling on_result with each request's Result as its group finishes.
        The policy must be decided for this engine: PolicyConfig.validate() then guarantees fixed groups and no
        prefix sharing."""
        if policy.engine != "naive":
            raise ValueError(f"the naive engine runs a policy decided for engine='naive', not {policy.engine!r}")
        items, cuda = job.requests, self.cuda
        if self.probed_tokens is not None and policy.max_batch_tokens > self.probed_tokens:
            raise ValueError(f"policy budget {policy.max_batch_tokens} tokens exceeds the probed "
                             f"{self.probed_tokens}; probe at the larger budget first")
        metrics.flat_keys = self.flat_keys
        if cuda:
            metrics.sync = torch.cuda.synchronize
            metrics.memory.setdefault("gpu_name", torch.cuda.get_device_name())
            metrics.memory["probe_peak_reserved_gb"] = self.probe_peak_reserved_gb
            metrics.memory.update(self.memory_facts)
        for gi, group in enumerate(policy.groups):  # every group before the first forward, not after earlier ones ran
            self._check(gi, group, items, policy, self.max_positions)
            self._check_memory(gi, group, items)
        peak_alloc = peak_reserved = 0
        with metrics.timer("inference") as inference:
            for gi, group in enumerate(policy.groups):
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                self._run_group(gi, group, items, on_result, metrics, inference.t0)
                if cuda:
                    reserved, allocated = torch.cuda.max_memory_reserved(), torch.cuda.max_memory_allocated()
                    peak_alloc, peak_reserved = max(peak_alloc, allocated), max(peak_reserved, reserved)
                    metrics.groups[-1]["peak_mem_reserved_gb"] = round(reserved / GiB, 2)
                    metrics.groups[-1]["peak_mem_allocated_gb"] = round(allocated / GiB, 2)
        if cuda:
            metrics.memory.update(peak_mem_allocated_gb=round(peak_alloc / GiB, 2),
                                  peak_mem_reserved_gb=round(peak_reserved / GiB, 2))

    # -------------------------------------------------------------------------------------------------

    @staticmethod
    def _check(gi, group, items, policy, max_positions=None):
        n, footprint = len(group.members), padded_tokens(group.members, items)
        # a lone request too: the probe ran at max_batch_tokens, so nothing past it has been exercised on this card
        if n > policy.max_batch_size or footprint > policy.max_batch_tokens:
            raise ValueError(f"group {gi} breaks the policy budget ({n} rows, {footprint} padded tokens vs "
                             f"{policy.max_batch_size} rows, {policy.max_batch_tokens} tokens); the engine never re-splits, "
                             f"so pass a larger max_batch_tokens")
        # Left padding keeps each row's real positions at 0 .. prompt_len + max_tokens - 2, as in the batchinfer engine
        for i in group.members if max_positions is not None else ():
            it = items[i]
            if it.prompt_len + it.req.max_tokens - 1 > max_positions:
                raise ValueError(f"request {it.req.id} needs {it.prompt_len + it.req.max_tokens - 1} positions; "
                                 f"the model has {max_positions}")

    def _check_memory(self, gi, group, items):
        """With a fit on this card, refuse a group predicted to peak above sizing.MAX_PEAK_SHARE of it. The fit priced
        padded tokens in rows of PROBE_ROW_TOKENS; a group whose rows are longer costs more by what fit() measured per
        token of row length (sizing.naive_group_peak_gb). Checked before the first forward, so a job with prompts too
        long for the fitted budget fails at the start, naming the group, rather than after earlier groups ran."""
        if self._cost is None:
            return
        in_use, token, row_length, total = self._cost
        rows, row_len = len(group.members), max(items[i].prompt_len for i in group.members)
        peak = naive_group_peak_gb(in_use, token, row_length, rows, row_len, padded_tokens(group.members, items))
        if peak > MAX_PEAK_SHARE * total:
            raise ValueError(f"group {gi} ({rows} rows of up to {row_len} prompt tokens) is predicted to peak at "
                             f"{peak:.1f} of {total:.1f} GiB, over the {MAX_PEAK_SHARE:.0%} limit: its rows are longer "
                             f"than the fit's {PROBE_ROW_TOKENS}-token probe rows; pass a smaller max_batch_tokens")

    def _new_cache(self):
        from transformers import DynamicCache
        try:
            return DynamicCache(config=self.model.config)
        except TypeError:
            return DynamicCache()

    @torch.inference_mode()
    def _run_group(self, gi, group, items, on_result, metrics, t0):
        """t0: when the inference timer started, the origin of the group's start_ms and its requests' *_ms."""
        rows = [items[i] for i in group.members]
        dev = self.device
        B, L, T = len(rows), max(r.prompt_len for r in rows), max(r.req.max_tokens for r in rows)
        prompt_len = torch.tensor([r.prompt_len for r in rows], dtype=torch.long, device=dev)
        max_new = torch.tensor([r.req.max_tokens for r in rows], dtype=torch.long, device=dev)
        ignore_eos = torch.tensor([r.req.ignore_eos for r in rows], dtype=torch.bool, device=dev)
        all_ignore = bool(ignore_eos.all())

        # Left-padded prompts. The mask is preallocated for the whole group and grows by a view per
        # step: HF sizes its causal mask from cache length + query length and indexes this directly.
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((B, L + T - 1), dtype=torch.long, device=dev)
        for b, r in enumerate(rows):
            input_ids[b, L - r.prompt_len:] = torch.tensor(r.token_ids, dtype=torch.long, device=dev)
            mask[b, L - r.prompt_len:L] = 1

        pad = torch.full((B,), self.pad_id, dtype=torch.long, device=dev)
        generated = torch.full((B, T), self.pad_id, dtype=torch.long, device=dev)
        n_out = torch.zeros(B, dtype=torch.long, device=dev)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        live_slots = torch.zeros((), dtype=torch.long, device=dev)
        steps = 0

        with metrics.timer("prefill") as prefill:
            # Real tokens get positions 0..prompt_len-1 regardless of padding; pads are clamped to 0 and masked.
            position_ids = (mask[:, :L].cumsum(-1) - 1).clamp(min=0)
            cache = self._new_cache()
            out = self.model(input_ids=input_ids, attention_mask=mask[:, :L], position_ids=position_ids,
                             past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1, :].argmax(-1)  # g_1 for every row; the cache holds slots 0..L-1

        with metrics.timer("decode") as decode:
            for k in range(1, T + 1):
                # g_k is the k-th generated token. Alive rows keep it and finish on EOS (unless
                # ignore_eos) or when they have produced their own max_tokens.
                generated[:, k - 1] = torch.where(alive, nxt, pad)
                n_out += alive
                alive &= ~((torch.isin(nxt, self.eos) & ~ignore_eos) | (n_out >= max_new))
                if k == T or (not all_ignore and not bool(alive.any())):
                    break
                # Decode step k feeds g_k at cache slot s = L + k - 1, sequence position prompt_len + k - 1.
                # Finished rows are dead slots: they are fed the pad id and their outputs stay frozen.
                s = L + k - 1
                mask[:, s] = 1
                out = self.model(input_ids=torch.where(alive, nxt, pad)[:, None], attention_mask=mask[:, :s + 1],
                                 position_ids=(prompt_len + (k - 1))[:, None], past_key_values=cache, use_cache=True)
                nxt = out.logits[:, -1, :].argmax(-1)
                steps += 1
                live_slots += alive.sum()
        del cache, out

        n_out_l, gen_l, live = n_out.tolist(), generated.tolist(), int(live_slots)
        eos = set(self.eos.tolist())
        finish = Counter()
        start_ms = 1000 * (prefill.t0 - t0)  # the group's compute starts with its prefill
        for b, r in enumerate(rows):
            ids = gen_l[b][:n_out_l[b]]
            reason = "stop" if (ids and ids[-1] in eos and not r.req.ignore_eos) else "length"
            finish[reason] += 1
            # g_1 came from prefill; g_n from decode step n-1. Rows in a static batch complete together,
            # so this is the best per-request figure this engine has.
            latency = prefill.elapsed + (decode.elapsed * (len(ids) - 1) / steps if steps else 0.0)
            on_result(Result(id=r.req.id, text=self.tokenizer.decode(ids, skip_special_tokens=True), token_ids=ids,
                             prompt_tokens=r.prompt_len, output_tokens=len(ids), finish_reason=reason,
                             group=gi, latency_s=round(latency, 4)))
            # every row is admitted and prefilled with its group; the finish is apportioned as latency_s is
            metrics.record_request(
                idx=group.members[b], id=r.req.id, analysis_kind=r.kind, group=gi, prompt_len=r.prompt_len,
                max_tokens=r.req.max_tokens, output_tokens=len(ids), prefix_hit_tokens=0, finish_reason=reason,
                admitted_step=None, prefill_start_step=None, first_token_step=None, finished_step=None,
                admitted_ms=round(start_ms, 3), prefill_start_ms=round(start_ms, 3),
                first_token_ms=round(start_ms + 1000 * prefill.elapsed, 3),
                finished_ms=round(start_ms + 1000 * latency, 3))
        metrics.add(requests=B, prompt_tokens=sum(r.prompt_len for r in rows), padded_prompt_tokens=B * L,
                    output_tokens=sum(n_out_l), decode_steps=steps, decode_slot_steps=steps * B,
                    live_slot_steps=live, finish_stop=finish["stop"], finish_length=finish["length"])
        metrics.add_group(group=gi, n=B, kinds=dict(Counter(r.kind for r in rows)),
                          padded=len({r.prompt_len for r in rows}) > 1, padded_len=L, T=T,
                          prompt_tokens=sum(r.prompt_len for r in rows), start_ms=round(start_ms, 3),
                          shared_prefix_len=group.shared_prefix_len, prefill_s=round(prefill.elapsed, 4),
                          decode_steps=steps, decode_s=round(decode.elapsed, 4), wasted_slots=steps * B - live)
