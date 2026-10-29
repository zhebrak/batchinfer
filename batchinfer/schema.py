"""The records handed between the stages of one job, in flow order (flow.run chains them; README "How it works"):

    list[Request] --analysis.analyze--> JobAnalysis --policy.decide(job, PolicyConfig, Hardware)--> Policy
    JobAnalysis + Policy --engine.run(job, policy, on_result, metrics)--> one Result per request
    inside the batchinfer engine, every step: Scheduler.next_step() -> Step --Executor.forward--> sampled ids
                                        --Scheduler.commit--> finished sequences

Who owns what:
- io is the only stage that reads a JSONL or builds a Request; a Request carries exactly io.CLIENT_FIELDS.
- analysis is the only stage that encodes (facts and metrics, no decisions); engines are the only stages that decode.
- policy is the only stage that reads a PolicyConfig; everything downstream reads the decisions through Policy.
  PolicyConfig.validate() is the one place that refuses a config an engine cannot run. Hardware reaches policy
  only as the Hardware record the engine measured on its card at load (sizing.py); policy itself never calls CUDA.
- Metrics sections: analysis and policy are set by flow.run, meta by the CLI; counts, memory, the per-group or
  per-step trace and the per-request trace by the engine that ran. Each timing key is written by whoever runs that
  phase: load and probe (the CLI), analysis and policy (flow.run), inference, prefill and decode (the engine).
"""
from dataclasses import asdict, dataclass, field

from .prefix import PrefixTrie

# batchinfer: ours (step_engine.StepEngine: paged KV, packed steps, continuous admission, prefix sharing).
# naive: the baseline (naive.NaiveEngine: fixed groups as left-padded batches over HF's DynamicCache).
ENGINES = ("batchinfer", "naive")
ORDERS = ("input", "max_tokens_desc", "decode_ratio_desc", "prefix_dfs")
NAIVE_ORDER = "input"  # the naive engine's default: arrival order, what a first implementation does
ADMISSIONS = ("groups", "continuous")
# How the batchinfer engine holds KV for a request (scheduler.py, rule 4). reserve: its whole life's blocks at
# admission, never preempted. grow: the prompt's blocks at admission, a decode block as decode reaches it, and
# admission against a projection of what every live request can hold at every future step (every max_tokens is
# known up front), so the pool never runs dry and nothing is preempted either. preempt: grow's allocation under a
# watermark, the youngest request preempted and recomputed when the pool runs dry anyway.
MEMORY = ("reserve", "grow", "preempt")
# How the batchinfer engine shares the KV of common prompt prefixes (scheduler.py, rule 3). none: every block is
# private. block: whole blocks through the job's prefix trie, computed once and read by every request that has them.
# token: block, and the leading tokens a request's own block shares with a sibling block already computed are copied
# from it rather than recomputed (token-level sharing on whole-block pages, the way a radix tree shares prefixes).
SHARING = ("none", "block", "token")
# What happens to a shared block when its last user in a job finishes (scheduler.py, rule 3). job: released. radix:
# kept, keyed by its prefix, for the next job the same engine runs (a long-running batch service, the bench's warm
# row), evicted oldest first when a job needs the room: the cache half of a radix tree. One job alone gains nothing.
CACHE = ("job", "radix")
# Prefill budget bounds for "auto" and for explicit values.
# 256: a bf16 GEMM over M tokens does about M FLOP per weight byte, so a step with fewer prefill tokens than the
#      card's FLOP-per-byte ratio (its ridge: metrics.ridge_flop_per_byte, 200 to 300 on the cards listed there)
#      stays bound by reading the weights; 256 is a power of two in that range. The floor is not per card: on
#      mixed-quick, budgets of 256, 583 and 16384 all ran the same wall because the job is bound by its longest
#      decode chain. The per-card number that matters is the measured one, metrics.ridge_tokens
#      (scripts/ridge_calibration.py).
# 16384: the prefill part of the executor's worst-step memory probe, so activation memory at this budget is checked.
MIN_PREFILL_BUDGET, MAX_PREFILL_BUDGET = 256, 16384
# The widest step the batchinfer engine builds, and what the executor's memory probe sizes for: MAX_SEQS sampled decode
# rows (vLLM's default cap on running requests on 80 GB cards, so the comparison is like for like; the pool, not this
# cap, bounds admission in practice) plus a MAX_PREFILL_BUDGET-token prefill chunk.
MAX_SEQS = 1024
# The fixed-groups budget when there is no card to measure (analyze, CPU runs). On a card, max_batch_tokens="auto"
# is what the engine measured there (Hardware); it never falls back to this.
DEFAULT_MAX_BATCH_TOKENS = 65536
DEFAULT_BATCH_SIZE = 64


# io -> analysis -------------------------------------------------------------------------------------

@dataclass
class Request:
    """Client view of one prompt. Nothing else ever reaches analysis or the engine."""
    id: str
    prompt: str
    max_tokens: int
    ignore_eos: bool = False
    labels: list[str] | None = None


# analysis -> policy, engine -----------------------------------------------------------------------------

@dataclass
class AnalyzedRequest:
    """A request after analysis: tokenized once, characterised. Facts only, no decisions."""
    req: Request
    token_ids: list[int]
    prompt_len: int
    label_ids: list[list[int]] | None
    kind: str  # "prefill_only" (max_tokens == 1) | "label" (labels given) | "generate"
    decode_ratio: float  # (max_tokens - 1) / prompt_len: an upper bound on decode forwards per prefill token


@dataclass
class JobAnalysis:
    """What analysis knows about the whole job before anything runs."""
    requests: list[AnalyzedRequest]  # input order
    prefix: PrefixTrie  # the prompts as a trie of KV blocks: which requests share which blocks. Read-only
    tokenizer: str
    metrics: dict  # flat scalars describing the job; see analysis.job_metrics


# engine (measured at load) -> policy ----------------------------------------------------------------

@dataclass(frozen=True)
class Hardware:
    """What an engine measured where it runs, at load, before any job: policy.decide reads it next to the job. None
    means not measured or not applicable. The two budgets are different quantities and never share a field."""
    gpu_name: str | None  # torch.cuda.get_device_name(); None off the GPU
    total_gb: float | None  # the card's memory, GiB; None off the GPU
    max_batch_tokens_fit: int | None  # naive engine: the padded-token budget NaiveEngine.fit() measured on this card
    kv_pool_tokens: int | None  # batchinfer engine: its KV pool in token slots, (num_blocks - 1) x block_size
    kv_block_size: int | None = None  # batchinfer engine: tokens per KV block (Limits.block_size)
    ridge_tokens: int | None = None  # batchinfer engine: the card's measured ridge (Limits.ridge_tokens), None uncalibrated


@dataclass(frozen=True)
class Limits:
    """What the executor built, published once (Executor.build) for the scheduler: the pool (num_blocks counts the pad
    block), its block size, the caps the probe sized the step for, the model's position limit, and the graphs it
    captured: decode-only row counts and mixed-step prefill token sizes (empty without them). The scheduler reads
    nothing else of the executor."""
    num_blocks: int
    block_size: int
    max_seqs: int
    max_prefill: int
    max_positions: int | None
    graph_rows: tuple = ()  # decode-only graph row sizes
    graph_tokens: tuple = ()  # prefill token sizes of the mixed graphs (graphs.MIXED_TOKENS); empty without them
    ridge_tokens: int | None = None  # the prefill tokens a graphed decode step carries at about no cost on this card
    kv_dtype: str = "bf16"  # the KV pool's dtype: bf16 (the model's) or fp8
    overlap: bool = False  # the engine plans and launches a step while the previous one runs (Scheduler, overlap)
    attention: str = "fa"  # the executor's attention: fa, torch, or cascade (then the scheduler groups each step's rows)
    pool: str = "static"  # static, elastic, or contiguous (then the scheduler hands each admitted sequence a slot)
    blocks_per_group: int | None = None  # elastic: the blocks that share one mapping (kv.BlockAllocator counts live blocks by it)


# config -> policy ------------------------------------------------------------------------------------

def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)  # True is an int in Python; never a budget


@dataclass(frozen=True)
class PolicyConfig:
    """What the user asked for, for one engine. Only policy.decide reads it. The defaults are the batchinfer engine's;
    a naive-engine config names admission="groups" and sharing="none" (the CLI and NaiveBackend do)."""
    engine: str = "batchinfer"  # the engine that will run the decisions; see validate() for what "naive" allows
    order: str = "prefix_dfs"
    admission: str = "continuous"
    chunk_prefill: bool = True  # batchinfer engine only
    # Batchinfer engine only: an int in [1, MAX_PREFILL_BUDGET], "auto" (policy.auto_prefill_budget, fixed for the job) or
    # "ridge" (each step with decode rows carries the card's ridge_tokens of prefill, what a graphed decode step
    # absorbs at about no cost, measured per card by scripts/ridge_calibration.py and read from metrics.CARDS; a card
    # without a measurement runs the default budget and warns; scheduler.ridge_budget). The
    # default is MAX_PREFILL_BUDGET, the widest step the engine sizes for: with fused layers and CUDA graphs the step
    # is GPU-bound, so fewer and larger prefill steps cost less, and the decode-only steps they leave replay a graph.
    prefill_budget: int | str = MAX_PREFILL_BUDGET
    # Fixed-groups partition (admission="groups"): an int, or "auto" for what the engine measured on its card
    # (policy.auto_max_batch_tokens), DEFAULT_MAX_BATCH_TOKENS when there is no card.
    max_batch_tokens: int | str = "auto"
    max_batch_size: int = DEFAULT_BATCH_SIZE
    # Batchinfer engine only: compute each shared prompt block once and read it from every request that has it
    # (scheduler.py, rule 3; SHARING). On by default with order=prefix_dfs: on mixed-quick sharing took the timed
    # pass from 42.6 to 27.7 s at the block-level reuse ceiling, and prefix_dfs holds no shared block for requests
    # far down the order (pinned peak 1 block against ~1,200 under decode_ratio_desc, a wall-time tie otherwise).
    # Any order is allowed with sharing; a shuffled one delays the long generations and can pin whole subtrees.
    sharing: str = "block"  # batchinfer engine only: one of SHARING
    cache: str = "job"  # batchinfer engine only: one of CACHE
    memory: str = "reserve"  # batchinfer engine only, continuous admission only: one of MEMORY
    # memory=preempt: the steps of decode growth admission leaves room for (scheduler._watermark). At 32 a CPU
    # replay of the capped dec50x32 row halved the preemptions and cut the recompute by a third (bench.suite's
    # "capped preempt lookahead 32" arm measures it)
    watermark_steps: int = 1

    def validate(self):
        """The one place a config an engine cannot run is refused, before anything loads."""
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, not {self.engine!r}")
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, not {self.order!r}")
        if self.admission not in ADMISSIONS:
            raise ValueError(f"admission must be one of {ADMISSIONS}, not {self.admission!r}")
        if self.memory not in MEMORY:
            raise ValueError(f"memory must be one of {MEMORY}, not {self.memory!r}")
        if not isinstance(self.watermark_steps, int) or isinstance(self.watermark_steps, bool) or self.watermark_steps < 1:
            raise ValueError(f"watermark_steps must be an int of at least 1, not {self.watermark_steps!r}")
        if self.memory != "reserve" and self.admission != "continuous":
            raise ValueError(f"memory={self.memory!r} grows a request's KV as it decodes, which fixed groups (admitted "
                             f"whole with their lives' blocks) do not: it needs admission='continuous'")
        if not isinstance(self.chunk_prefill, bool):  # bool('off') is True: never coerce, refuse
            raise ValueError(f"chunk_prefill must be a bool, not {self.chunk_prefill!r} (bench --opt takes true/false or on/off)")
        if self.sharing not in SHARING:
            raise ValueError(f"sharing must be one of {SHARING}, not {self.sharing!r}")
        if self.cache not in CACHE:
            raise ValueError(f"cache must be one of {CACHE}, not {self.cache!r}")
        if self.cache == "radix" and self.sharing == "none":
            raise ValueError("cache='radix' keeps shared blocks for the next job: it needs sharing")
        if self.engine == "naive":
            if self.admission != "groups":
                raise ValueError("the naive engine runs fixed groups only: admission must be 'groups'")
            if self.cache != "job":
                raise ValueError("the naive engine keeps no KV between jobs: cache needs the batchinfer engine")
            if self.sharing != "none":
                raise ValueError("the naive engine computes every prompt token: sharing needs the batchinfer engine")
            if self.memory != "reserve":
                raise ValueError("the naive engine holds a group's KV for the group's life: memory needs the "
                                 "batchinfer engine")
            # "auto", "ridge" and the default leave the budget to the engine, and this engine has none (decide
            # records None); like chunk_prefill, only a setting that differs from the default is refused
            if self.prefill_budget not in ("auto", "ridge", MAX_PREFILL_BUDGET) or not self.chunk_prefill:
                raise ValueError("the naive engine prefills each group in one forward: a fixed prefill_budget and "
                                 "chunk_prefill are batchinfer-engine settings")
        if self.prefill_budget not in ("auto", "ridge") and not (_is_int(self.prefill_budget)
                                                                 and 1 <= self.prefill_budget <= MAX_PREFILL_BUDGET):
            raise ValueError(f"prefill_budget must be 'auto', 'ridge' or an int in [1, {MAX_PREFILL_BUDGET}], "
                             f"not {self.prefill_budget!r}")
        if self.max_batch_tokens != "auto" and not (_is_int(self.max_batch_tokens) and self.max_batch_tokens >= 1):
            raise ValueError(f"max_batch_tokens must be 'auto' or a positive int, not {self.max_batch_tokens!r}")
        if not (_is_int(self.max_batch_size) and self.max_batch_size >= 1):
            raise ValueError(f"max_batch_size must be a positive int, not {self.max_batch_size!r}")
        return self


# policy -> engine ------------------------------------------------------------------------------------

@dataclass
class Group:
    """Requests that run together: every member is admitted at once and they are co-resident. The naive engine runs
    a group as one padded batch and computes every prompt token of every member; shared_prefix_len measures what
    it could have shared. The batchinfer engine admits a group whole and shares whole KV blocks through the job's prefix
    trie, within and across groups, when the policy turns sharing on."""
    members: list[int]  # indices into JobAnalysis.requests
    shared_prefix_len: int  # longest common token prefix over members (0 for a single member)


@dataclass(frozen=True)
class Policy:
    """What policy decided for this job and this engine, once; policy.decide is its one constructor. Every setting
    has one field, and a setting the engine does not apply is None, so a recorded policy never shows a knob that
    played no part: the naive engine has no chunk_prefill or prefill_budget, and max_batch_tokens and
    max_batch_size exist only under admission="groups". Metrics records to_dict()."""
    engine: str  # the engine these decisions are for, one of ENGINES
    order: str  # the order's name, one of ORDERS, as asked
    admission_order: tuple[int, ...]  # indices into JobAnalysis.requests; the groups flattened when groups is set
    groups: tuple[Group, ...] | None  # fixed groups in execution order when admission == "groups", else None
    admission: str
    chunk_prefill: bool | None  # batchinfer engine only
    prefill_budget: int | None  # batchinfer engine only: maximum prefill tokens per step (decode rows ride on top), decided
    #                             before the first step; under ridge, the cap of a step without decode rows
    prefill_budget_auto: bool | None  # batchinfer engine only: prefill_budget came from policy.auto_prefill_budget
    prefill_budget_ridge: bool | None  # batchinfer engine only: steps with decode rows are filled to ridge_tokens
    ridge_tokens: int | None  # batchinfer engine only: the card's measured ridge, when prefill_budget_ridge
    sharing: str | None  # batchinfer engine only: one of SHARING
    cache: str | None  # batchinfer engine only: one of CACHE
    memory: str | None  # batchinfer engine only: one of MEMORY
    watermark_steps: int | None  # memory=preempt: steps of decode growth admission leaves room for
    max_batch_tokens: int | None  # fixed-groups footprint budget, rows x (longest prompt + longest max_tokens)
    max_batch_tokens_auto: bool | None  # max_batch_tokens came from policy.auto_max_batch_tokens (groups only)
    max_batch_size: int | None
    stats: dict  # facts about the groups decision (policy.group_stats); {} under continuous admission

    def to_dict(self):
        """What metrics.json records as its policy section: scalars, plus the head of the admission order."""
        return {"engine": self.engine, "order": self.order, "admission": self.admission,
                "chunk_prefill": self.chunk_prefill, "prefill_budget": self.prefill_budget,
                "prefill_budget_auto": self.prefill_budget_auto, "prefill_budget_ridge": self.prefill_budget_ridge,
                "ridge_tokens": self.ridge_tokens,
                "sharing": self.sharing, "cache": self.cache, "memory": self.memory, "watermark_steps": self.watermark_steps,
                "max_batch_tokens": self.max_batch_tokens, "max_batch_tokens_auto": self.max_batch_tokens_auto,
                "max_batch_size": self.max_batch_size,
                "groups": len(self.groups) if self.groups is not None else None,
                "admission_order_head": list(self.admission_order[:20]), **self.stats}


# scheduler -> executor, inside the batchinfer engine -------------------------------------------------------

@dataclass
class Row:
    """One sequence's slice of a step: token_ids at positions start .. start + len - 1."""
    seq: int  # index into JobAnalysis.requests
    start: int
    token_ids: list[int]
    block_ids: list[int]  # the sequence's physical KV blocks: shared trie blocks first, then its own
    sample: bool  # the slice ends at the sequence's last known token, so its next token is sampled
    decode: bool  # a decode row (prompt fully computed, feeds its last sampled id), else a prefill chunk
    # overlap: the row's one token is the previous step's sampled id at this slot of its output, still on the device
    # when the row is planned (token_ids holds a placeholder); None when token_ids is the token
    prev: int | None = None
    slot: int = 0  # pool=contiguous: the row's region (its sequence's for its admitted life); 0 is the pad slot


@dataclass
class Step:
    """One forward: what the scheduler hands the executor. The executor returns one sampled id per sampling row,
    in row order, and the scheduler's commit() applies them."""
    rows: list[Row]  # decode rows first, then prefill chunks in admission order
    # sharing=token: (source block, destination block, k) KV copies the executor runs before the forward: the first
    # k slots of a computed block, into a block a row of this step then continues from position k
    copies: list = field(default_factory=list)
    # attention=cascade: the rows grouped by shared computed prefix (cascade.Group), every row in exactly one
    groups: list = field(default_factory=list)
    # pool=contiguous: slots this step's admissions took (their regions hold a finished sequence's mappings to drop)
    slot_resets: list = field(default_factory=list)

    def counts(self):
        """(decode rows, prefill tokens, prefill rows)."""
        d = sum(1 for r in self.rows if r.decode)
        return d, sum(len(r.token_ids) for r in self.rows if not r.decode), len(self.rows) - d

    @property
    def tokens(self):
        return sum(len(r.token_ids) for r in self.rows)


# engine -> io ---------------------------------------------------------------------------------------

@dataclass
class Result:
    id: str
    text: str  # decoded once at completion, special tokens skipped
    token_ids: list[int]  # generated ids, including the EOS id when finish_reason == "stop"
    prompt_tokens: int
    output_tokens: int  # == len(token_ids)
    finish_reason: str  # "stop" | "length"
    group: int | None  # fixed-group index; None under continuous admission
    latency_s: float  # admission -> last token (batchinfer engine); group start -> last token, apportioned (naive)

    def to_dict(self):
        return asdict(self)
