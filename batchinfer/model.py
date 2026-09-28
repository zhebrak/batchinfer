"""Loading the HF model both engines run, and the facts about it they share. Importing this module sets the
CUDA allocator config, so every entry point that may start CUDA imports it before torch does."""
import os

# Before torch initialises CUDA, so the CLI and the bench row run the same allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402


def load_model(model_name, torch_dtype, attn):
    import transformers
    from transformers import AutoModelForCausalLM
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": torch_dtype}
    return AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn, **dtype_kw)


def versions():
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}


def stop_token_ids(model):
    """The ids greedy decoding stops on unless a request sets ignore_eos: generation_config's eos ids, else the
    model config's."""
    eos = model.generation_config.eos_token_id
    eos = model.config.eos_token_id if eos is None else eos
    return [eos] if isinstance(eos, int) else list(eos)
