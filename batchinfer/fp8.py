"""FP8 weights for the fused layers (quant="fp8"): each linear projection's weight stored as e4m3 with one float32
scale per output channel (absmax / 448), the activation quantised per token on the fly, and the matmul run by vLLM's
cutlass_scaled_mm with both scales. Halves the weight bytes a decode step reads (16.4 GB to 8.2 GB for Qwen3-8B) and
frees that much memory for the KV pool; costs two kernels per projection (the activation quant and the scaled GEMM)
inside the graphs. lm_head, the embeddings and the norms stay bf16. On the CPU the same arithmetic runs in float as
the oracle the tests check the kernels against; the engine refuses fp8 where there are no FP8 tensor cores (compute
capability below 8.9) or off the fused path. vLLM's own online fp8 scheme is per-tensor, so the two diverge
legitimately; the bench's quality gate compares against this engine's bf16 row, not against vLLM's fp8 row.
"""
import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)  # 448
MIN_CAPABILITY = (8, 9)  # FP8 tensor cores: Ada and Hopper on


def fp8_problems(device, fused):
    """Why quant="fp8" cannot run here; empty when it can."""
    cuda = device.type == "cuda"
    return [what for what, bad in (
        ("quant=fp8 runs on the fused layers (layers=fused) only", not fused),
        ("quant=fp8 needs a GPU with FP8 tensor cores (compute capability 8.9 or above)",
         not cuda or torch.cuda.get_device_capability(device) < MIN_CAPABILITY),
    ) if bad]


def quantize_weight(w):
    """bf16 [N, K] -> (e4m3 [N, K], float32 scale [N, 1]): per output channel, absmax to 448."""
    scale = (w.detach().abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    q = (w.detach().float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


def quantize_act(x):
    """[n, K] -> (e4m3 [n, K], float32 scale [n, 1]): per token, absmax to 448 (what scaled_fp8_quant does on the GPU)."""
    scale = (x.detach().abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    return (x.detach().float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


class Linear8:
    """One projection's weight in fp8, called as the bf16 weight was used: y = x @ W^T, [n, K] -> [n, N]."""

    def __init__(self, w, ops=None):
        """w: the bf16 weight [N, K]; ops: vllm._custom_ops on the GPU, None for the float oracle on the CPU."""
        self.w8, self.scale = quantize_weight(w)
        self.w8t = self.w8.t()  # [K, N] column-major, the layout cutlass_scaled_mm takes for its second operand
        self.ops = ops
        self.shape = tuple(w.shape)

    def __call__(self, x):
        if self.ops is None:  # the oracle: the kernels' arithmetic in float
            a, sa = quantize_act(x)
            return ((a.float() * sa) @ (self.w8.float() * self.scale).t()).to(x.dtype)
        a, sa = self.ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return self.ops.cutlass_scaled_mm(a, self.w8t, sa, self.scale.t(), x.dtype)
