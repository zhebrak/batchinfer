"""batchinfer.fp8: the weight and activation quantisation and the Linear8 oracle, on the CPU. The kernels (vLLM's
scaled_fp8_quant and cutlass_scaled_mm) run the same arithmetic; tests/test_executor_gpu.py checks them."""
import torch

from batchinfer import fp8

torch.manual_seed(0)


def test_weight_quantisation_is_per_output_channel_within_an_e4m3_step():
    w = torch.randn(64, 96, dtype=torch.bfloat16) * torch.logspace(-2, 2, 64, dtype=torch.bfloat16)[:, None]
    q, scale = fp8.quantize_weight(w)
    assert q.dtype == fp8.FP8 and q.shape == w.shape and scale.shape == (64, 1) and scale.dtype == torch.float32
    back = q.float() * scale
    err = (back - w.float()).abs().amax(dim=1) / w.float().abs().amax(dim=1)
    assert float(err.max()) < 1 / 16  # e4m3 keeps 3 mantissa bits: within one step of the channel's largest value
    assert torch.equal(q.float().abs().amax(dim=1), torch.full((64,), fp8.FP8_MAX))  # the absmax lands on 448


def test_activation_quantisation_is_per_token():
    x = torch.randn(5, 32, dtype=torch.bfloat16) * torch.tensor([1, 10, 100, 0.1, 1000], dtype=torch.bfloat16)[:, None]
    a, scale = fp8.quantize_act(x)
    assert a.shape == x.shape and scale.shape == (5, 1)
    assert torch.allclose(scale[:, 0], x.float().abs().amax(dim=1) / fp8.FP8_MAX)


def test_linear8_oracle_is_within_a_few_percent_of_the_bf16_matmul():
    """e4m3 keeps 3 mantissa bits on both operands, so on Gaussian data the product's relative error is a few percent
    in norm (each element within a 1/8 step); the fp8 row's quality gate in the bench is the check that matters."""
    w, x = torch.randn(48, 96, dtype=torch.bfloat16), torch.randn(7, 96, dtype=torch.bfloat16)
    lin = fp8.Linear8(w)
    assert lin.shape == (48, 96) and lin.w8t.shape == (96, 48) and lin.w8t.stride() == (1, 96)  # column-major
    y, ref = lin(x), x.float() @ w.float().t()
    assert y.dtype == torch.bfloat16 and y.shape == (7, 48)
    assert float((y.float() - ref).norm() / ref.norm()) < 0.05
    assert torch.equal(fp8.Linear8(w)(x), y)  # deterministic


def test_fp8_is_refused_off_the_fused_path_and_off_fp8_tensor_cores():
    cpu = torch.device("cpu")
    assert fp8.fp8_problems(cpu, fused=True) == ["quant=fp8 needs a GPU with FP8 tensor cores (compute capability 8.9 or above)"]
    assert len(fp8.fp8_problems(cpu, fused=False)) == 2
