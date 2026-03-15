import math
import torch
import torch.nn.functional as F
from torch import nn

from src.models.transformer import QKDotProduct, CausalMask, Softmax, WeightedValues, Add
from src.liftr_modes.ejvp_fns import *


def ejvp_auto(f, X, dX, randomness=None):
    vmap_kwargs = {} if randomness is None else {"randomness": randomness}
    batched_jvp = torch.vmap(
        lambda x: torch.func.jvp(f, (x,), (dX,))[1],
        **vmap_kwargs,
    )(X)
    return torch.mean(batched_jvp, dim=0)

def ejvp_auto_multiarg(f, args, dargs, randomness=None):
    vmap_kwargs = {} if randomness is None else {"randomness": randomness}
    batched_jvp = torch.vmap(
        lambda *x: torch.func.jvp(f, x, dargs)[1],
        **vmap_kwargs,
    )(*args)
    return torch.mean(batched_jvp, dim=0)

b, h, s, d = 4, 2, 3, 8  # batch size, num heads, sequence length, head dimension

class TestLinearJVP:
    def test_linear1d_jvp_e_delta(self):
        X = torch.rand(b, d*h)
        e_deltaX = torch.rand(d*h)
        lin = nn.Linear(d*h, d*h, bias=False)

        e_jvp_auto = ejvp_auto(lambda X: lin(X), X, e_deltaX)
        e_jvp_manual = ejvp_lin_manual(lin.weight, e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

    def test_linear2d_jvp_e_delta(self):
        X = torch.rand(b, s, d*h)
        e_deltaX = torch.rand(s, d*h)
        lin = nn.Linear(d*h, d*h, bias=False)

        e_jvp_auto = ejvp_auto(lambda X: lin(X), X, e_deltaX)
        e_jvp_manual = ejvp_lin_manual(lin.weight, e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

class TestReLUJVP:
    def test_relu1d_jvp_e_delta(self):
        X = torch.randn(b, d*h)
        e_deltaX = torch.rand(d*h)
        relu = nn.ReLU()

        avg_jac = torch.mean((X >= 0).float(), dim=0)

        e_jvp_auto = ejvp_auto(lambda X: relu(X), X, e_deltaX)
        e_jvp_manual = ejvp_relu_manual(avg_jac, e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

    def test_relu2d_jvp_e_delta(self):
        X = torch.randn(b, s, d*h)
        e_deltaX = torch.rand(s, d*h)
        relu = nn.ReLU()

        avg_jac = torch.mean((X >= 0).float(), dim=0)

        e_jvp_auto = ejvp_auto(lambda X: relu(X), X, e_deltaX)
        e_jvp_manual = ejvp_relu_manual(avg_jac, e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

class TestQKDotProductJVP:
    def test_qk_dot_product_jvp_q(self):
        Q = torch.rand(b, h, s, d)
        K = torch.rand(b, h, s, d)
        e_deltaQ = torch.rand(h, s, d)
        qk_module = QKDotProduct(d)

        eK = torch.mean(K, dim=0)

        e_jvp_auto = ejvp_auto_multiarg(lambda Q, K: qk_module((Q, K)), (Q, K), (e_deltaQ, torch.zeros_like(eK))).squeeze()
        e_jvp_manual = ejvp_q_manual((eK, torch.tensor(d)), e_deltaQ)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

    def test_qk_dot_product_jvp_k(self):
        Q = torch.rand(b, h, s, d)
        K = torch.rand(b, h, s, d)
        e_deltaK = torch.rand(h, s, d)
        qk_module = QKDotProduct(d)

        eQ = torch.mean(Q, dim=0)

        e_jvp_auto = ejvp_auto_multiarg(lambda Q, K: qk_module((Q, K)), (Q, K), (torch.zeros_like(eQ), e_deltaK)).squeeze()
        e_jvp_manual = ejvp_k_manual((eQ, torch.tensor(d)), e_deltaK)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

class TestSoftmaxJVP:
    def test_softmax_jvp(self):
        X = torch.rand(b, h, s, s)
        e_deltaX = torch.rand(h, s, s)
        softmax = Softmax(dim=-1)

        eA = torch.mean(softmax(X), dim=0)
        outer_avgs = torch.mean(torch.einsum('bhij,bhik->bhijk', softmax(X), softmax(X)), dim=0)

        e_jvp_auto = ejvp_auto(lambda X: softmax(X), X, e_deltaX)
        e_jvp_manual = ejvp_softmax_manual((eA, outer_avgs), e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

class TestWeightedValuesJVP:
    def test_weighted_values_jvp_a(self):
        A = torch.rand(b, h, s, s)
        V = torch.rand(b, s, h*d)
        e_deltaA = torch.rand(h, s, s)
        wv_module = WeightedValues(d)

        def f_a(V, A):
            return wv_module((V.unsqueeze(0), A.unsqueeze(0))).squeeze(0)

        eV_mh = torch.mean(V, dim=0).view(s, h, d).transpose(0, 1)  

        e_jvp_auto = ejvp_auto_multiarg(f_a, (V, A), (torch.zeros(s, h*d), e_deltaA))
        e_jvp_manual = ejvp_a_manual(eV_mh, e_deltaA).transpose(0, 1).reshape(s, h*d)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

    def test_weighted_values_jvp_v(self):
        A = torch.rand(b, h, s, s)
        V = torch.rand(b, s, h*d)
        e_deltaV = torch.rand(s, h*d)
        wv_module = WeightedValues(d)

        def f_v(V, A):
            return wv_module((V.unsqueeze(0), A.unsqueeze(0))).squeeze(0)

        eA = torch.mean(A, dim=0)  
        e_deltaV_mh = e_deltaV.view(s, h, d).transpose(0, 1) 

        e_jvp_auto = ejvp_auto_multiarg(f_v, (V, A), (e_deltaV, torch.zeros_like(eA)))
        e_jvp_manual = ejvp_v_manual(eA, e_deltaV_mh).transpose(0, 1).reshape(s, h*d)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)


class TestAddJVP:
    def test_add_jvp_a(self):
        A = torch.rand(b, s, d*h)
        B = torch.rand(b, s, d*h)
        e_deltaA = torch.rand(s, d*h)
        add_module = Add()

        e_jvp_auto = ejvp_auto_multiarg(lambda A, B: add_module((A, B)), (A, B), (e_deltaA, torch.zeros_like(e_deltaA)))

        assert torch.allclose(e_jvp_auto, e_deltaA)

    def test_add_jvp_b(self):
        A = torch.rand(b, s, d*h)
        B = torch.rand(b, s, d*h)
        e_deltaB = torch.rand(s, d*h)
        add_module = Add()

        e_jvp_auto = ejvp_auto_multiarg(lambda A, B: add_module((A, B)), (A, B), (torch.zeros_like(e_deltaB), e_deltaB))

        assert torch.allclose(e_jvp_auto, e_deltaB)

class TestCausalMaskJVP:
    def test_causal_mask_jvp(self):
        X = torch.rand(b, h, s, s)
        e_deltaX = torch.rand(h, s, s)
        causal_mask = CausalMask()

        e_jvp_auto = ejvp_auto(lambda X: causal_mask(X), X, e_deltaX)
        e_jvp_manual = ejvp_causal_mask_manual(1, e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

def gelu_derivative(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf

class TestGELUJVP:
    def test_gelu1d_jvp_e_delta(self):
        X = torch.rand(b, d * h)
        e_deltaX = torch.rand(d * h)
        gelu = nn.GELU()

        e_jvp_auto = ejvp_auto(lambda X: gelu(X), X, e_deltaX)
        e_jvp_manual = ejvp_gelu_manual(torch.mean(gelu_derivative(X), dim=0), e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

    def test_gelu2d_jvp_e_delta(self):
        X = torch.rand(b, s, d * h)
        e_deltaX = torch.rand(s, d * h)
        gelu = nn.GELU()

        e_jvp_auto = ejvp_auto(lambda X: gelu(X), X, e_deltaX)
        e_jvp_manual = ejvp_gelu_manual(torch.mean(gelu_derivative(X), dim=0), e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)

class TestDropoutJVP:
    p = 0.1
    b_dropout = 2048
    torch.manual_seed(0)

    def test_dropout1d_jvp_e_delta(self):
        X = torch.rand(self.b_dropout, d * h)
        e_deltaX = torch.rand(d * h)
        dropout = nn.Dropout(p=self.p)

        e_jvp_auto = ejvp_auto(lambda X: dropout(X), X, e_deltaX, randomness="different")
        e_jvp_manual = ejvp_dropout_manual(torch.ones(d * h), e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual, atol=1e-2)

    def test_dropout2d_jvp_e_delta(self):
        X = torch.rand(self.b_dropout, s, d * h)
        e_deltaX = torch.rand(s, d * h)
        dropout = nn.Dropout(p=self.p)

        e_jvp_auto = ejvp_auto(lambda X: dropout(X), X, e_deltaX, randomness="different")
        e_jvp_manual = ejvp_dropout_manual(torch.ones(s, d * h), e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual, atol=1e-1)


def layernorm_jacobians_batch(x, eps):
    orig_shape, d_model = x.shape[:-1], x.shape[-1]
    x_flat = x.reshape(-1, d_model)
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = (var + eps).rsqrt()
    I = torch.eye(d_model, device=x.device, dtype=x.dtype)
    ones_outer = torch.ones((d_model, d_model), device=x.device, dtype=x.dtype)
    term1 = (I - ones_outer / d_model).unsqueeze(0) * inv_std.unsqueeze(-1)
    x_centered = x_flat - mean
    term2 = torch.einsum("bi,bj->bij", x_centered, x_centered) * (inv_std ** 3).unsqueeze(-1) / d_model
    return (term1 - term2).reshape(*orig_shape, d_model, d_model)


def layernorm_derivative_avg(x_batch, normalized_shape, eps):
    if x_batch.dim() == 2:
        return layernorm_jacobians_batch(x_batch, eps).mean(dim=0)
    if x_batch.dim() == 3:
        batch_size, seq_len, d_model = x_batch.shape
        jac_blocks = layernorm_jacobians_batch(x_batch.reshape(-1, d_model), eps)
        return jac_blocks.reshape(batch_size, seq_len, d_model, d_model).mean(dim=0)
    raise ValueError(f"Unsupported input shape for LayerNorm: {tuple(x_batch.shape)}")

class TestLayerNormJVP:
    eps = 1e-5

    def test_layernorm_jvp_e_delta(self):
        X = torch.rand(b, s, d * h)
        e_deltaX = torch.rand(s, d * h)

        jac_avg = layernorm_derivative_avg(X, d * h, self.eps)

        e_jvp_auto = ejvp_auto(lambda X: F.layer_norm(X, (d * h,), eps=self.eps), X, e_deltaX)
        e_jvp_manual = ejvp_layernorm_manual(jac_avg, e_deltaX)

        assert torch.allclose(e_jvp_auto, e_jvp_manual)


def run_all_tests():
    test_classes = [
        TestLinearJVP,
        TestReLUJVP,
        TestQKDotProductJVP,
        TestSoftmaxJVP,
        TestWeightedValuesJVP,
        TestAddJVP,
        TestCausalMaskJVP,
        TestGELUJVP,
        TestDropoutJVP,
        TestLayerNormJVP,
    ]

    passed, failed = 0, 0
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]

        print(f"\n{test_class.__name__}")
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  PASSED  {method_name}")
                passed += 1
            except Exception as e:
                print(f"  FAILED  {method_name}: {e}")
                failed += 1

    total = passed + failed
    print(f"\n{'─' * 46}")
    print(f"  {passed}/{total} passed" + (f"  ({failed} failed)" if failed else ""))
    print(f"{'─' * 46}\n")

if __name__ == "__main__":
    run_all_tests()
