import torch 
from torch import nn
from torch.nn.functional import linear

from src.models.transformer import QKDotProduct, CausalMask, Softmax, WeightedValues, Add


def ejvp_lin_manual(weight, vec):
    """JVP for a linear layer: W @ vec."""
    return linear(vec, weight)

def ejvp_relu_manual(avg_jac, vec):
    """JVP for ReLU: element-wise multiply by average Jacobian."""
    return avg_jac * vec

def ejvp_q_manual(avg_jac, vec):
    """JVP for attention scores with respect to query: (dQ K^T) / sqrt(d)."""
    eK, d_model = avg_jac
    ejvp_q = torch.einsum('hid,hjd->hij', vec, eK) / torch.sqrt(d_model)
    return ejvp_q

def ejvp_k_manual(avg_jac, vec):
    """JVP for attention scores with respect to key: (Q dK^T) / sqrt(d)."""
    eQ, d_model = avg_jac
    ejvp_k = torch.einsum('hid,hjd->hij', eQ, vec) / torch.sqrt(d_model)
    return ejvp_k

def ejvp_softmax_manual(avg_jac, vec):
    """JVP for softmax: diag(eA) @ vec - outer_avgs @ vec."""
    eA, outer_avgs = avg_jac
    term1 = eA * vec
    if vec.ndim == 4:
        term2 = torch.einsum('bhijk,bhik->bhij', outer_avgs, vec)
    else:
        term2 = torch.einsum('hijk,hik->hij', outer_avgs, vec)
    return term1 - term2

def ejvp_v_manual(eA, vec_v):
    """JVP for weighted values with respect to V: A @ dV."""
    ejvp_v = torch.einsum('hij,hjk->hik', eA, vec_v)
    return ejvp_v

def ejvp_a_manual(eV, vec_a):
    """JVP for weighted values with respect to attention scores: dA @ V."""
    ejvp_a = torch.einsum('hij,hjk->hik', vec_a, eV)
    return ejvp_a

def ejvp_causal_mask_manual(dummy_one, vec):
    """JVP for causal mask: lower-triangular projection."""
    return torch.tril(vec)

def ejvp_gelu_manual(avg_jac, vec):
    """JVP for GELU: element-wise multiply by average Jacobian."""
    return avg_jac * vec

def ejvp_dropout_manual(avg_jac, vec):
    """JVP for dropout (identity): pass-through."""
    return avg_jac * vec

def ejvp_layernorm_manual(avg_jac, vec):
    """JVP for LayerNorm: apply average Jacobian via einsum."""
    if vec.ndim == 3:
        return torch.einsum('sij,bsj->bsi', avg_jac, vec)
    return torch.einsum('sij,sj->si', avg_jac, vec)


EJVP_FNS = {
    nn.Linear: ejvp_lin_manual,
    nn.ReLU: ejvp_relu_manual,
    QKDotProduct: (ejvp_q_manual, ejvp_k_manual),
    Softmax: ejvp_softmax_manual, 
    WeightedValues: (ejvp_v_manual, ejvp_a_manual),
    CausalMask: ejvp_causal_mask_manual,
    nn.GELU: ejvp_gelu_manual,
    nn.Dropout: ejvp_dropout_manual,
    nn.LayerNorm: ejvp_layernorm_manual,
}
