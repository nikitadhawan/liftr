
import torch
from torch import zeros

from src.utils import cov_from_jvp
from .ejvp_fns import EJVP_FNS


def weighted_values_stats_dict(layer, x, task_id, diag_cov):
    """Accumulate E[V] and E[A] statistics for the attention-weighted values operation."""
    v, a = x
    assert v.ndim == 3
    assert a.ndim == 4
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size, seq_len, d_model = v.shape
    num_heads = d_model // layer.d_head
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    v = v.view(batch_size, seq_len, num_heads, layer.d_head).transpose(1, 2)
    eV_old = stats_dict.get(f"{task_id}_eV", zeros(*v.shape[1:]).to(v.device))
    sum_V = v.sum(0)
    eV = (data_size_old * eV_old + sum_V) / (data_size_old + batch_size)

    eA_old = stats_dict.get(f"{task_id}_eK", zeros(*a.shape[1:]).to(a.device))
    sum_A = a.sum(0)
    eA = (data_size_old * eA_old + sum_A) / (data_size_old + batch_size)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_eV": eV,
        f"{task_id}_eA": eA,
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def weighted_values_moments(layer, ECov_dv, ECov_da, prev_task_id, diag_cov):
    """Propagate output moments deterministically through the attention-weighted values operation."""
    E_dv, Cov_dv = ECov_dv
    E_da, Cov_da = ECov_da
    assert E_dv.ndim == 2
    assert E_da.ndim == 3
    assert Cov_dv.ndim == E_dv.ndim * 2
    assert Cov_da.ndim == E_da.ndim * 2
    eV = layer._stats_dict[f"{prev_task_id}_eV"]
    eA = layer._stats_dict[f"{prev_task_id}_eA"]
    ejvp_v_fn, ejvp_a_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_v_fn is None or ejvp_a_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    seq_len, d_model = E_dv.shape
    num_heads = d_model // layer.d_head
    E_dv = E_dv.view(seq_len, num_heads, layer.d_head).transpose(0, 1)
    Cov_dv = Cov_dv.view(seq_len, num_heads, layer.d_head, seq_len, num_heads, layer.d_head).transpose(0, 1).transpose(3, 4)
    E_dout = ejvp_v_fn(eA, E_dv) + ejvp_a_fn(eV, E_da)
    Cov_dout = cov_from_jvp(ejvp_v_fn, eA, Cov_dv) + cov_from_jvp(ejvp_a_fn, eV, Cov_da)
    E_dout = E_dout.transpose(0, 1).contiguous().view(seq_len, d_model)
    Cov_dout = Cov_dout.transpose(0, 1).transpose(3, 4).contiguous().view(seq_len, d_model, seq_len, d_model)
    return (E_dout, Cov_dout)

def weighted_values_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through the attention-weighted values operation."""
    dv, da = dx
    assert dv.ndim == 3
    assert da.ndim == 4
    sample_size, seq_len, d_model = dv.shape
    num_heads = d_model // layer.d_head
    eV = layer._stats_dict[f"{prev_task_id}_eV"]
    eA = layer._stats_dict[f"{prev_task_id}_eA"]
    
    mu_0 = eA[:, :, 0]
    S_0 = 1 + mu_0**2 # assuming identity covariance
    alpha0 = (mu_0 - S_0) / (S_0 - mu_0**2 + 1e-6)  # (num_heads, seq_len)
    alpha = alpha0[:, :, None] * eA  # (num_heads, seq_len, seq_len)
    alpha = torch.clamp(alpha, min=1e-6)
    alpha_flat = alpha.view(-1, alpha.shape[-1])  # (num_heads * seq_len, seq_len)
    dirichlet_dist = torch.distributions.Dirichlet(alpha_flat)
    score_samples_flat = dirichlet_dist.sample((sample_size,))  # (sample_size, num_heads * seq_len, seq_len)
    score_samples = score_samples_flat.view(sample_size, eA.shape[0], eA.shape[1], eA.shape[2])

    z = torch.randn(sample_size, eV.numel(), device=eV.device)
    v = z + eV.reshape(-1)
    v = v.view(sample_size, num_heads, seq_len, layer.d_head)

    ejvp_v_fn, ejvp_a_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_v_fn is None or ejvp_a_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    dv = dv.view(sample_size, seq_len, num_heads, layer.d_head).transpose(1, 2)
    dout = torch.vmap(ejvp_v_fn)(score_samples, dv) + torch.vmap(ejvp_a_fn)(v, da)
    dout = dout.transpose(1, 2).contiguous().view(sample_size, seq_len, d_model)
    return dout