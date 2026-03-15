
import torch
from torch import zeros, zeros_like
import torch.nn.functional as F

from src.utils import cov_from_jvp
from .ejvp_fns import EJVP_FNS


def softmax_stats_dict(layer, x, task_id, diag_cov):
    """Accumulate E[softmax(x)] and outer-product statistics for a softmax layer."""
    assert x.ndim == 4
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size = x.shape[0]
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    eA_old = stats_dict.get(f"{task_id}_eA", zeros(*x.shape[1:]).to(x.device))
    sum_A = layer(x).sum(0)
    eA = (data_size_old * eA_old + sum_A) / (data_size_old + batch_size)

    softmax_out = layer(x)
    outer_avgs_old = stats_dict.get(f"{task_id}_outer_avgs", zeros(*x.shape[1:], x.shape[-1]).to(x.device))
    if diag_cov:
        diag_sums = torch.einsum('bhij->hij', softmax_out ** 2)  # (head, seq_len, seq_len)
        diag_sums_flat = diag_sums.view(-1, diag_sums.shape[-1])  # (head * seq_len, seq_len)
        diag_matrices_flat = torch.diag_embed(diag_sums_flat, dim1=-2, dim2=-1)  # (head * seq_len, seq_len, seq_len)
        sum_outer_avgs = diag_matrices_flat.view(x.shape[1], x.shape[-1], x.shape[-1], x.shape[-1])  # (head, seq_len, seq_len, seq_len)
    else:
        sum_outer_avgs = torch.einsum('bhij,bhik->hijk', softmax_out, softmax_out)
    outer_avgs = (data_size_old * outer_avgs_old + sum_outer_avgs) / (data_size_old + batch_size)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_eA": eA,
        f"{task_id}_outer_avgs": outer_avgs,
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def softmax_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Propagate output moments deterministically through a softmax layer."""
    assert E_dx.ndim in [1, 3]
    assert Cov_dx.ndim == E_dx.ndim * 2
    eA = layer._stats_dict[f"{prev_task_id}_eA"]
    outer_avgs = layer._stats_dict[f"{prev_task_id}_outer_avgs"]
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    E_dout = ejvp_fn((eA, outer_avgs), E_dx)
    Cov_dout = cov_from_jvp(ejvp_fn, (eA, outer_avgs), Cov_dx)
    return (E_dout, Cov_dout)


def softmax_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through a softmax layer via Dirichlet samples."""
    assert dx.ndim == 4
    eA = layer._stats_dict[f"{prev_task_id}_eA"]
    outer_avgs = layer._stats_dict[f"{prev_task_id}_outer_avgs"]

    mu_0 = eA[:, :, 0]
    S_0 = outer_avgs[:, :, 0, 0]
    alpha0 = (mu_0 - S_0) / (S_0 - mu_0**2 + 1e-6)
    alpha = alpha0[:, :, None] * eA
    alpha = torch.clamp(alpha, min=1e-6)
    alpha_flat = alpha.view(-1, alpha.shape[-1])
    dirichlet_dist = torch.distributions.Dirichlet(alpha_flat)
    score_samples_flat = dirichlet_dist.sample((sample_size,))
    score_samples = score_samples_flat.view(sample_size, eA.shape[0], eA.shape[1], eA.shape[2])

    inner_sum = torch.einsum('bhik,bhik->bhi', score_samples, dx)
    term2 = score_samples * inner_sum.unsqueeze(-1)
    term1 = score_samples * dx

    dout = term1 - term2
    return dout