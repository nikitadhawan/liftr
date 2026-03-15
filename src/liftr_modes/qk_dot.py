
import torch
from torch import zeros

from src.utils import cov_from_jvp
from .ejvp_fns import EJVP_FNS


def qk_dot_stats_dict(layer, x, task_id, diag_cov):
    """Accumulate E[Q] and E[K] statistics for the query-key dot product."""
    q, k = x
    assert q.ndim == 3
    assert k.ndim == 3
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size, seq_len, d_model = q.shape
    num_heads = d_model // layer.d_head
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    q = q.view(batch_size, seq_len, num_heads, layer.d_head).transpose(1, 2)
    eQ_old = stats_dict.get(f"{task_id}_eQ", zeros(*q.shape[1:]).to(q.device))
    sum_Q = q.sum(0)
    eQ = (data_size_old * eQ_old + sum_Q) / (data_size_old + batch_size)

    k = k.view(batch_size, seq_len, num_heads, layer.d_head).transpose(1, 2)
    eK_old = stats_dict.get(f"{task_id}_eK", zeros(*k.shape[1:]).to(k.device))
    sum_K = k.sum(0)
    eK = (data_size_old * eK_old + sum_K) / (data_size_old + batch_size)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_eQ": eQ,
        f"{task_id}_eK": eK,
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def qk_dot_moments(layer, ECov_dq, ECov_dk, prev_task_id, diag_cov):
    """Propagate output moments deterministically through the query-key dot product."""
    E_dq, Cov_dq = ECov_dq
    E_dk, Cov_dk = ECov_dk
    assert E_dq.ndim == 2
    assert E_dk.ndim == 2
    assert Cov_dq.ndim == E_dq.ndim * 2
    assert Cov_dk.ndim == E_dk.ndim * 2
    eQ = layer._stats_dict[f"{prev_task_id}_eQ"]
    eK = layer._stats_dict[f"{prev_task_id}_eK"]
    ejvp_q_fn, ejvp_k_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_q_fn is None or ejvp_k_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    seq_len, d_model = E_dq.shape
    num_heads = d_model // layer.d_head
    E_dq = E_dq.view(seq_len, num_heads, layer.d_head).transpose(0, 1)
    E_dk = E_dk.view(seq_len, num_heads, layer.d_head).transpose(0, 1)
    Cov_dq = Cov_dq.view(seq_len, num_heads, layer.d_head, seq_len, num_heads, layer.d_head).transpose(0, 1).transpose(3, 4)
    Cov_dk = Cov_dk.view(seq_len, num_heads, layer.d_head, seq_len, num_heads, layer.d_head).transpose(0, 1).transpose(3, 4)
    d_head = torch.tensor(layer.d_head, device=eQ.device)
    E_dout = ejvp_q_fn((eK, d_head), E_dq) + ejvp_k_fn((eQ, d_head), E_dk)
    Cov_dout = cov_from_jvp(ejvp_q_fn, (eK, d_head), Cov_dq) + cov_from_jvp(ejvp_k_fn, (eQ, d_head), Cov_dk)
    return (E_dout, Cov_dout)

def qk_dot_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through the query-key dot product."""
    dq, dk = dx
    assert dq.ndim == 3
    assert dk.ndim == 3
    sample_size, seq_len, d_model = dq.shape
    num_heads = d_model // layer.d_head

    eQ = layer._stats_dict[f"{prev_task_id}_eQ"]
    eK = layer._stats_dict[f"{prev_task_id}_eK"]
    z = torch.randn(sample_size, eQ.numel(), device=eQ.device)
    q = z + eQ.reshape(-1)
    q = q.view(sample_size, num_heads, seq_len, layer.d_head)
    z = torch.randn(sample_size, eK.numel(), device=eK.device)
    k = z + eK.reshape(-1)
    k = k.view(sample_size, num_heads, seq_len, layer.d_head)
    
    ejvp_q_fn, ejvp_k_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_q_fn is None or ejvp_k_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    dq = dq.view(sample_size, seq_len, num_heads, layer.d_head).transpose(1, 2)
    dk = dk.view(sample_size, seq_len, num_heads, layer.d_head).transpose(1, 2)
    d_heads = torch.tensor([layer.d_head for _ in range(sample_size)], device=q.device)
    dout = torch.vmap(ejvp_q_fn)(
        (k, d_heads), dq
    ) + torch.vmap(ejvp_k_fn)(
        (q, d_heads), dk
    )
    return dout