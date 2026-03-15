import torch
from torch import (
    einsum,
    zeros,
)

from torch.nn.functional import linear

from src.utils import cov_from_jvp, sample_from_multivariate_normal
from .ejvp_fns import EJVP_FNS


def linear_stats_dict(layer, x, task_id, diag_cov):
    """Accumulate E[x] and Cov[x] statistics for a linear layer."""
    assert x.ndim in [2, 3]
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size = x.shape[0]
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    E_x_old = stats_dict.get(f"{task_id}_E_x", zeros(*x.shape[1:]).to(x.device))
    sum_x = x.sum(0)
    E_x = (data_size_old * E_x_old + sum_x) / (data_size_old + batch_size)
    
    Cov_zero = zeros(*x.shape[1:]) if diag_cov else zeros(*x.shape[1:], *x.shape[1:])
    Cov_x_old = stats_dict.get(f"{task_id}_Cov_x", Cov_zero.to(x.device))
    if diag_cov:
        if x.ndim == 2:
            Cov_x_sum = einsum("ni,ni->i", x, x)
        else:
            Cov_x_sum = einsum("nij,nij->ij", x, x)
    else:
        if x.ndim == 2:
            Cov_x_sum = einsum("ni,nj->ij", x, x) 
        else:
            Cov_x_sum = einsum("nij,nkl->ijkl", x, x) 
    Cov_x = (data_size_old * Cov_x_old + Cov_x_sum) / (data_size_old + batch_size)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_E_x": E_x,
        f"{task_id}_Cov_x": Cov_x,
        f"{task_id}_data_size": data_size,
    })
    if f"{task_id}_W0" not in stats_dict and f"{task_id}_b0" not in stats_dict:
        stats_dict.update({
            f"{task_id}_W0": layer.weight.clone().detach(),
            f"{task_id}_b0": layer.bias.clone().detach(),
        })
    return stats_dict

def linear_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Propagate output moments deterministically through a linear layer."""
    assert E_dx.ndim in [1, 2]
    assert Cov_dx.ndim == E_dx.ndim * 2
    W0 = layer._stats_dict[f"{prev_task_id}_W0"]
    b0 = layer._stats_dict[f"{prev_task_id}_b0"]
    W1, b1 = layer.weight, layer.bias
    dW, db = W1 - W0, b1 - b0
    E_x = layer._stats_dict[f"{prev_task_id}_E_x"]
    Cov_x = layer._stats_dict[f"{prev_task_id}_Cov_x"]
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    E_dout = ejvp_fn(W1, E_dx) + linear(E_x, dW) + db
    Cov_dout = cov_from_jvp(ejvp_fn, W1, Cov_dx)

    if diag_cov:
        if Cov_x.ndim == 1:
            Cov_dout = Cov_dout + (dW * Cov_x) @ dW.T
        else:
            Cov_dW = torch.einsum('jn,in,ln -> ijl', dW, Cov_x, dW)
            indices = torch.arange(Cov_x.shape[0])
            Cov_dout[indices, :, indices, :] = Cov_dout[indices, :, indices, :] + Cov_dW
    else:
        if Cov_x.ndim == 2:
            Cov_dout = Cov_dout + dW @ Cov_x @ dW.T
        else:
            Cov_dout = Cov_dout + torch.einsum(
                'jm,imkn,ln->ijkl', dW, Cov_x, dW
            )

    return (E_dout, Cov_dout)

def linear_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through a linear layer."""
    assert dx.ndim in [2, 3]
    W0 = layer._stats_dict[f"{prev_task_id}_W0"]
    b0 = layer._stats_dict[f"{prev_task_id}_b0"]
    W1, b1 = layer.weight, layer.bias
    dW, db = W1 - W0, b1 - b0
    E_x = layer._stats_dict[f"{prev_task_id}_E_x"]
    Cov_x = layer._stats_dict[f"{prev_task_id}_Cov_x"]
    
    x_shape = E_x.shape
    if E_x.ndim > 1:
        E_x = E_x.reshape(-1)
        if diag_cov:
            Cov_x = Cov_x.reshape(-1)
        else:
            Cov_x = Cov_x.reshape(E_x.shape[0], E_x.shape[0])
    
    if diag_cov:
        Var_x = Cov_x - E_x**2
        z = torch.randn(sample_size, E_x.shape[0], device=E_x.device)
        scale = torch.sqrt(torch.clamp(Var_x, min=0.0))
        x = E_x + z * scale
    else:
        E_xxt = E_x.unsqueeze(-1) @ E_x.unsqueeze(-2)
        Var_x = Cov_x - E_xxt
        x = sample_from_multivariate_normal(E_x, Var_x, sample_size)
    x = x.reshape((sample_size,) + x_shape)
    
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    dout = ejvp_fn(W1, dx) + linear(x, dW) + db
    return dout