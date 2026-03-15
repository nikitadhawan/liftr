import torch
import torch.nn.functional as F

from src.utils import cov_from_jvp
from .ejvp_fns import EJVP_FNS


def layernorm_jacobians_batch(x, eps):
    """Compute per-sample LayerNorm Jacobian matrices for a batch of inputs."""
    orig_shape, d_model = x.shape[:-1], x.shape[-1]

    x_flat = x.reshape(-1, d_model)
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = (var + eps).rsqrt()

    I = torch.eye(d_model, device=x.device, dtype=x.dtype)
    ones_outer = torch.ones((d_model, d_model), device=x.device, dtype=x.dtype)
    term1_base = I - ones_outer / d_model
    term1 = term1_base.unsqueeze(0) * inv_std.unsqueeze(-1)

    x_centered = x_flat - mean
    term2 = torch.einsum("bi,bj->bij", x_centered, x_centered)
    term2 = term2 * (inv_std ** 3).unsqueeze(-1) / d_model

    jac = term1 - term2
    return jac.reshape(*orig_shape, d_model, d_model)

def layernorm_derivative_sum(x_batch, normalized_shape, gamma, eps):
    """Sum Jacobians over a batch and scale by gamma; handles 2D and 3D inputs."""
    if x_batch.dim() == 2:
        jac = layernorm_jacobians_batch(x_batch, eps).sum(dim=0)
        return jac * gamma.unsqueeze(-1)
    if x_batch.dim() == 3:
        batch_size, seq_len, d_model = x_batch.shape
        jac_blocks = layernorm_jacobians_batch(x_batch.reshape(-1, d_model), eps)
        jac_blocks = jac_blocks.reshape(batch_size, seq_len, d_model, d_model)
        jac_blocks = jac_blocks * gamma.unsqueeze(-1)
        sum_jac_blocks = jac_blocks.sum(dim=0)
        return sum_jac_blocks
    raise ValueError(f"Unsupported input shape for LayerNorm: {tuple(x_batch.shape)}")
    
def layer_norm_stats_dict(layer, x, task_id, diag_cov):
    """Accumulate average normalized input and average Jacobian for a LayerNorm layer."""
    assert x.ndim in [2, 3]
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size = x.shape[0]
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)

    if f"{task_id}_W0" not in stats_dict and f"{task_id}_b0" not in stats_dict:
        stats_dict.update({
            f"{task_id}_W0": layer.weight.clone().detach(),
            f"{task_id}_b0": layer.bias.clone().detach(),
        })

    x_normalized = F.layer_norm(x, layer.normalized_shape, weight=None, bias=None, eps=layer.eps)
    avg_x_norm_old = stats_dict.get(f"{task_id}_avg_x_norm", torch.zeros_like(x[0]))
    avg_x_norm = (data_size_old * avg_x_norm_old + x_normalized.sum(dim=0)) / (data_size_old + batch_size)
    
    avg_jac_old = stats_dict.get(f"{task_id}_avg_jac", torch.zeros(*x.shape[1:], x.shape[-1]).to(x.device))
    gamma = stats_dict[f"{task_id}_W0"]
    sum_jac = layernorm_derivative_sum(x, layer.normalized_shape, gamma, layer.eps)
    avg_jac = (data_size_old * avg_jac_old + sum_jac) / (data_size_old + batch_size)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_avg_jac": avg_jac,
        f"{task_id}_avg_x_norm": avg_x_norm, 
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def layer_norm_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Propagate output moments deterministically through a LayerNorm layer."""
    assert E_dx.ndim in [1, 2]
    assert Cov_dx.ndim == E_dx.ndim * 2
    W0 = layer._stats_dict[f"{prev_task_id}_W0"]
    b0 = layer._stats_dict[f"{prev_task_id}_b0"]
    W1, b1 = layer.weight, layer.bias
    dW, db = W1 - W0, b1 - b0
    avg_x_norm = layer._stats_dict[f"{prev_task_id}_avg_x_norm"]
    avg_jac = layer._stats_dict[f"{prev_task_id}_avg_jac"]
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    ejvp_dW = avg_x_norm * dW
    E_dout = ejvp_fn(avg_jac, E_dx) + ejvp_dW + db
    Cov_dout = cov_from_jvp(ejvp_fn, avg_jac, Cov_dx)
    if Cov_dout.ndim == 2:
        Cov_dout = Cov_dout + torch.outer(ejvp_dW, ejvp_dW)
    else:
        Cov_dout = Cov_dout + torch.einsum('ij,kl->ijkl', ejvp_dW, ejvp_dW)
    return (E_dout, Cov_dout)

def layer_norm_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through a LayerNorm layer."""
    assert dx.ndim in [2, 3]
    W0 = layer._stats_dict[f"{prev_task_id}_W0"]
    b0 = layer._stats_dict[f"{prev_task_id}_b0"]
    W1, b1 = layer.weight, layer.bias
    dW, db = W1 - W0, b1 - b0
    avg_x_norm = layer._stats_dict[f"{prev_task_id}_avg_x_norm"]
    avg_jac = layer._stats_dict[f"{prev_task_id}_avg_jac"]
    x = torch.randn((sample_size,) + avg_x_norm.shape, device=dW.device, dtype=dW.dtype)

    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    dout = ejvp_fn(avg_jac, dx) + (x * dW) + db
    return dout