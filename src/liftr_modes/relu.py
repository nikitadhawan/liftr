
from torch import zeros
from torch.distributions.bernoulli import Bernoulli

from src.utils import cov_from_jvp
from .ejvp_fns import EJVP_FNS


def relu_stats_dict(layer, x, task_id, diag_cov):
    """Accumulate average activation Jacobian (P(x > 0)) for a ReLU layer."""
    assert x.ndim in [2, 3]
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size = x.shape[0]
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    avg_jac_old = stats_dict.get(f"{task_id}_avg_jac", zeros(*x.shape[1:]).to(x.device))
    sum_jac = (x >= 0).float().sum(0)
    avg_jac = (data_size_old * avg_jac_old + sum_jac) / (data_size_old + batch_size)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_avg_jac": avg_jac,
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def relu_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Propagate output moments deterministically through a ReLU layer."""
    assert E_dx.ndim in [1, 2]
    assert Cov_dx.ndim == E_dx.ndim * 2
    avg_jac = layer._stats_dict[f"{prev_task_id}_avg_jac"]
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    E_dout = ejvp_fn(avg_jac, E_dx)
    Cov_dout = cov_from_jvp(ejvp_fn, avg_jac, Cov_dx)
    return (E_dout, Cov_dout)

def relu_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through a ReLU layer via Bernoulli masks."""
    assert dx.ndim in [2, 3]
    avg_jac = layer._stats_dict[f"{prev_task_id}_avg_jac"]
    avg_jac_samples = Bernoulli(probs=avg_jac).sample((sample_size,))
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    dout = ejvp_fn(avg_jac_samples, dx)
    return dout