
from src.utils import cov_from_jvp
from .ejvp_fns import EJVP_FNS


def causal_mask_stats_dict(layer, x, task_id, diag_cov):
    """Track data size for the causal mask (no input statistics needed)."""
    assert x.ndim == 4
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size = x.shape[0]
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def causal_mask_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Propagate output moments deterministically through the causal mask."""
    assert E_dx.ndim in [1, 3]
    assert Cov_dx.ndim == E_dx.ndim * 2
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    E_dout = ejvp_fn(1, E_dx)
    Cov_dout = cov_from_jvp(ejvp_fn, 1, Cov_dx)
    return (E_dout, Cov_dout)


def causal_mask_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through the causal mask."""
    assert dx.ndim == 4
    ejvp_fn = EJVP_FNS.get(layer.__class__, None)
    if ejvp_fn is None:
        raise NotImplementedError(f"No eJVP function for {layer.__class__.__name__}")

    return ejvp_fn(1, dx)