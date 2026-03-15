from torch import zeros, zeros_like

def embedding_stats_dict(layer, x, task_id, diag_cov):
    """No-op: embedding weights are frozen so no statistics are needed."""
    stats_dict = getattr(layer, "_stats_dict", {})
    return stats_dict

def embedding_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Return zero moments (embedding weights are frozen)."""
    E_dout = zeros_like(layer(E_dx.long()))
    Cov_dout = zeros(*E_dout.shape, *E_dout.shape).to(Cov_dx.device)
    return (E_dout, Cov_dout)

def embedding_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Return zero output change (embedding weights are frozen)."""
    dout = zeros_like(layer(dx.long()))
    return dout