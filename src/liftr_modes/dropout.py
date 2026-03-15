
def dropout_stats_dict(layer, x, task_id, diag_cov):
    """No-op: dropout is treated as identity since its average derivative is 1."""
    assert x.ndim in [2, 3, 4]
    stats_dict = getattr(layer, "_stats_dict", {})
    return stats_dict

def dropout_moments(layer, E_dx, Cov_dx, prev_task_id, diag_cov):
    """Pass moments through unchanged (dropout is treated as identity)."""
    assert E_dx.ndim in [1, 2, 3]
    assert Cov_dx.ndim == E_dx.ndim * 2
    return (E_dx, Cov_dx)

def dropout_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Pass output change through unchanged (dropout is treated as identity)."""
    assert dx.ndim in [2, 3, 4]
    return dx