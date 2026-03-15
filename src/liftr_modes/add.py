

def add_stats_dict(layer, x, task_id, diag_cov):
    """Track data size for a residual addition (no input statistics needed)."""
    a, b = x
    assert a.ndim == 3
    assert b.ndim == 3
    stats_dict = getattr(layer, "_stats_dict", {})
    batch_size = a.shape[0]
    data_size_old = stats_dict.get(f"{task_id}_data_size", 0)
    
    data_size = data_size_old + batch_size
    stats_dict.update({
        f"{task_id}_data_size": data_size,
    })
    return stats_dict

def add_moments(layer, ECov_da, ECov_db, prev_task_id, diag_cov):
    """Propagate output moments through residual addition by summing inputs."""
    E_da, Cov_da = ECov_da
    E_db, Cov_db = ECov_db
    assert E_da.ndim == 2
    assert E_db.ndim == 2
    assert Cov_da.ndim == E_da.ndim * 2
    assert Cov_db.ndim == E_db.ndim * 2

    E_dout = E_da + E_db
    Cov_dout = Cov_da + Cov_db
    return (E_dout, Cov_dout)

def add_stoch_out(layer, dx, prev_task_id, sample_size, diag_cov):
    """Propagate output change stochastically through residual addition."""
    da, db = dx
    assert da.ndim == 3
    assert db.ndim == 3

    return da + db
