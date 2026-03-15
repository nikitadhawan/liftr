import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def set_random_seed(i):
    """Set all relevant random seeds for reproducibility."""
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def abs_diff(x1, x2, reduce='mean'):
    """Mean or sum of per-sample L1 distances between x1 and x2."""
    if reduce == 'mean':
        return torch.mean(0.5 * torch.sum(abs(x2 - x1), dim=1))
    elif reduce == 'sum':
        return torch.sum(0.5 * torch.sum(abs(x2 - x1), dim=1))
        
def euclidean_dist(x1, x2, reduce='mean'):
    """Mean or sum of per-sample squared Euclidean distances between x1 and x2."""
    if reduce == 'mean':
        return torch.mean(0.5 * torch.sum((x2 - x1)**2, dim=1))
    elif reduce == 'sum':
        return torch.sum(0.5 * torch.sum((x2 - x1)**2, dim=1))


def euclidean_normalized(x1, x2, reduce='mean', normalize=True):
    """Euclidean distance between softmax-normalized x1 and x2."""
    x1, x2 = F.softmax(x1, dim=1), F.softmax(x2, dim=1)
    if reduce == 'mean':
        return torch.mean(0.5 * torch.sum((x2 - x1)**2, dim=1))
    elif reduce == 'sum':
        return torch.sum(0.5 * torch.sum((x2 - x1)**2, dim=1))


def kl_div(x1, x2, reduce='mean'):
    """KL divergence from softmax(x1) to softmax(x2)."""
    if reduce == 'mean':
        return F.kl_div(F.log_softmax(x2, dim=1), F.softmax(x1, dim=1), reduction="batchmean").clamp(min=0.)
    elif reduce == 'sum':
        return F.kl_div(F.log_softmax(x2, dim=1), F.softmax(x1, dim=1), reduction="sum").clamp(min=0.)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW
}

OUTPUT_METRICS = {
    "abs_diff": abs_diff,
    "euc": euclidean_dist,
    "euc_norm": euclidean_normalized,
    "kl": kl_div,
}


def cov_from_jvp(jvp_fn, avg_jac, Cov_dx):
    """Propagate input covariance through a linear JVP: Cov_dout = J Cov_dx J^T."""
    assert Cov_dx.ndim in [2, 4, 6]
    x, y = None, None
    if Cov_dx.ndim == 6:
        x, y, z, _, _, _ = Cov_dx.shape
        Cov_dx = Cov_dx.reshape(x*y*z, x*y*z)
        temp = torch.vmap(lambda col: jvp_fn(avg_jac, col.reshape(x, y, z)).flatten())(Cov_dx.T)
        Cov_dout = torch.vmap(lambda col: jvp_fn(avg_jac, col.reshape(x, y, z)))(temp.T)
        _, x, y, z = Cov_dout.shape
        Cov_dout = Cov_dout.reshape(x, y, z, x, y, z)
    elif Cov_dx.ndim == 4:
        x, y, _, _ = Cov_dx.shape
        Cov_dx = Cov_dx.reshape(x*y, x*y)
        temp = torch.vmap(lambda col: jvp_fn(avg_jac, col.reshape(x, y)).flatten())(Cov_dx.T)
        Cov_dout = torch.vmap(lambda col: jvp_fn(avg_jac, col.reshape(x, y)))(temp.T)
        _, x, y = Cov_dout.shape
        Cov_dout = Cov_dout.reshape(x, y, x, y)
    else:
        temp = torch.vmap(lambda col: jvp_fn(avg_jac, col))(Cov_dx.T)
        Cov_dout = torch.vmap(lambda col: jvp_fn(avg_jac, col))(temp.T)
    return Cov_dout
    

def sample_from_multivariate_normal(mean, cov, n_samples):
    """Sample from a multivariate normal via eigendecomposition."""
    L, V = torch.linalg.eigh(cov)
    L = torch.clamp(L, min=0.0)
    scale_matrix = V @ torch.diag(torch.sqrt(L))
    z = torch.randn(n_samples, mean.shape[0], device=mean.device)
    samples = mean + z @ scale_matrix.T
    return samples
    

