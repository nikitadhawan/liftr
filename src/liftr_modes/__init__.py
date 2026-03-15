"""Layer-wise registries mapping module types to stats, moment, and stochastic output functions."""
import torch.nn as nn

from src.models.transformer import QKDotProduct, Softmax, WeightedValues, Add, CausalMask

from .embedding import embedding_stats_dict, embedding_moments, embedding_stoch_out
from .linear import linear_stats_dict, linear_moments, linear_stoch_out
from .relu import relu_stats_dict, relu_moments, relu_stoch_out
from .qk_dot import qk_dot_stats_dict, qk_dot_moments, qk_dot_stoch_out
from .softmax import softmax_stats_dict, softmax_moments, softmax_stoch_out
from .weighted_values import weighted_values_stats_dict, weighted_values_moments, weighted_values_stoch_out
from .add import add_stats_dict, add_moments, add_stoch_out
from .causal_mask import causal_mask_stats_dict, causal_mask_moments, causal_mask_stoch_out
from .dropout import dropout_stats_dict, dropout_moments, dropout_stoch_out
from .layer_norm import layer_norm_stats_dict, layer_norm_moments, layer_norm_stoch_out


LAYER_STATS_DICTS = {
    nn.Linear: linear_stats_dict,
    nn.ReLU: relu_stats_dict,
    QKDotProduct: qk_dot_stats_dict,
    Softmax: softmax_stats_dict,
    WeightedValues: weighted_values_stats_dict,
    nn.Embedding: embedding_stats_dict,
    Add: add_stats_dict,
    CausalMask: causal_mask_stats_dict,
    nn.Dropout: dropout_stats_dict,
    nn.LayerNorm: layer_norm_stats_dict,
}

LAYER_MOMENTS = {
    nn.Linear: linear_moments,
    nn.ReLU: relu_moments,
    QKDotProduct: qk_dot_moments,
    Softmax: softmax_moments,
    WeightedValues: weighted_values_moments,
    nn.Embedding: embedding_moments,
    Add: add_moments,
    CausalMask: causal_mask_moments,
    nn.Dropout: dropout_moments,
    nn.LayerNorm: layer_norm_moments,
}

LAYER_STOCH_OUT = {
    nn.Linear: linear_stoch_out,
    nn.ReLU: relu_stoch_out,
    QKDotProduct: qk_dot_stoch_out,
    Softmax: softmax_stoch_out,
    WeightedValues: weighted_values_stoch_out,
    nn.Embedding: embedding_stoch_out,
    Add: add_stoch_out,
    CausalMask: causal_mask_stoch_out,
    nn.Dropout: dropout_stoch_out,
    nn.LayerNorm: layer_norm_stoch_out,
}