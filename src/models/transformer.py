import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QKDotProduct(nn.Module):
    """Wrapper for attention operations to make them compatible with moment propagation."""
    
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head
    
    def forward(self, x):
        Q, K = x 
        batch_size, seq_len, d_model = Q.shape
        num_heads = d_model // self.d_head
        Q = Q.view(batch_size, seq_len, num_heads, self.d_head)
        K = K.view(batch_size, seq_len, num_heads, self.d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        return torch.einsum('...hid,...hjd->...hij', Q, K) / math.sqrt(self.d_head)


class Softmax(nn.Module):
    """Softmax wrapper module for moment propagation compatibility."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.softmax(x, dim=self.dim)


class WeightedValues(nn.Module):
    """Applies attention scores to value projections."""

    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head
        
    def forward(self, x):
        V, scores = x
        batch_size, seq_len, d_model = V.shape
        num_heads = d_model // self.d_head
        V = V.view(batch_size, seq_len, num_heads, self.d_head)
        V = V.transpose(1, 2)
        values = torch.einsum('...hij,...hjk->...hik', scores, V)
        values = values.transpose(1, 2)
        return values.contiguous().view(batch_size, seq_len, d_model)


class Add(nn.Module):
    """Residual addition module."""

    def forward(self, x):
        x, out = x
        return x + out


class CausalMask(nn.Module):
    """Applies a causal (lower-triangular) mask to attention logits."""

    def forward(self, x):
        seq_len = x.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        return torch.tril(x) - 1e10 * (1 - mask)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""
    def __init__(self, d_model, num_heads, dropout=0.0, causal=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention_scores = QKDotProduct(self.d_head)
        self.causal_mask = CausalMask()
        self.softmax = Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.values = WeightedValues(self.d_head)
        
    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = self.attention_scores((Q, K))

        if self.causal:
            scores = self.causal_mask(scores)

        scores = self.softmax(scores)
        scores = self.dropout(scores)
        values = self.values((V, scores))

        out = self.W_o(values)
        return out

class FFN(nn.Module):
    """Two-layer feed-forward network with ReLU activation."""
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block: self-attention + FFN with pre-norm residuals."""

    def __init__(self, d_model, num_heads, d_hidden):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.sa_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FFN(d_model, d_hidden)
        self.out_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.add = Add()
    
    def forward(self, x):
        out = self.attention(x)
        res = self.add((x, out))
        res = self.sa_ln(res)
        out = self.ffn(res)
        res = self.add((res, out))
        res = self.out_ln(res)
        
        return res


class Transformer(nn.Module):
    """Decoder-only transformer with token embedding and linear unembedding."""
    def __init__(self, output_shape, d_model, num_heads, d_hidden, num_blocks=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.output_shape = output_shape 
        self.num_blocks = num_blocks

        self.embedding = nn.Embedding(output_shape, d_model) 
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_hidden) for _ in range(num_blocks)
        ])  
        self.unembed = nn.Linear(d_model, output_shape)
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        out = self.unembed(x)
        return out
    