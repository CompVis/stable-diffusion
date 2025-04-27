# ldm/modules/rope_utils.py

import torch

def build_rope_cache(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    sin_emb = emb.sin()[None, None, :, :]  # (1, 1, seq_len, head_dim)
    cos_emb = emb.cos()[None, None, :, :]
    return sin_emb, cos_emb

def apply_rope(x, rope_cache):
    sin_emb, cos_emb = rope_cache
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_out = torch.cat([x1 * cos_emb - x2 * sin_emb,
                       x1 * sin_emb + x2 * cos_emb], dim=-1)
    return x_out
