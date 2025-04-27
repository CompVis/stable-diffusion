"""
Bounded Attention patch for Stable-Diffusion v1 UNet
(implements Dahary et al., 2024)
"""
from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F

@dataclass
class SubjectMask:
    start: int
    end: int          # half-open [start, end)

    def slice(self, L: int) -> slice:
        return slice(max(0, self.start), min(L, self.end))


# ---------- helpers ---------------------------------------------------------

def _build_key_mask(L: int, subjects: List[SubjectMask], device) -> torch.Tensor:
    """
    returns [Nsubj+1, 1, 1, L] bool
      - first N slices = subjects
      - last slice     = background (everything not in any subject span)
    """
    full = torch.zeros(L, dtype=torch.bool, device=device)
    subj_masks = []
    covered = torch.zeros_like(full)

    for sm in subjects:
        m = full.clone()
        m[sm.slice(L)] = True
        covered |= m
        subj_masks.append(m)

    # background “bucket”
    background = ~covered
    return torch.stack([*subj_masks, background])[:, None, None, :]  # [N+1,1,1,L]


def _safe_softmax(attn: torch.Tensor, mask: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    softmax with -inf masking that **guarantees** each row has ≥1 valid key.
    if a row would be all -inf we instead fall back to an un-masked softmax
    for that row only (uniform attention ≈ no harm, avoids nans).
    """
    max_neg = -torch.finfo(attn.dtype).max
    attn = attn.masked_fill(~mask, max_neg)

    # rows where everything is masked
    all_masked = (mask.sum(dim=dim, keepdim=True) == 0)
    if all_masked.any():
        attn = attn.masked_fill(all_masked, 0.0)

    return F.softmax(attn, dim=dim, dtype=torch.float32)


# ---------- monkey-patch machinery -----------------------------------------

_patch: Optional[tuple] = None

def enable_bounded_attention(model, subjects: List[SubjectMask]):
    """
    Enable bounded attention on **all** CrossAttention layers in `model`.
    """
    global _patch
    if _patch is not None:
        raise RuntimeError("already enabled")

    from ldm.modules.attention import CrossAttention            # import locally
    orig_forward = CrossAttention.forward

    def forward_ba(self, x, context=None, mask=None):
        h      = self.heads
        context = x if context is None else context              # self-attention

        B, Lq, _  = x.shape
        Lk         = context.shape[1]
        device     = context.device

        # build / cache masks
        if (not hasattr(self, "_ba_kmask")
                or self._ba_kmask.shape[-1] != Lk):
            self._ba_kmask = _build_key_mask(Lk, subjects, device)   # [N+1,1,1,Lk]

        # decide, **per query token**, which bucket to use
        # rule of thumb: token ∈ subject_i  → bucket i
        #                else               → background bucket (−1 index)
        bucket_ids = torch.full((Lk,), len(subjects), device=device)
        for i, sm in enumerate(subjects):
            bucket_ids[sm.slice(Lk)] = i                          # assign ids

        # projections
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        dim_head  = q.shape[-1] // h
        q, k, v   = map(lambda t: t.view(B, -1, h, dim_head).transpose(1, 2),
                        (q, k, v))                                 # (B,h,Len,dh)

        attn  = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,h,Lq,Lk)

        # broadcast key-mask to (B,h,Lq,Lk) by picking the right bucket for each query
        # 1.  Union-of-subject + background bucket  → shape (1,1,1,Lk)
        kmask = self._ba_kmask.any(0, keepdim=True)               # (1,1,1,Lk)

        # 2.  Bring in SD’s own mask (if it exists)
        if mask is not None:                                      # mask: (B,1,1,Lk)
            kmask = kmask & mask

        # 3.  Broadcast to (B,h,Lq,Lk) automatically
        probs = _safe_softmax(attn, kmask, dim=-1)

        out   = torch.matmul(probs, v)                            # (B,h,Lq,dh)
        out   = out.transpose(1, 2).reshape(B, Lq, h * dim_head)
        return self.to_out(out)

    CrossAttention.forward = forward_ba
    _patch = (CrossAttention, orig_forward)


def disable_bounded_attention():
    global _patch
    if _patch is None:
        return
    cls, orig_fwd = _patch
    cls.forward   = orig_fwd
    _patch        = None


@contextlib.contextmanager
def bounded_attention(model, subjects: List[SubjectMask]):
    enable_bounded_attention(model, subjects)
    try:
        yield
    finally:
        disable_bounded_attention()
