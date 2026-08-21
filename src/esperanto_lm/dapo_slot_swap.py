"""DAPO dynamic-sampling slot-swap helpers.

TRL's `_generate_and_score_completions` returns a batch of `n_groups*num_gens`
rollouts. DAPO retries only help when a group has non-zero-std rewards. When
`n_groups==1`, "best of N attempts" is equivalent to the paper's slot-swap
(you either keep the single group or replace it wholesale). For `n_groups>1`,
best-of-N is strictly weaker: dead groups A,_ + _,B in two attempts leaves
one dead in either "best" choice — slot-swap would fill both.

These helpers do the per-slot swap and pad reconciliation across attempts.
Pure functions so they unit-test without a trainer / model / cuda.

TRL 0.18 pad conventions (as of grpo_trainer.py):
  prompt_ids / prompt_mask         -> LEFT-padded  (pad_token_id / 0 mask)
  completion_ids / completion_mask -> RIGHT-padded (pad_token_id / 0 mask)
  old_per_token_logps              -> RIGHT-padded (0.0)
  advantages                       -> 1D per row, no padding
"""
from __future__ import annotations

from typing import Optional

import torch


# Per-field pad side + pad value. "left" = grow on the left (front of seq),
# "right" = grow on the right (back of seq). If a field isn't listed we
# fall back to right-pad with 0 — safe default for masks/logps.
FIELD_PAD_SPEC = {
    "prompt_ids": ("left", 0),
    "prompt_mask": ("left", 0),
    "completion_ids": ("right", 0),
    "completion_mask": ("right", 0),
    "old_per_token_logps": ("right", 0.0),
}


def active_mask(result: Optional[dict], num_gens: int):
    """(bool_mask[n_groups], n_groups). Group is 'active' iff any of its
    num_gens advantages exceeds 1e-6 in abs value. Returns (None, 0) if
    result/advantages missing."""
    if result is None:
        return None, 0
    adv = result.get("advantages")
    if adv is None:
        return None, 0
    n = adv.shape[0]
    g = n // num_gens
    if g == 0:
        return None, 0
    adv_g = adv[:g * num_gens].view(g, num_gens)
    return (adv_g.abs() > 1e-6).any(dim=1), g


def _pad_to(t: torch.Tensor, target_len: int, side: str, pad_value) -> torch.Tensor:
    """Pad a 2D tensor along dim=1 up to target_len on the given side."""
    cur = t.shape[1]
    if cur == target_len:
        return t
    if cur > target_len:
        raise ValueError(f"cannot pad down: cur={cur} target={target_len}")
    pad_shape = (t.shape[0], target_len - cur)
    pad_block = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    if side == "left":
        return torch.cat([pad_block, t], dim=1)
    return torch.cat([t, pad_block], dim=1)


def swap_slot(dst: dict, src: dict, slot: int, num_gens: int) -> None:
    """In-place: overwrite dst's group-`slot` rows with src's group-`slot`
    rows. Reconciles pad-length differences on 2D fields via FIELD_PAD_SPEC.

    Note: dst is mutated. 2D tensors that need growing are REPLACED in dst
    (a new tensor is allocated); 1D tensors and same-length 2D tensors have
    only the sliced rows overwritten."""
    a = slot * num_gens
    b = a + num_gens
    for k in list(dst.keys()):
        d, s = dst.get(k), src.get(k)
        if d is None or s is None:
            continue
        if d.ndim == 1 and s.ndim == 1:
            d[a:b] = s[a:b]
            continue
        if d.ndim == 2 and s.ndim == 2:
            side, pv = FIELD_PAD_SPEC.get(k, ("right", 0))
            dl, sl = d.shape[1], s.shape[1]
            if dl < sl:
                # grow dst — must replace the reference so subsequent ops see the new shape
                d = _pad_to(d, sl, side, pv)
                dst[k] = d
            elif sl < dl:
                s = _pad_to(s, dl, side, pv)
            d[a:b] = s[a:b]
            continue
        # Unknown rank — leave untouched. TRL 0.18 output has only 1D/2D
        # tensors so this shouldn't fire.


def slot_swap(best: dict, src: dict, num_gens: int,
              best_mask=None, src_mask=None) -> int:
    """For every slot dead in best but active in src, swap src's slot into
    best. Mutates best in place and updates best_mask (if provided). Returns
    number of slots swapped."""
    if best_mask is None:
        best_mask, _ = active_mask(best, num_gens)
    if src_mask is None:
        src_mask, _ = active_mask(src, num_gens)
    if best_mask is None or src_mask is None:
        return 0
    if best_mask.numel() != src_mask.numel():
        # Shape mismatch (fresh batch had different n_groups). Refuse to
        # slot-swap; caller can fall back to whole-batch replacement.
        return 0
    n = best_mask.numel()
    swapped = 0
    for slot in range(n):
        if not bool(best_mask[slot]) and bool(src_mask[slot]):
            swap_slot(best, src, slot, num_gens)
            best_mask[slot] = True
            swapped += 1
    return swapped
