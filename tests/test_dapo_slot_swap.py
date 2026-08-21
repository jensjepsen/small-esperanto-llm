"""Unit tests for DAPO fresh-prompts slot-swap.

Covers the n_groups>1 case that best-of-N was unable to serve. The pure-
helpers module (`dapo_slot_swap`) is tested end-to-end without needing the
full trainer / model / cuda stack.
"""
from __future__ import annotations

import pytest
import torch

from esperanto_lm.dapo_slot_swap import active_mask, slot_swap, swap_slot


# ---------------------------------------------------------------------- helpers


def _mk(adv, prompt_len, comp_len, *, seed=0):
    """Fabricate a TRL 0.18-shaped result dict.

    `adv` is a python list; its length must be a multiple of num_gens. Every
    tensor is filled with a distinct pattern derived from `seed` so the test
    can check "did the slot from src ACTUALLY overwrite the dst rows?"."""
    g = torch.Generator().manual_seed(seed)
    n = len(adv)
    return {
        "prompt_ids": torch.randint(1, 100, (n, prompt_len), generator=g),
        "prompt_mask": torch.ones((n, prompt_len), dtype=torch.long),
        "completion_ids": torch.randint(1, 100, (n, comp_len), generator=g),
        "completion_mask": torch.ones((n, comp_len), dtype=torch.long),
        "advantages": torch.tensor(adv, dtype=torch.float32),
        "old_per_token_logps": torch.randn((n, comp_len), generator=g),
    }


# ---------------------------------------------------------------------- active_mask


def test_active_mask_all_dead():
    r = _mk([0.0, 0.0, 0.0, 0.0], prompt_len=5, comp_len=7)
    m, g = active_mask(r, num_gens=2)
    assert g == 2
    assert m.tolist() == [False, False]


def test_active_mask_mixed():
    # 3 groups × 2 gens: g0 dead, g1 active, g2 dead
    r = _mk([0.0, 0.0, 0.9, -0.9, 1e-9, -1e-9], prompt_len=3, comp_len=4)
    m, g = active_mask(r, num_gens=2)
    assert g == 3
    assert m.tolist() == [False, True, False]


def test_active_mask_none_result():
    m, g = active_mask(None, num_gens=2)
    assert m is None and g == 0


def test_active_mask_missing_advantages():
    m, g = active_mask({"prompt_ids": torch.zeros(4, 2)}, num_gens=2)
    assert m is None and g == 0


def test_active_mask_zero_groups_short_batch():
    # fewer rows than num_gens -> zero groups (defensive branch)
    r = _mk([0.5], prompt_len=2, comp_len=2)
    m, g = active_mask(r, num_gens=4)
    assert g == 0 and m is None


# ---------------------------------------------------------------------- swap_slot


def test_swap_slot_matching_lens():
    # 2 groups × 2 gens, matching prompt/comp lens. Swap slot 0.
    dst = _mk([0.0, 0.0, 1.0, -1.0], prompt_len=3, comp_len=4, seed=1)
    src = _mk([0.5, -0.5, 0.0, 0.0], prompt_len=3, comp_len=4, seed=2)

    # snapshot slot 0 from src (rows [0:2]) BEFORE we mutate
    exp_pids = src["prompt_ids"][0:2].clone()
    exp_cids = src["completion_ids"][0:2].clone()
    exp_adv = src["advantages"][0:2].clone()

    # snapshot slot 1 from dst so we prove it's untouched
    unchanged_pids = dst["prompt_ids"][2:4].clone()
    unchanged_adv = dst["advantages"][2:4].clone()

    swap_slot(dst, src, slot=0, num_gens=2)

    assert torch.equal(dst["prompt_ids"][0:2], exp_pids)
    assert torch.equal(dst["completion_ids"][0:2], exp_cids)
    assert torch.equal(dst["advantages"][0:2], exp_adv)
    # slot 1 must be untouched
    assert torch.equal(dst["prompt_ids"][2:4], unchanged_pids)
    assert torch.equal(dst["advantages"][2:4], unchanged_adv)


def test_swap_slot_pads_dst_up_for_longer_src_completion():
    # src has a longer completion — dst must grow (right-pad on completions).
    dst = _mk([0.0, 0.0], prompt_len=3, comp_len=4, seed=3)
    src = _mk([0.7, -0.7], prompt_len=3, comp_len=6, seed=4)

    src_cids_slot = src["completion_ids"][0:2].clone()
    src_cmask_slot = src["completion_mask"][0:2].clone()
    src_logps_slot = src["old_per_token_logps"][0:2].clone()

    swap_slot(dst, src, slot=0, num_gens=2)

    assert dst["completion_ids"].shape[1] == 6
    assert dst["completion_mask"].shape[1] == 6
    assert dst["old_per_token_logps"].shape[1] == 6
    # slot rows must equal src verbatim (no re-padding on right side needed)
    assert torch.equal(dst["completion_ids"][0:2], src_cids_slot)
    assert torch.equal(dst["completion_mask"][0:2], src_cmask_slot)
    assert torch.equal(dst["old_per_token_logps"][0:2], src_logps_slot)


def test_swap_slot_pads_src_up_for_longer_dst_completion():
    # dst longer — src is right-padded up to dst's length before slice.
    dst = _mk([0.0, 0.0, 0.4, 0.4], prompt_len=3, comp_len=6, seed=5)
    src = _mk([0.9, -0.9, 0.0, 0.0], prompt_len=3, comp_len=4, seed=6)

    src_cids = src["completion_ids"][0:2].clone()  # width 4
    swap_slot(dst, src, slot=0, num_gens=2)

    # dst keeps width 6
    assert dst["completion_ids"].shape[1] == 6
    # first 4 cols of swapped rows come from src, last 2 are pad=0 (right-pad)
    assert torch.equal(dst["completion_ids"][0:2, :4], src_cids)
    assert torch.all(dst["completion_ids"][0:2, 4:] == 0)
    # completion_mask likewise pad=0 -> exact match of the pattern
    assert torch.all(dst["completion_mask"][0:2, 4:] == 0)


def test_swap_slot_pads_prompt_on_LEFT():
    # dst.prompt_len < src.prompt_len — dst must LEFT-grow.
    dst = _mk([0.0, 0.0], prompt_len=3, comp_len=4, seed=7)
    src = _mk([0.6, -0.6], prompt_len=5, comp_len=4, seed=8)

    src_pids_slot = src["prompt_ids"][0:2].clone()
    swap_slot(dst, src, slot=0, num_gens=2)
    assert dst["prompt_ids"].shape[1] == 5
    # swapped slot must equal src's slot verbatim (no re-padding)
    assert torch.equal(dst["prompt_ids"][0:2], src_pids_slot)


def test_swap_slot_pads_src_prompt_on_LEFT():
    # dst.prompt_len > src.prompt_len — src is LEFT-padded up.
    dst = _mk([0.0, 0.0], prompt_len=6, comp_len=4, seed=9)
    src = _mk([0.5, -0.5], prompt_len=3, comp_len=4, seed=10)

    src_pids = src["prompt_ids"][0:2].clone()  # width 3
    swap_slot(dst, src, slot=0, num_gens=2)
    assert dst["prompt_ids"].shape[1] == 6
    # LEFT-pad means the first (6-3)=3 cols are pad=0, tail matches src
    assert torch.all(dst["prompt_ids"][0:2, :3] == 0)
    assert torch.equal(dst["prompt_ids"][0:2, 3:], src_pids)


def test_swap_slot_missing_field_in_src_is_silently_skipped():
    # If src drops a field (e.g. old_per_token_logps=None when beta=0), we
    # must not crash; that field just stays as-is in dst.
    dst = _mk([0.0, 0.0], prompt_len=3, comp_len=4, seed=11)
    src = _mk([0.5, -0.5], prompt_len=3, comp_len=4, seed=12)
    src["old_per_token_logps"] = None
    dst_lp_before = dst["old_per_token_logps"].clone()
    swap_slot(dst, src, slot=0, num_gens=2)
    assert torch.equal(dst["old_per_token_logps"], dst_lp_before)


# ---------------------------------------------------------------------- slot_swap orchestration


def test_slot_swap_n_groups_gt_1_fills_all_deads():
    # 4 groups × 2 gens. best: g0 dead, g1 active, g2 dead, g3 dead.
    # src:  g0 active, g1 dead,   g2 active, g3 dead.
    # After swap: best should have g0/g1/g2 active, g3 still dead.
    best = _mk([0.0, 0.0, 0.9, -0.9, 0.0, 0.0, 0.0, 0.0],
               prompt_len=3, comp_len=4, seed=20)
    src  = _mk([0.6, -0.6, 0.0, 0.0, 0.8, -0.8, 0.0, 0.0],
               prompt_len=3, comp_len=4, seed=21)

    # Snapshot expected rows.
    exp_g0 = src["prompt_ids"][0:2].clone()
    exp_g2 = src["prompt_ids"][4:6].clone()
    keep_g1 = best["prompt_ids"][2:4].clone()
    keep_g3 = best["prompt_ids"][6:8].clone()
    exp_adv_g0 = src["advantages"][0:2].clone()
    exp_adv_g2 = src["advantages"][4:6].clone()

    n_swapped = slot_swap(best, src, num_gens=2)
    assert n_swapped == 2

    assert torch.equal(best["prompt_ids"][0:2], exp_g0)
    assert torch.equal(best["prompt_ids"][2:4], keep_g1)   # untouched active
    assert torch.equal(best["prompt_ids"][4:6], exp_g2)
    assert torch.equal(best["prompt_ids"][6:8], keep_g3)   # untouched dead
    # advantages carried over — this is critical since group-relative
    # advantage was computed on src's group not best's group
    assert torch.equal(best["advantages"][0:2], exp_adv_g0)
    assert torch.equal(best["advantages"][4:6], exp_adv_g2)

    # active_mask after swap: g0=T, g1=T, g2=T, g3=F
    m_after, _ = active_mask(best, num_gens=2)
    assert m_after.tolist() == [True, True, True, False]


def test_slot_swap_all_best_active_is_noop():
    # If every group already active, slot_swap must not touch anything.
    best = _mk([0.5, -0.5, 0.7, -0.7], prompt_len=3, comp_len=4, seed=30)
    src = _mk([0.9, -0.9, 0.8, -0.8], prompt_len=3, comp_len=4, seed=31)
    before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in best.items()}
    n = slot_swap(best, src, num_gens=2)
    assert n == 0
    for k, v in before.items():
        assert torch.equal(best[k], v), f"field {k} was mutated"


def test_slot_swap_ngroups_1_matches_best_of_N_semantics():
    # n_groups==1: this is the original case. Dead best + active src should
    # produce a fully-swapped best (equivalent to whole-batch replacement).
    best = _mk([0.0, 0.0, 0.0, 0.0], prompt_len=3, comp_len=4, seed=40)  # 1 group × 4 gens
    src  = _mk([0.7, -0.7, 0.3, -0.3], prompt_len=3, comp_len=4, seed=41)
    n = slot_swap(best, src, num_gens=4)
    assert n == 1
    assert torch.equal(best["advantages"], src["advantages"])
    assert torch.equal(best["prompt_ids"], src["prompt_ids"])
    assert torch.equal(best["completion_ids"], src["completion_ids"])


def test_slot_swap_shape_mismatch_returns_zero():
    # Fresh batch had different n_groups (e.g. drop_last on data loader).
    # Refuse to swap; caller falls back to best-of-N.
    best = _mk([0.0, 0.0, 0.0, 0.0], prompt_len=3, comp_len=4, seed=50)   # 2 groups × 2
    src  = _mk([0.5, -0.5, 0.9, -0.9, 0.3, -0.3], prompt_len=3, comp_len=4, seed=51)  # 3 groups × 2
    before_adv = best["advantages"].clone()
    n = slot_swap(best, src, num_gens=2)
    assert n == 0
    assert torch.equal(best["advantages"], before_adv)


def test_slot_swap_updates_provided_masks_in_place():
    # If caller passes best_mask, our updates must be reflected there so
    # they see when the loop can early-exit.
    best = _mk([0.0, 0.0, 0.0, 0.0], prompt_len=3, comp_len=4, seed=60)  # 2 groups × 2
    src  = _mk([0.7, -0.7, 0.9, -0.9], prompt_len=3, comp_len=4, seed=61)
    m_best, _ = active_mask(best, num_gens=2)
    m_src, _ = active_mask(src, num_gens=2)
    assert m_best.tolist() == [False, False]
    assert m_src.tolist() == [True, True]
    n = slot_swap(best, src, num_gens=2, best_mask=m_best, src_mask=m_src)
    assert n == 2
    assert m_best.tolist() == [True, True]


def test_slot_swap_partial_when_src_only_has_some():
    # src active on slots [0, 2], dead on [1, 3]. best dead everywhere.
    # After swap best active is [T, F, T, F]; needs another src to finish.
    best = _mk([0.0]*8, prompt_len=3, comp_len=4, seed=70)
    src  = _mk([0.9, -0.9, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0],
               prompt_len=3, comp_len=4, seed=71)
    n = slot_swap(best, src, num_gens=2)
    assert n == 2
    m_after, _ = active_mask(best, num_gens=2)
    assert m_after.tolist() == [True, False, True, False]


def test_slot_swap_composition_two_srcs_fills_all():
    # simulate two DAPO attempts. best dead everywhere, src1 fills [0,2],
    # src2 fills [1,3]. Composition should end fully active.
    best = _mk([0.0]*8, prompt_len=3, comp_len=4, seed=80)
    src1 = _mk([0.7, -0.7, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0],
               prompt_len=3, comp_len=4, seed=81)
    src2 = _mk([0.0, 0.0, 0.9, -0.9, 0.0, 0.0, 0.3, -0.3],
               prompt_len=3, comp_len=4, seed=82)
    slot_swap(best, src1, num_gens=2)
    slot_swap(best, src2, num_gens=2)
    m_after, _ = active_mask(best, num_gens=2)
    assert m_after.tolist() == [True, True, True, True]
    # slot 0 from src1, slot 1 from src2
    assert torch.equal(best["prompt_ids"][0:2], src1["prompt_ids"][0:2])
    assert torch.equal(best["prompt_ids"][2:4], src2["prompt_ids"][2:4])
    assert torch.equal(best["prompt_ids"][4:6], src1["prompt_ids"][4:6])
    assert torch.equal(best["prompt_ids"][6:8], src2["prompt_ids"][6:8])
