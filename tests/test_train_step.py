"""Sanity check: single forward + backward + optimizer step."""

import torch
import pytest

from esperanto_lm.config import make_llama_config
from esperanto_lm.model import create_model


def test_single_train_step():
    """A single training step should complete without NaN loss."""
    config = make_llama_config("tiny")
    model = create_model(config)
    model.train()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward
    output = model(input_ids=input_ids, labels=labels)
    loss = output.loss

    assert loss is not None
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    # Backward
    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed"

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Second forward to verify model still works after step
    output2 = model(input_ids=input_ids, labels=labels)
    assert not torch.isnan(output2.loss), "Loss is NaN after optimizer step"
