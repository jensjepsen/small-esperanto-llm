"""Tests for model instantiation and forward pass."""

import torch
import pytest

from esperanto_lm.config import make_llama_config
from esperanto_lm.model import count_parameters, create_model


@pytest.fixture
def tiny_config():
    return make_llama_config("tiny")


@pytest.fixture
def tiny_model(tiny_config):
    return create_model(tiny_config)


def test_model_instantiates(tiny_model):
    """Model should instantiate without errors."""
    assert tiny_model is not None


def test_forward_pass_shape(tiny_model, tiny_config):
    """Forward pass should return logits of correct shape."""
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        output = tiny_model(input_ids)
    assert output.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)


def test_parameter_count(tiny_model):
    """Parameter count should be within 10% of expected ~12M."""
    n_params = count_parameters(tiny_model)
    expected = 8_600_000
    assert abs(n_params - expected) / expected < 0.10, (
        f"Parameter count {n_params:,} is not within 10% of expected {expected:,}"
    )
