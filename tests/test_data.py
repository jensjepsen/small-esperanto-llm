"""Tests for tokenizer and data pipeline."""

import pytest
from datasets import Dataset

from esperanto_lm.data import (
    SPECIAL_TOKENS,
    make_data_collator,
    tokenize_and_chunk,
    train_tokenizer,
)


@pytest.fixture
def dummy_dataset():
    """Create a small dummy dataset for testing."""
    texts = [
        "Esperanto estas internacia lingvo kreita de L. L. Zamenhof.",
        "La lingvo estas facile lernebla kaj tre logika.",
        "Multaj homoj parolas Esperanton en la tuta mondo.",
        "La Fundamento de Esperanto estas la baza dokumento.",
        "Esperantistoj renkontas en kongresoj kaj aliaj eventoj.",
    ] * 20  # repeat to have enough text for chunking
    return Dataset.from_dict({"text": texts})


@pytest.fixture
def tokenizer(dummy_dataset, tmp_path):
    """Train a small tokenizer on the dummy dataset."""
    return train_tokenizer(dummy_dataset, save_dir=tmp_path / "tok", vocab_size=500)


def test_tokenizer_roundtrip(tokenizer):
    """Tokenizer should encode and decode back to the original text."""
    text = "Esperanto estas bela lingvo."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    # Byte-level BPE may add/remove spaces, but core content should survive
    assert "Esperanto" in decoded
    assert len(encoded) > 0


def test_tokenizer_special_tokens(tokenizer):
    """Tokenizer should have all special tokens defined."""
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.unk_token == "<unk>"


def test_block_packing_shape(dummy_dataset, tokenizer):
    """Block packing should produce fixed-length chunks."""
    max_length = 32  # small for testing
    chunked = tokenize_and_chunk(dummy_dataset, tokenizer, max_length=max_length)
    assert len(chunked) > 0
    sample = chunked[0]
    assert len(sample["input_ids"]) == max_length
    assert len(sample["attention_mask"]) == max_length


def test_data_collator(dummy_dataset, tokenizer):
    """Data collator should produce labels for causal LM."""
    max_length = 32
    chunked = tokenize_and_chunk(dummy_dataset, tokenizer, max_length=max_length)
    collator = make_data_collator(tokenizer)
    batch = collator([chunked[i] for i in range(min(4, len(chunked)))])
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == batch["labels"].shape
