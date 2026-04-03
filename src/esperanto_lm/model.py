"""Model instantiation from config."""

from transformers import LlamaConfig, LlamaForCausalLM


def create_model(config: LlamaConfig) -> LlamaForCausalLM:
    """Create a randomly initialized LLaMA model from a config."""
    return LlamaForCausalLM(config)


def count_parameters(model: LlamaForCausalLM) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
