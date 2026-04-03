"""Trainer setup and entry point."""

import argparse

from rich.console import Console
from transformers import AutoModelForCausalLM, Trainer

from esperanto_lm.config import make_llama_config, make_training_args
from esperanto_lm.data import (
    download_dataset,
    filter_short_articles,
    load_combined_dataset,
    load_tokenizer,
    make_data_collator,
    tokenize_and_chunk,
)
from esperanto_lm.evaluate import compute_perplexity, save_perplexity
from esperanto_lm.model import count_parameters, create_model

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train an Esperanto LLaMA model")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Model config to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default=None,
        help="Load model weights from a directory (no optimizer state)",
    )
    parser.add_argument(
        "--min-article-length",
        type=int,
        default=0,
        help="Drop articles shorter than this many characters",
    )
    parser.add_argument(
        "--use-hplt",
        action="store_true",
        help="Include HPLT web corpus data (must download first)",
    )
    parser.add_argument(
        "--use-gutenberg",
        action="store_true",
        help="Include Gutenberg books (must download first)",
    )
    parser.add_argument(
        "--use-mc4",
        action="store_true",
        help="Include mc4 web corpus (must download first)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push final model to HF Hub (e.g. 'jensjepsen/esperanto-llm-small')",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"runs/{args.config}"

    console.print(f"[bold green]Loading config:[/] {args.config}")
    model_config = make_llama_config(args.config)
    training_args = make_training_args(args.config, output_dir, hub_model_id=args.push_to_hub)

    console.print("[bold green]Loading tokenizer...")
    tokenizer = load_tokenizer()
    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id

    console.print("[bold green]Creating model...")
    if args.from_pretrained:
        console.print(f"[bold]Loading weights from:[/] {args.from_pretrained}")
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained)
    else:
        model = create_model(model_config)
    n_params = count_parameters(model)
    console.print(f"[bold]Parameters:[/] {n_params:,}")

    console.print("[bold green]Loading and tokenizing dataset...")
    dataset = load_combined_dataset(use_hplt=args.use_hplt, use_gutenberg=args.use_gutenberg, use_mc4=args.use_mc4)
    console.print(f"[bold]Train examples:[/] {len(dataset['train']):,}")
    console.print(f"[bold]Test examples:[/] {len(dataset['test']):,}")
    if args.min_article_length > 0:
        before = len(dataset["train"])
        dataset["train"] = filter_short_articles(dataset["train"], args.min_article_length)
        dataset["test"] = filter_short_articles(dataset["test"], args.min_article_length)
        after = len(dataset["train"])
        console.print(f"[bold]Filtered articles:[/] {before:,} -> {after:,} (min {args.min_article_length} chars)")
    max_length = model_config.max_position_embeddings
    console.print(f"[bold]Chunk length:[/] {max_length}")
    train_dataset = tokenize_and_chunk(dataset["train"], tokenizer, max_length=max_length)
    eval_dataset = tokenize_and_chunk(dataset["test"], tokenizer, max_length=max_length)

    data_collator = make_data_collator(tokenizer)

    console.print("[bold green]Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    console.print("[bold green]Evaluating...")
    ppl = compute_perplexity(trainer)
    console.print(f"[bold]Perplexity:[/] {ppl:.2f}")
    save_perplexity(args.config, ppl)

    console.print("[bold green]Saving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    if args.push_to_hub:
        console.print(f"[bold green]Pushing final model to HF Hub:[/] {args.push_to_hub}")
        trainer.push_to_hub()

    console.print("[bold green]Done!")


if __name__ == "__main__":
    main()
