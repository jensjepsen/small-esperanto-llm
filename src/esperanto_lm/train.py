"""Trainer setup and entry point."""

import argparse
import os
from pathlib import Path

# Apply Liger kernels (RMSNorm, SwiGLU, RoPE, fused LM head + CE) before
# any Llama model is created. Saves ~30-40% wall time + ~40% VRAM on H100.
# Skipped via ESPLLM_NO_LIGER=1 if a future env-bump breaks compatibility.
if os.getenv("ESPLLM_NO_LIGER") != "1":
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        # FLCE on: fuses LM-head linear + CE into one kernel, avoids
        # materializing the full (B, T, V) logits tensor. ~30% additional
        # VRAM savings on top of base Liger + small wall win. Caveat: it
        # replaces the LM-head module → param-group shape changes →
        # breaks `--resume-from-checkpoint` for ckpts saved without Liger.
        # We use `--from-pretrained` (weights only, fresh optimizer) so
        # this is safe; flip back to False if you need to resume an
        # optimizer-bearing ckpt from a non-Liger run.
        apply_liger_kernel_to_llama(
            rope=True, rms_norm=True, swiglu=True,
            fused_linear_cross_entropy=True, cross_entropy=False,
        )
    except ImportError:
        pass  # liger optional — train without if not installed

import torch
# Force flash + mem-efficient SDPA backends; cuDNN's frontend has been flaky.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Enable TF32 for matmul + cuDNN on Ampere+ (cap >= 8). Free speedup on
# the few fp32 paths that remain when bf16 is on; no effect on Pascal/Volta.
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from rich.console import Console
from transformers import AutoModelForCausalLM, Trainer

from esperanto_lm.config import make_llama_config, make_training_args
from datasets import concatenate_datasets

from esperanto_lm.data import (
    load_benchmark_qa_dataset,
    chunk_dataset,
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
        help="Model config to use (a YAML file in configs/, e.g. tiny|small|medium|large|large_continue)",
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
        "--tokenizer",
        type=str,
        default="tokenizer_morpheme",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--min-article-length",
        type=int,
        default=0,
        help="Drop articles shorter than this many characters",
    )
    parser.add_argument(
        "--no-wiki",
        action="store_true",
        help="Exclude Wikipedia data",
    )
    parser.add_argument(
        "--use-hplt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include HPLT web corpus data (default: on)",
    )
    parser.add_argument(
        "--use-gutenberg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Gutenberg books (default: on)",
    )
    parser.add_argument(
        "--use-mc4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include mc4 web corpus",
    )
    parser.add_argument(
        "--use-factoids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Wikidata factoid paragraphs (default: on)",
    )
    parser.add_argument(
        "--use-sentences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Tatoeba sentence corpus (default: on)",
    )
    parser.add_argument(
        "--use-tekstaro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Tekstaro de Esperanto curated corpus (default: on)",
    )
    parser.add_argument(
        "--use-liberafolio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Libera Folio EO journalism (default: on)",
    )
    parser.add_argument(
        "--use-fineweb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include FineWeb-2 epo_Latn web corpus (default: on)",
    )
    parser.add_argument(
        "--use-wiki-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include v6-MT-translated EN→EO Wikipedia gaps "
             "(jensjepsen/esperanto-wiki-gaps; default: on)",
    )
    parser.add_argument(
        "--use-wikisource",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include native EO Wikisource / Vikifontaro literature "
             "(jensjepsen/esperanto-wikisource). Default: off — HF dataset "
             "is not yet published because MediaWiki extracts API returns "
             "empty for most Wikisource pages (they transclude PDF text via "
             "<pages/> templates). Needs bulk-XML-dump or parse-API+HTML-strip "
             "extraction. Set default=True once dataset is published.",
    )
    parser.add_argument(
        "--use-algebra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include procedural sympy-generated algebra equations + "
             "solution chains (jensjepsen/esperanto-algebra-pretrain; "
             "default: on)",
    )
    parser.add_argument(
        "--use-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include benchmark train splits as 'Demando: ... Respondo: ...' "
             "Q/A pairs (default: on). Tokenized separately from the main "
             "corpus so adding/removing doesn't invalidate the big-corpus "
             "tokenization cache. Includes sciq/copa/piqa/mmlu(aux_train)/"
             "triviaqa train splits — val/test held out for eval.",
    )
    parser.add_argument(
        "--pretokenized-dataset",
        type=str,
        nargs="+",
        default=None,
        help="One or more HF repos of pre-tokenized pretraining datasets to "
             "use INSTEAD of loading the raw sources and running tokenize. "
             "All datasets must share the same schema (input_ids + "
             "attention_mask). Multiple repos are concatenated at load time — "
             "useful for combining main + supplement + math shards. Chunking "
             "still runs locally (cheap). Skips the ~50-min morpheme "
             "tokenization phase.",
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

    console.print(f"[bold green]Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(Path(args.tokenizer))
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

    max_length = model_config.max_position_embeddings
    console.print(f"[bold]Chunk length:[/] {max_length}")
    # Serialize dataset prep across DDP ranks: rank 0 builds the HF
    # cache, other ranks wait then read it. Without this, each rank
    # repeats the multi-GB tokenize+chunk independently, doubling
    # workspace cache usage and wall time.
    from accelerate import PartialState
    accel_state = PartialState()
    with accel_state.main_process_first():
        if args.pretokenized_dataset:
            # Standard HF pattern for large pretokenized corpora: streaming.
            # Trainer only iterates the dataset — never needs random access —
            # so streaming IterableDatasets are perfect. Zero in-memory
            # materialization, zero arrow cache build, zero setup time.
            # Chunk map runs lazily as batches are pulled.
            from datasets import load_dataset as _ld
            from datasets import interleave_datasets
            import os as _os
            _os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

            repos = args.pretokenized_dataset
            console.print(f"[bold green]Loading pre-tokenized (streaming):[/] "
                          f"{len(repos)} repo(s): {repos}")

            train_parts, test_parts = [], []
            for repo in repos:
                train_parts.append(_ld(repo, split="train", streaming=True))
                try:
                    test_parts.append(_ld(repo, split="test", streaming=True))
                except Exception:
                    console.print(f"  [dim]{repo}: no test split — skipping[/]")

            # interleave_datasets round-robins across streams; concatenate
            # would exhaust one repo before touching next → bad shuffling.
            # Weights are size-proportional so `all_exhausted` matches
            # "every source seen once" rather than upsampling small ones
            # to the largest's size (default uniform weights would give
            # 3× overexposure to 2M-row supplement vs 93M-row main).
            if len(train_parts) > 1:
                from huggingface_hub import HfApi
                hf_api = HfApi()
                sizes = []
                for repo in repos:
                    info = hf_api.dataset_info(repo)
                    ds_info = (getattr(info.card_data, "dataset_info", None)
                               if info.card_data else None)
                    if isinstance(ds_info, dict):
                        ds_info = [ds_info]
                    n_rows = None
                    for cfg in (ds_info or []):
                        for s in cfg.get("splits", []):
                            if s.get("name") == "train":
                                n_rows = s.get("num_examples")
                                break
                    if not n_rows:
                        raise RuntimeError(f"No train row count for {repo}")
                    sizes.append(n_rows)
                total = sum(sizes)
                probabilities = [s / total for s in sizes]
                console.print(
                    "[bold]Interleave weights (size-proportional):[/] "
                    + ", ".join(f"{r.split('/')[-1]}={p:.3f}" for r, p in zip(repos, probabilities))
                )
                train_tok = interleave_datasets(train_parts,
                                                probabilities=probabilities,
                                                stopping_strategy="all_exhausted")
            else:
                train_tok = train_parts[0]

            if test_parts:
                eval_source = (interleave_datasets(test_parts, stopping_strategy="all_exhausted")
                               if len(test_parts) > 1 else test_parts[0])
            else:
                # No test split anywhere → take first N rows of train stream.
                eval_source = train_tok.take(1000)
                # Advance train past the same rows so we don't train on eval.
                train_tok = train_tok.skip(1000)

            # chunk_dataset uses .map(num_proc=...) which is incompatible
            # with streaming. For streaming, apply a lightweight per-batch
            # chunker that only uses batched=True (no num_proc).
            def _chunk_stream(examples):
                from itertools import chain
                concat = {k: list(chain.from_iterable(examples[k])) for k in examples}
                total = (len(concat["input_ids"]) // max_length) * max_length
                return {k: [v[i:i + max_length] for i in range(0, total, max_length)]
                        for k, v in concat.items()}

            train_dataset = train_tok.map(_chunk_stream, batched=True, batch_size=1000)
            eval_dataset = eval_source.map(_chunk_stream, batched=True, batch_size=1000)
            console.print("[bold]Streaming pretokenized loaded (lazy).[/]")
        else:
            console.print("[bold green]Loading and tokenizing dataset...")
            dataset = load_combined_dataset(
                use_wiki=not args.no_wiki, use_hplt=args.use_hplt,
                use_gutenberg=args.use_gutenberg, use_mc4=args.use_mc4,
                use_factoids=args.use_factoids, use_sentences=args.use_sentences,
                use_tekstaro=args.use_tekstaro, use_liberafolio=args.use_liberafolio,
                use_fineweb=args.use_fineweb,
                use_wiki_gaps=args.use_wiki_gaps,
                use_wikisource=args.use_wikisource,
                use_algebra=args.use_algebra,
                min_article_length=args.min_article_length,
            )
            console.print(f"[bold]Train examples:[/] {len(dataset['train']):,}")
            console.print(f"[bold]Test examples:[/] {len(dataset['test']):,}")
            train_dataset = tokenize_and_chunk(dataset["train"], tokenizer, max_length=max_length)
            eval_dataset = tokenize_and_chunk(dataset["test"], tokenizer, max_length=max_length)

        if args.use_benchmarks:
            console.print("[bold green]Loading benchmark Q/A pairs...")
            bench = load_benchmark_qa_dataset()
            if bench is not None:
                console.print(f"[bold]Benchmark examples:[/] {len(bench):,}")
                bench_tok = tokenize_and_chunk(bench, tokenizer, max_length=max_length)
                train_dataset = concatenate_datasets([train_dataset, bench_tok])
                console.print(f"[bold]Combined train (post-bench):[/] {len(train_dataset):,}")

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
