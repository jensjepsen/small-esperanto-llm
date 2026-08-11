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
        # ESPLLM_LIGER_FLCE=0 turns FLCE off (falls back to HF's standard
        # linear + CE, ~10-15% slower on CE path). Needed when
        # per_device_batch × seq_len > MAX_FUSED_SIZE (32,768) — Liger's
        # chunk kernel has int32 indexing that overflows and crashes with
        # "illegal memory access" past that ceiling. Keep the other Liger
        # kernels (RoPE/RMSNorm/SwiGLU) — they have no such limit.
        _flce = os.getenv("ESPLLM_LIGER_FLCE", "1") != "0"
        apply_liger_kernel_to_llama(
            rope=True, rms_norm=True, swiglu=True,
            fused_linear_cross_entropy=_flce, cross_entropy=not _flce,
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
        "--no-stream",
        action="store_true",
        help="Load --pretokenized-dataset non-streaming (fully materialized). "
             "Use for small corpora (<100M tokens) — avoids epoch-boundary "
             "shuffle crashes on single-shard iterable datasets.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push final model to HF Hub (e.g. 'jensjepsen/esperanto-llm-small')",
    )
    parser.add_argument(
        "--min-doc-tokens", type=int, default=0,
        help="If >0, filter the pretokenized streaming source to only docs "
             "with `len(input_ids) >= min_doc_tokens` before chunking. Useful "
             "for RoPE-extension continued pretrain — filtering to docs "
             "≥ max_length (e.g. 2048) ensures each training chunk is a "
             "coherent long span, not stitched fragments of short docs.",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=0,
        help="If >0, cap the eval_dataset to first N chunks via .take(N)/"
             ".select(range(N)). Use to shrink HF eval_loss cost when the "
             "real signal comes from long_short/mc-logprob callbacks.",
    )
    parser.add_argument(
        "--rope-extend-theta", type=float, default=None,
        help="RoPE-extension continued pretrain: after --from-pretrained "
             "loads the base, bump the model's rope_theta to this value "
             "(e.g. 500000 for a 4× extension), set max_position_embeddings "
             "to the YAML's value, and re-instantiate every LlamaRotaryEmbedding "
             "module so the new theta takes effect. Only meaningful with "
             "--from-pretrained; ignored otherwise.",
    )
    parser.add_argument(
        "--long-short-eval",
        action="store_true",
        help="Attach LongShortPerplexityCallback: measures eval/short_nll "
             "(positions [0, --short-len)) and eval/long_nll (positions "
             "[--short-len, model.max_position_embeddings)) on every eval "
             "step. Use during RoPE-extension continued pretraining to see "
             "both regression and extension progress live in wandb.",
    )
    parser.add_argument(
        "--mc-logprob-eval",
        action="store_true",
        help="Attach MCLogprobCallback: measures eval/sciq_mc_logprob and "
             "eval/citmc_logprob (length-normalized log P scoring) on every "
             "eval step. Use during STEM mid-train to track discrimination "
             "gains without needing chat-template compliance.",
    )
    parser.add_argument("--mc-logprob-n-sciq", type=int, default=200)
    parser.add_argument("--mc-logprob-n-citmc", type=int, default=300)
    parser.add_argument("--mc-logprob-n-arc", type=int, default=1167)
    parser.add_argument(
        "--long-short-eval-docs", type=int, default=128,
        help="Number of held-out docs for the long/short eval (default 128; "
             "32 was too noisy — one outlier-hard doc bin can make the whole "
             "trajectory look like it's regressing when it's actually not).",
    )
    parser.add_argument(
        "--long-short-eval-short-len", type=int, default=512,
        help="Split point between 'short' and 'long' halves (default 512, "
             "matching the base's original max_position_embeddings).",
    )
    parser.add_argument(
        "--long-short-eval-batch-size", type=int, default=4,
        help="Batch size for the long/short eval (default 4).",
    )
    parser.add_argument(
        "--long-short-eval-cache-dir", type=str,
        default="data/long_short_eval_cache",
        help="Where to persist the held-out tokenized docs across restarts.",
    )
    parser.add_argument(
        "--long-short-eval-dataset", type=str,
        default="jensjepsen/danish-pretrain",
        help="HF dataset to sample held-out long docs from (streaming).",
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

    # RoPE extension: raise theta + max_position on the loaded model and
    # rebuild every rotary_emb module so the new frequencies take effect.
    # Must run BEFORE gradient_checkpointing / Liger patches wrap the model,
    # otherwise the rotary_emb replacement won't propagate through the wrapper.
    if args.from_pretrained and args.rope_extend_theta is not None:
        new_theta = args.rope_extend_theta
        new_max = model_config.max_position_embeddings
        console.print(f"[bold yellow]RoPE extension:[/] theta "
                      f"{model.config.rope_theta} → {new_theta}, "
                      f"max_position_embeddings {model.config.max_position_embeddings} → {new_max}")
        model.config.rope_theta = new_theta
        model.config.max_position_embeddings = new_max
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        device = next(model.parameters()).device
        # Transformers ≥4.45 moved rotary_emb to the top-level model.model
        # (one shared instance). Older versions had one per attention layer.
        # Handle both.
        replaced = 0
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = LlamaRotaryEmbedding(model.config).to(device)
            replaced += 1
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "rotary_emb"):
                attn.rotary_emb = LlamaRotaryEmbedding(model.config).to(device)
                replaced += 1
        console.print(f"[bold]Rebuilt {replaced} LlamaRotaryEmbedding module(s) "
                      f"with the new theta.")

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

            streaming = not args.no_stream
            train_parts, test_parts = [], []
            for repo in repos:
                train_parts.append(_ld(repo, split="train", streaming=streaming))
                try:
                    test_parts.append(_ld(repo, split="test", streaming=streaming))
                except Exception:
                    console.print(f"  [dim]{repo}: no test split — skipping[/]")

            # interleave_datasets round-robins across streams; concatenate
            # would exhaust one repo before touching next → bad shuffling.
            # Weights are size-proportional so `all_exhausted` matches
            # "every source seen once" rather than upsampling small ones
            # to the largest's size (default uniform weights would give
            # 3× overexposure to 2M-row supplement vs 93M-row main).
            if len(train_parts) > 1:
                import urllib.parse, urllib.request, json as _json
                sizes = []
                for repo in repos:
                    url = ("https://datasets-server.huggingface.co/size?dataset="
                           + urllib.parse.quote(repo, safe=""))
                    with urllib.request.urlopen(url, timeout=30) as r:
                        data = _json.loads(r.read())
                    n_rows = next(
                        (s["num_rows"] for s in data["size"]["splits"]
                         if s["split"] == "train"),
                        None,
                    )
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

            # Optional long-doc filter (used for RoPE extension). Docs
            # already ≥ max_length yield at least one chunk that's a single
            # contiguous span, not a stitching of short docs.
            if args.min_doc_tokens > 0:
                threshold = args.min_doc_tokens
                console.print(f"[bold]Filtering pretokenized stream to docs "
                              f"≥{threshold} tokens[/]")
                train_tok = train_tok.filter(
                    lambda r: len(r["input_ids"]) >= threshold)
                eval_source = eval_source.filter(
                    lambda r: len(r["input_ids"]) >= threshold)

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
            if args.max_eval_samples > 0:
                eval_dataset = eval_dataset.take(args.max_eval_samples)
                console.print(f"[bold]Capping eval_dataset to {args.max_eval_samples} chunks.[/]")
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
            if args.max_eval_samples > 0 and len(eval_dataset) > args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
                console.print(f"[bold]Capping eval_dataset to {args.max_eval_samples} chunks.[/]")

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

    if args.long_short_eval:
        from esperanto_lm.long_short_eval_callback import LongShortPerplexityCallback
        max_len = model_config.max_position_embeddings
        console.print(f"[bold green]Attaching long/short eval callback:[/] "
                      f"short=[0,{args.long_short_eval_short_len}) "
                      f"long=[{args.long_short_eval_short_len},{max_len})")
        trainer.add_callback(LongShortPerplexityCallback(
            tokenizer=tokenizer,
            cache_dir=args.long_short_eval_cache_dir,
            n_docs=args.long_short_eval_docs,
            max_len=max_len,
            short_len=args.long_short_eval_short_len,
            batch_size=args.long_short_eval_batch_size,
            dataset_name=args.long_short_eval_dataset,
        ))

    if args.mc_logprob_eval:
        from esperanto_lm.mc_logprob_callback import MCLogprobCallback
        console.print(f"[bold green]Attaching MC-logprob eval callback:[/] "
                      f"sciq n={args.mc_logprob_n_sciq}  "
                      f"citmc n={args.mc_logprob_n_citmc}")
        trainer.add_callback(MCLogprobCallback(
            tokenizer=tokenizer,
            n_sciq=args.mc_logprob_n_sciq,
            n_citmc=args.mc_logprob_n_citmc,
            n_arc=args.mc_logprob_n_arc,
        ))

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
