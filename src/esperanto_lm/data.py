"""Dataset download, tokenizer training, tokenization, and data collation."""

import json
from pathlib import Path
from typing import Iterator

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

DATA_DIR = Path("data/eo_wiki")
HPLT_DIR = Path("data/hplt")
GUTENBERG_DIR = Path("data/gutenberg")
MC4_DIR = Path("data/mc4/eo")
FACTOIDS_PATH = Path("/mnt/data2/wikidata5m/eo_factoids/factoid_text.jsonl")
SENTENCES_PATH = Path("data/epo_sentences.tsv")
TOKENIZER_DIR = Path("tokenizer_morpheme")
VOCAB_SIZE = 8_000
MAX_LENGTH = 512
SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>"]


def download_dataset(save_dir: Path = DATA_DIR) -> DatasetDict:
    """Download Esperanto Wikipedia and split into train/validation."""
    if save_dir.exists():
        return load_from_disk(str(save_dir))

    ds = load_dataset("wikimedia/wikipedia", "20231101.eo", split="train")
    splits = ds.train_test_split(test_size=0.05, seed=42)
    dataset = DatasetDict({"train": splits["train"], "test": splits["test"]})
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_dir))
    return dataset


def load_hplt_dataset(hplt_dir: Path = HPLT_DIR) -> Dataset | None:
    """Load HPLT JSONL files into a Dataset."""
    jsonl_files = sorted(hplt_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    ds = load_dataset(
        "json",
        data_files=[str(f) for f in jsonl_files],
        split="train",
    )
    ds = ds.filter(
        lambda x: x.get("filter") != "discard" and bool(x.get("text", "").strip()),
        num_proc=4,
    )
    # Keep only the text column to save memory
    ds = ds.select_columns(["text"])
    return ds


def load_gutenberg_dataset(gutenberg_dir: Path = GUTENBERG_DIR) -> Dataset | None:
    """Load Gutenberg plain text files into a Dataset."""
    txt_files = sorted(gutenberg_dir.glob("*.txt"))
    if not txt_files:
        return None

    texts = []
    for path in txt_files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)

    return Dataset.from_dict({"text": texts})


def load_mc4_dataset(mc4_dir: Path = MC4_DIR) -> Dataset | None:
    """Load mc4 Esperanto dataset from disk."""
    if not mc4_dir.exists():
        return None
    ds = load_from_disk(str(mc4_dir))
    if "text" not in ds.column_names:
        return None
    ds = ds.select_columns(["text"])
    return ds


def load_sentences_dataset(sentences_path: Path = SENTENCES_PATH) -> Dataset | None:
    """Load Tatoeba-style TSV sentences."""
    if not sentences_path.exists():
        return None
    texts = []
    with open(sentences_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                text = parts[2].strip()
                if text:
                    texts.append(text)
    return Dataset.from_dict({"text": texts})


def load_factoids_dataset(factoids_path: Path = FACTOIDS_PATH) -> Dataset | None:
    """Load generated Wikidata factoid paragraphs."""
    if not factoids_path.exists():
        return None
    # Read only the text field to avoid schema conflicts between
    # single-entity (entity_id: str) and comparison (entity_ids: list) records
    texts = []
    with open(factoids_path) as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("text", "").strip():
                texts.append(doc["text"])
    return Dataset.from_dict({"text": texts})


def load_combined_dataset(
    wiki_dir: Path = DATA_DIR,
    hplt_dir: Path = HPLT_DIR,
    gutenberg_dir: Path = GUTENBERG_DIR,
    mc4_dir: Path = MC4_DIR,
    factoids_path: Path = FACTOIDS_PATH,
    sentences_path: Path = SENTENCES_PATH,
    use_wiki: bool = True,
    use_hplt: bool = False,
    use_gutenberg: bool = False,
    use_mc4: bool = False,
    use_factoids: bool = False,
    use_sentences: bool = False,
) -> DatasetDict:
    """Load datasets based on flags, returning train/test splits."""
    base_train = []
    base_test = []

    if use_wiki:
        wiki = download_dataset(wiki_dir)
        base_train.append(wiki["train"])
        base_test.append(wiki["test"])

    extra_train = []
    extra_test = []

    if use_hplt:
        hplt = load_hplt_dataset(hplt_dir)
        if hplt is not None:
            hplt_splits = hplt.train_test_split(test_size=0.05, seed=42)
            extra_train.append(hplt_splits["train"])
            extra_test.append(hplt_splits["test"])

    if use_gutenberg:
        gutenberg = load_gutenberg_dataset(gutenberg_dir)
        if gutenberg is not None:
            gutenberg_splits = gutenberg.train_test_split(test_size=0.05, seed=42)
            extra_train.append(gutenberg_splits["train"])
            extra_test.append(gutenberg_splits["test"])

    if use_mc4:
        mc4 = load_mc4_dataset(mc4_dir)
        if mc4 is not None:
            mc4_splits = mc4.train_test_split(test_size=0.05, seed=42)
            extra_train.append(mc4_splits["train"])
            extra_test.append(mc4_splits["test"])

    if use_factoids:
        factoids = load_factoids_dataset(factoids_path)
        if factoids is not None:
            factoid_splits = factoids.train_test_split(test_size=0.05, seed=42)
            extra_train.append(factoid_splits["train"])
            extra_test.append(factoid_splits["test"])

    if use_sentences:
        sentences = load_sentences_dataset(sentences_path)
        if sentences is not None:
            sentence_splits = sentences.train_test_split(test_size=0.05, seed=42)
            extra_train.append(sentence_splits["train"])
            extra_test.append(sentence_splits["test"])

    all_train = base_train + extra_train
    all_test = base_test + extra_test

    if not all_train:
        raise ValueError("No data sources selected")

    return DatasetDict({
        "train": concatenate_datasets(all_train) if len(all_train) > 1 else all_train[0],
        "test": concatenate_datasets(all_test) if len(all_test) > 1 else all_test[0],
    })


def _corpus_iterator(dataset) -> Iterator[str]:
    """Yield text strings from the dataset for tokenizer training."""
    for example in dataset:
        text = example["text"]
        if text.strip():
            yield text


def train_tokenizer(
    dataset, save_dir: Path = TOKENIZER_DIR, vocab_size: int = VOCAB_SIZE
) -> PreTrainedTokenizerFast:
    """Train a ByteLevel BPE tokenizer on the dataset."""
    save_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        _corpus_iterator(dataset),
        vocab_size=vocab_size,
        min_frequency=3,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.save_model(str(save_dir))

    return _wrap_tokenizer(save_dir, tok)


def _wrap_tokenizer(
    tokenizer_dir: Path, tok: ByteLevelBPETokenizer | None = None
) -> PreTrainedTokenizerFast:
    """Wrap a saved ByteLevelBPE tokenizer as a PreTrainedTokenizerFast."""
    if tok is not None:
        # Build from the live tokenizer object
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok._tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
    else:
        # Load from a previously saved tokenizer.json
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
    tokenizer.save_pretrained(str(tokenizer_dir))
    return tokenizer


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR) -> PreTrainedTokenizerFast:
    """Load a previously saved tokenizer."""
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))


def filter_short_articles(dataset, min_length: int):
    """Remove articles shorter than min_length characters."""
    return dataset.filter(lambda x: len(x["text"]) >= min_length, num_proc=4)


def tokenize_and_chunk(dataset, tokenizer: PreTrainedTokenizerFast, max_length: int = MAX_LENGTH):
    """Tokenize dataset and pack into fixed-length blocks."""

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    def group_texts(examples):
        # Concatenate all token IDs
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop the remainder that doesn't fill a full block
        total_length = (total_length // max_length) * max_length
        result = {
            k: [v[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, v in concatenated.items()
        }
        return result

    chunked = tokenized.map(group_texts, batched=True, num_proc=4)
    return chunked


def make_data_collator(tokenizer: PreTrainedTokenizerFast) -> DataCollatorForLanguageModeling:
    """Create a causal LM data collator (no masking)."""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
