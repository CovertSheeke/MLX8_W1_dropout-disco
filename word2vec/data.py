import logging
import os
from pathlib import Path
from typing import Union
import urllib.request
from zipfile import ZipFile

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

from tokeniser import build_vocab, tokenise, get_tokens_as_indices


logger = logging.getLogger(__name__)


DATA_DIR = os.getenv("DATA_DIR", ".data")
TRAIN_PROCESSED_FILENAME = "hn_posts_train_processed.parquet"
TEXT8_ZIP_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_FILENAME = "text8"
TEXT8_EXPECTED_LENGTH = 100_000_000  # text8 is expected to be 100 million chars

# manually set env vars also used in data ingest (see .env.example / db-utils.py)
TITLES_FILE = "hn_posts_titles.parquet"
MINIMAL_FETCH_ONLY_TITLES = True


# custom map-style dataset for CBOW
class CBOWDataset(Dataset):
    """
    Map-style dataset so DataLoader can shuffle & use multiple workers.
    Keeps *only* the integer-encoded corpus in RAM.
    """

    def __init__(self, int_tokens: list[int], context_size: int = 3) -> None:
        assert context_size > 0
        self.tokens = int_tokens
        self.context = context_size

    def __len__(self) -> int:
        # we cannot form a full context window at the edge
        return max(0, len(self.tokens) - 2 * self.context)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # shift because we skipped the first `context` tokens
        # ctx is the context window around the tgt (target) token
        tgt = idx + self.context
        ctx = (
            self.tokens[tgt - self.context : tgt]  # left side of tgt context
            + self.tokens[tgt + 1 : tgt + 1 + self.context]  # right side
        )
        return (
            torch.tensor(ctx, dtype=torch.long),  # shape is [2 * context]
            torch.tensor(self.tokens[tgt], dtype=torch.long),  # 0D scalar tensor
        )


def build_cbow_dataset(
    context_size: int = 3,
    min_freq: int = 5,
    subsampling_threshold: float = 1e-5,
    include_hn_titles: bool = True,
) -> tuple[CBOWDataset, dict]:
    # TODO: also get comments from HN, to be appended to the corpus ??
    hn_titles = ""
    if include_hn_titles:
        if MINIMAL_FETCH_ONLY_TITLES:
            parquet_path = os.path.join(DATA_DIR, TITLES_FILE)
        else:
            parquet_path = os.path.join(DATA_DIR, TRAIN_PROCESSED_FILENAME)
        logger.info(f"Loading data from {parquet_path}...")
        hn_posts = pd.read_parquet(parquet_path)
        # show some basic info about the data we pulled from parquet
        logger.info(
            f"Found {len(hn_posts)} records at {parquet_path}: {hn_posts.info()}"
        )
        # pull out all titles and concatenate into single string, separated by whitespace
        hn_titles = " ".join(hn_posts["title"].tolist())
    else:
        logger.info("Skipping HN titles, using only text8 corpus.")

    # ensure text8 corpus is downloaded to data dir
    text8_path = get_text8(cache_dir=DATA_DIR)

    # TODO: consider keeping titles separate, rather than concatening them all and losing boundaries
    # build the big string
    with open(text8_path, "r", encoding="utf-8") as f:
        corpus_txt = f.read() + " " + hn_titles

    tokens = tokenise(corpus_txt)
    logger.info(f"Produced tokenised corpus of {len(tokens)} tokens.")

    # build vocab and integer-encoded corpus reference list
    vocab = build_vocab(tokens, min_freq, subsampling_threshold)
    int_tokens = get_tokens_as_indices(tokens, vocab)
    logger.info(
        f"Built vocabulary of {len(vocab)} words with frequency threshold of {min_freq}."
    )

    # finally, instantiate dataset
    dataset = CBOWDataset(int_tokens, context_size=context_size)
    return dataset, vocab


def get_dl_from_ds(
    dataset: Union[CBOWDataset, Subset[CBOWDataset]],
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Returns a DataLoader for the given dataset, with specified batch size and options.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # good for GPU use
        collate_fn=cbow_collate,  # use custom collate function
    )


# default collate_fn would work but we include this for clarity / documentation
def cbow_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stacks tuples produced by CBOWDataset into 2D / 1D tensors for the trainer.
    """
    # counter-intuitively, we use zip to 'unzip' (or transpose) the batch into two lists of length B
    ctx, tgt = zip(*batch)  # each ctx is [2w], tgt is a scalar
    return torch.stack(ctx), torch.stack(tgt)


def get_text8(
    cache_dir: Union[str, Path] = DATA_DIR,
    text8_filename: str = TEXT8_FILENAME,
) -> Path:
    """
    Ensure the Matt Mahoney 'text8' corpus is present locally, downloading it
    once and caching it for subsequent runs.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{text8_filename}.zip"
    txt_path = cache_dir / text8_filename

    # txt already present? then we are done
    if txt_path.exists():
        logger.info(f"text8 file found - using cached data at {txt_path}")
        return txt_path

    # ensure full text8 file is present, and otherwise download and extract as necessary
    if not zip_path.exists():
        logger.info(f"Downloading text8 corpus to {zip_path}...")
        urllib.request.urlretrieve(TEXT8_ZIP_URL, zip_path)
    logger.info(f"Extracting {zip_path} to {txt_path}...")
    with ZipFile(zip_path, "r") as zf:
        zf.extract(text8_filename, cache_dir)
    return txt_path


# expose text8 logic to run directly in case desirable
if __name__ == "__main__":
    path = get_text8()
    print(f"text8 data ready at: {path}")
