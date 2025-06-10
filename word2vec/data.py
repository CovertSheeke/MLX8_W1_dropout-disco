import logging
import os
from pathlib import Path
from typing import Union
import urllib.request
import zipfile

import torch
import pandas as pd
from torch.utils.data import Dataset

from tokeniser import build_vocab, tokenise, get_tokens_as_indices


logger = logging.getLogger(__name__)


DATA_DIR = os.getenv("DATA_DIR", ".data")
TRAIN_PROCESSED_FILENAME = "hn_posts_train_processed.parquet"
TEXT8_ZIP_URL = "http://mattmahoney.net/dc/text8.zip"
FILE_NAME = "text8"
ZIP_NAME = FILE_NAME + ".zip"


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


# TODO: split out test dataset for full corpus (ie. from both HN posts and text8)
def build_cbow_dataset(
    context_size: int = 3,
    min_freq: int = 5,
) -> tuple[CBOWDataset, dict]:
    parquet_path = os.path.join(DATA_DIR, TRAIN_PROCESSED_FILENAME)
    logger.debug(f"Loading data from {parquet_path}...")
    hn_posts = pd.read_parquet(parquet_path)
    # show some basic info about the data we pulled from parquet
    logger.info(f"Found {len(hn_posts)} records at {parquet_path}: {hn_posts.info()}")
    # pull out all titles and concatenate into single string, separated by whitespace
    hn_titles = " ".join(hn_posts["title"].tolist())

    # ensure text8 corpus is downloaded to data dir
    text8_filepath = get_text8(cache_dir=DATA_DIR)

    # build the big string
    with open(text8_filepath, "r", encoding="utf-8") as f:
        corpus_txt = f.read() + " " + hn_titles

    tokens = tokenise(corpus_txt)
    logger.info(f"Produced tokenised corpus of {len(tokens)} tokens.")

    # build vocab and integer-encoded corpus reference list
    vocab = build_vocab(tokens, min_freq=min_freq)
    int_tokens = get_tokens_as_indices(tokens, vocab)
    logger.info(
        f"Built vocabulary of {len(vocab)} words with frequency threshold of {min_freq}."
    )

    # finally, instantiate dataset
    dataset = CBOWDataset(int_tokens, context_size=context_size)
    return dataset, vocab


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


def get_text8(cache_dir: Union[str, Path] = DATA_DIR) -> Path:
    """
    Ensure the Matt Mahoney 'text8' corpus is present locally, downloading it
    once and caching it for subsequent runs.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / ZIP_NAME
    txt_path = cache_dir / FILE_NAME  # the file inside the zip

    # already unzipped?
    if txt_path.exists():
        return txt_path

    # if we don't have the zip, need to download
    if not zip_path.exists():
        logger.info(f"Downloading text8 corpus to {zip_path}...")
        urllib.request.urlretrieve(TEXT8_ZIP_URL, zip_path)

    # finally, unzip the file
    logger.info(f"Extracting {zip_path} to {cache_dir}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(FILE_NAME, cache_dir)
    return txt_path


# expose text8 logic to run directly in case desirable
if __name__ == "__main__":
    path = get_text8()
    print(f"text8 data ready at: {path}")
