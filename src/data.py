import logging
import os
from pathlib import Path
from typing import Union
import urllib.request
from zipfile import ZipFile

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

from tokeniser import (
    build_vocab,
    tokenise,
    get_tokens_as_indices,
    DOC_END_TOKEN,
    DOC_START_TOKEN,
    PAD_TOKEN,
)


logger = logging.getLogger(__name__)


DATA_DIR = os.getenv("DATA_DIR", ".data")
TRAIN_PROCESSED_FILENAME = "hn_posts_train_processed.parquet"
TEXT8_ZIP_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_FILENAME = "text8"
TEXT8_EXPECTED_LENGTH = 100_000_000  # text8 is expected to be 100 million chars

# manually set env vars also used in data ingest (see .env.example / db-utils.py)
TITLES_FILE = "hn_posts_titles.parquet"
MINIMAL_FETCH_ONLY_TITLES = True

# include a debug mode - in which case we only load a fraction of text8 (for very short runs)
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"


# custom map-style dataset for CBOW
class CBOWDataset(Dataset):
    """
    Map-style dataset so DataLoader can shuffle & use multiple workers.
    Keeps *only* the integer-encoded corpus in RAM.
    """

    # note that references to 'token' here refer to the integer index in the vocab, not a string
    def __init__(
        self,
        vocab: dict[str, int],
        context_size: int = 3,
        text8_token_idx: list[int] = [],
        hn_titles_token_idx: list[list[int]] = [],
    ) -> None:
        assert context_size > 0
        self.context_size = context_size

        self.start_token = vocab[DOC_START_TOKEN]
        self.end_token = vocab[DOC_END_TOKEN]
        self.pad_token = vocab[PAD_TOKEN]

        self.text8_token_idx = text8_token_idx
        self.text8_len = len(text8_token_idx)
        # we require that some text8 corpus is provided (unlike HN titles, which are optional)
        assert self.text8_len > 0

        # prepapre some global vars to track progress through HN titles
        self.hn_titles_token_idx = hn_titles_token_idx
        self.hn_titles_lengths: list[int] = []
        self.cumulative_hn_titles_start_indices: list[int] = []
        current_offset = 0
        for title_tokens in self.hn_titles_token_idx:
            length = len(title_tokens)
            if length > 0:  # only consider non-empty titles for indexing
                self.hn_titles_lengths.append(length)
                self.cumulative_hn_titles_start_indices.append(current_offset)
                current_offset += length
        self.hn_titles_len = current_offset  # sum of lengths of non-empty HN titles

    def __len__(self) -> int:
        return self.text8_len + self.hn_titles_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"Getting item {idx} from CBOWDataset...")

        current_doc_tokens: list[int] = []
        target_idx_in_doc: int = 0
        # start by dispensing context windows/targets from text8 corpus, then move on to HN titles (if any provided)
        if idx < self.text8_len:
            logger.debug(
                f"idx {idx} is within text8_len {self.text8_len}. Accessing text8 corpus..."
            )
            current_doc_tokens = self.text8_token_idx
            target_idx_in_doc = idx
            logger.debug(
                f"text8: current_doc_tokens length: {len(current_doc_tokens)}, target_idx_in_doc: {target_idx_in_doc}"
            )
        elif self.hn_titles_len > 0:
            logger.debug(
                f"idx {idx} is beyond text8_len {self.text8_len}. Accessing HN titles..."
            )
            hn_corpus_idx = (
                idx - self.text8_len
            )  # index relative to start of HN titles corpus
            logger.debug(
                f"Calculated hn_corpus_idx: {hn_corpus_idx} (idx {idx} - text8_len {self.text8_len})"
            )
            doc_found = False
            logger.debug(
                f"Iterating through {len(self.hn_titles_token_idx)} original HN titles to find document for hn_corpus_idx {hn_corpus_idx}."
            )
            processed_title_idx = 0
            for i, original_title_tokens in enumerate(self.hn_titles_token_idx):
                logger.debug(
                    f"  HN loop: original title index i={i}, len(original_title_tokens)={len(original_title_tokens)}"
                )
                if len(original_title_tokens) == 0:  # skip empty titles (as before)
                    logger.debug(
                        f"  HN loop: skipping empty original title at index i={i}"
                    )
                    continue

                if processed_title_idx >= len(self.hn_titles_lengths):
                    logger.warning(
                        f"  HN loop: processed_title_idx {processed_title_idx} is out of bounds for self.hn_titles_lengths (len {len(self.hn_titles_lengths)}) - breaking!"
                    )
                    break

                # use lengths and start indices of processed (non-empty) titles
                title_start_offset = self.cumulative_hn_titles_start_indices[
                    processed_title_idx
                ]
                if (
                    hn_corpus_idx >= title_start_offset
                    and hn_corpus_idx
                    < title_start_offset + self.hn_titles_lengths[processed_title_idx]
                ):
                    current_doc_tokens = (
                        original_title_tokens  # use the actual token list
                    )
                    target_idx_in_doc = hn_corpus_idx - title_start_offset
                    doc_found = True
                    logger.debug(
                        f"  HN loop: Document found for hn_corpus_idx {hn_corpus_idx}. Original title index i={i}, processed_title_idx={processed_title_idx}."
                    )
                    logger.debug(
                        f"  HN loop: current_doc_tokens length: {len(current_doc_tokens)}, target_idx_in_doc: {target_idx_in_doc}"
                    )
                    break
                processed_title_idx += 1
            if not doc_found:
                raise IndexError(
                    f"Index {idx} (hn_corpus_idx {hn_corpus_idx}) out of bounds for HN titles."
                )
        else:
            raise IndexError(
                f"Index {idx} out of bounds for CBOWDataset overall. "
                "Ensure idx is less than the total number of tokens in text8 + HN titles."
            )

        tgt_token = current_doc_tokens[target_idx_in_doc]
        ctx_tokens = []
        logger.debug(
            f"Generating left context for target_idx_in_doc {target_idx_in_doc}, context_size {self.context_size}"
        )
        # left context (relative to target)
        for i in range(self.context_size):
            pos = target_idx_in_doc - self.context_size + i
            if pos == -1:
                ctx_tokens.append(self.start_token)
            elif pos < -1:
                ctx_tokens.append(self.pad_token)
            else:
                ctx_tokens.append(current_doc_tokens[pos])

        # right context
        logger.debug(
            f"Generating right context for target_idx_in_doc {target_idx_in_doc}, context_size {self.context_size}, len(current_doc_tokens) {len(current_doc_tokens)}"
        )
        for i in range(self.context_size):
            pos = target_idx_in_doc + 1 + i
            if pos == len(current_doc_tokens):
                ctx_tokens.append(self.end_token)
            if pos > len(current_doc_tokens):
                ctx_tokens.append(self.pad_token)
            else:
                ctx_tokens.append(current_doc_tokens[pos])

        logger.debug(
            f"Final ctx_tokens: {ctx_tokens} (len {len(ctx_tokens)}), tgt_token: {tgt_token}"
        )
        return (
            torch.tensor(ctx_tokens, dtype=torch.long),  # shape is [2 * context]
            torch.tensor(tgt_token, dtype=torch.long),  # 0D scalar tensor
        )


def build_cbow_dataset(
    context_size: int = 3,
    min_freq: int = 5,
    subsampling_threshold: float = 1e-5,
    include_hn_titles: bool = True,
) -> tuple[CBOWDataset, dict]:
    # TODO: also get comments from HN, to be appended to the corpus ??
    hn_titles: list[str] = []
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
        hn_titles = hn_posts["title"].tolist()
    else:
        logger.info("Skipping HN titles, using only text8 corpus")

    # ensure text8 corpus is downloaded to data dir
    text8_path = get_text8(cache_dir=DATA_DIR)

    # build the big string for vocab and tokenisation
    with open(text8_path, "r", encoding="utf-8") as f:
        if DEBUG_MODE:
            logger.debug(
                f"Debug mode: only using first {TEXT8_EXPECTED_LENGTH // 100} chars of text8"
            )
            corpus_txt = f"{f.read(TEXT8_EXPECTED_LENGTH // 100)} {' '.join(hn_titles)}"
        else:
            corpus_txt = f"{f.read()} {' '.join(hn_titles)}"

    # tokenise full corpus and build vocab
    tokens = tokenise(corpus_txt)
    logger.info(f"Produced tokenised corpus of {len(tokens)} tokens.")
    vocab = build_vocab(tokens, min_freq, subsampling_threshold)
    logger.info(
        f"Built vocabulary of {len(vocab)} words (tokens) with frequency threshold of {min_freq}."
    )

    # separately tokenise full text8 and each HN title, and convert to indices
    text8_token_idx: list[int] = get_tokens_as_indices(tokenise(corpus_txt), vocab)
    hn_titles_token_idx: list[list[int]] = [
        get_tokens_as_indices(tokenise(title), vocab) for title in hn_titles
    ]

    # finally, instantiate dataset
    logger.info("Building CBOW dataset...")
    dataset = CBOWDataset(
        vocab=vocab,
        context_size=context_size,
        text8_token_idx=text8_token_idx,
        hn_titles_token_idx=hn_titles_token_idx,
    )
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
