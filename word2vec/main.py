import logging
import os

import torch
import wandb
import pandas as pd

from model import CBOWModel
from trainer import Word2VecTrainer
from tokeniser import build_vocab, tokenise
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


# constants (excluding hyperparameters)
VOCAB_SIZE = 30000  # TMP
RNG_SEED = 42
CHECKPOINT_FREQUENCY = 5  # how often to save model weights during training
MODEL_NAME = "word2vec_{}_{}"
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
WANDB_TEAM = "freemvmt-london"
WANDB_PROJECT = "word2vec"

# no. of words to consider on each side of target (OG paper recommends 4)
# this could be exposed as a hyperparameter (ie. moved to config)
CBOW_CONTEXT_SIZE = 3


# hyperparamters should be encoded and varied from this config constant
CONFIG = {
    "architecture": "cbow",
    "dataset": "text8",
    "epochs": 10,
    "batch_size": 1024,
    "learning_rate": 1e-2,
    "embedding_dimensions": 100,
    "embedding_max_norm": 1.0,
    "tokeniser_freq_threshold": 5,
}


def train() -> None:
    # set up logger (`export DEBUG_MODE=1` in executing shell to see debug logs)
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set seed for reproducibility of pseudo-random number generation
    torch.manual_seed(RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RNG_SEED)

    # set up wandb to track our experiments (ie. parameter sweeping)
    run = wandb.init(
        entity=WANDB_TEAM,
        project=WANDB_PROJECT,
        config=CONFIG,
    )
    logger.debug(f"WandB run {run.id} initialised with config: {run.config}")

    # TODO: load dataset, create dataloaders to be submitted to trainer
    parquet_path = "postgresql/.data/hn_posts_train_processed.parquet"
    df_hn = pd.read_parquet(parquet_path)
    print(f"Loaded data from {parquet_path}")

    # Show DataFrame info
    print("\nDataFrame info:")
    df_hn.info()
    titles = df_hn["title"].tolist()
    print(f"First 10 titles: {titles[:10]}")
    all_titles = " ".join(titles)

    # load text8 string
    with open("text8", "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} lines from {'text8'}")

    bag = all_titles + " " + " ".join(lines)

    class CBOWDataset(Dataset):
        def __init__(self, text, context_size: int = CBOW_CONTEXT_SIZE):
            self.context_size = context_size
            self.tokens = tokenise(text)
            self.vocab = build_vocab(
                self.tokens, frequency_threshold=run.config["tokeniser_freq_threshold"]
            )
            # Tokenise text into indices, skipping unknowns
            self.length = len(self.tokens)
            print(f"Dataset length: {self.length}")
            print(f"Vocabulary size: {len(self.vocab)}")

        def __len__(self):
            # Only positions where full context is available
            return self.length - 2 * self.context_size

        def __getitem__(self, idx):
            idx += self.context_size
            context = (
                self.tokens[idx - self.context_size : idx]
                + self.tokens[idx + 1 : idx + 1 + self.context_size]
            )
            target = self.tokens[idx]

            # Skip samples where any context token or the target isn't in vocabulary
            context_ids = [self.vocab.get(token, -1) for token in context]
            target_id = self.vocab.get(target, -1)

            # Check if all tokens are in vocabulary (-1 means token not found)
            if -1 in context_ids or target_id == -1:
                # If any tokens are missing, try another index
                # This is a simple approach - you might want a more sophisticated strategy
                return self[(idx + 1) % (len(self) - 1)]
            return (torch.tensor(context_ids, dtype=torch.long), target_id)

    # Build dataset and dataloader
    dataset = CBOWDataset(text=bag)

    train_dl = DataLoader(
        dataset,
        batch_size=run.config["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    logger.info("Initialising model...")
    # TODO: vocab size to be determined from when dataset is integrated
    model = CBOWModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=run.config["embedding_dimensions"],
        embed_max_norm=run.config["embedding_max_norm"],
    )

    logger.info("Initialising trainer...")
    trainer = Word2VecTrainer(
        model=model,
        epochs=run.config["epochs"],
        batch_size=run.config["batch_size"],
        train_dl=train_dl,
        train_steps=1000,  # what is this ??
        val_dl=None,  # to be defined
        val_steps=100,  # to be defined
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        learning_rate=run.config["learning_rate"],
        device=device,
        model_dir="weights",
        model_name=MODEL_NAME.format(
            run.config["architecture"],
            run.config["dataset"],
        ),
        wandb_runner=run,
    )

    logger.info("Starting training...")
    trainer.train()

    # finish run and upload any remaining data
    run.finish()


if __name__ == "__main__":
    train()
