import logging
import os

import torch
from torch.utils.data import random_split
import wandb

from model import CBOWModel
from trainer import Word2VecTrainer
from utils import get_device
from data import (
    build_cbow_dataset,
    get_dl_from_ds,
)


logger = logging.getLogger(__name__)


# constants (excluding hyperparameters)
RNG_SEED = 42
CHECKPOINT_FREQUENCY = 5  # how often to save model weights during training
MODEL_NAME = "word2vec_{}_{}_{}"  # format: architecture, dataset, wandb run
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
WANDB_TEAM = "freemvmt-london"
WANDB_PROJECT = "word2vec"

# hot swap essential config from command line
EPOCHS = int(os.environ.get("EPOCHS", 30))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2048))


# hyperparameters should be encoded and varied from this config constant
CONFIG = {
    "architecture": "cbow",
    "dataset": "text8-hn",
    "context_size": 2,
    "freq_threshold": 5,
    "subsampling_threshold": None,  # set to None to disable freq. subsampling
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,  # batch size for training, validation, and test
    "learning_rate": 1e-3,  # initial lr for Adam (may want to decrease if not using scheduler)
    "use_scheduler": True,  # whether to step lr down linearly over epochs
    "embedding_dimensions": 300,
    "embedding_max_norm": 1.0,
    "train_proportion": 0.9,  # proportion of dataset to use for training
    "val_proportion": 0.05,  # proportion of dataset to use for validation
    "include_hn_titles": True,  # whether to include HN titles in the dataset
}


def train() -> None:
    # set up logger (`export DEBUG_MODE=1` in executing shell to see debug logs)
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

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
    logger.info(f"WandB run {run.id} initialised with config: {run.config}")

    if not (run.config.train_proportion + run.config.val_proportion) < 1.0:
        raise ValueError(
            "Training and validation proportions must sum to less than 1. "
            f"Got: train={run.config.train_proportion}, val={run.config.val_proportion}"
        )

    logger.info("Building dataset...")
    ds, vocab = build_cbow_dataset(
        context_size=run.config["context_size"],
        min_freq=run.config["freq_threshold"],
        subsampling_threshold=run.config["subsampling_threshold"],
        include_hn_titles=run.config["include_hn_titles"],
    )

    # split dataset into train, validation, and test sets
    ds_len = len(ds)
    train_size = int(run.config.train_proportion * ds_len)
    val_size = int(run.config.val_proportion * ds_len)
    test_size = ds_len - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RNG_SEED),
    )
    logger.info(
        f"Dataset split into: train ({len(train_ds)}), validation ({len(val_ds)}), test ({len(test_ds)})"
    )

    train_dl = get_dl_from_ds(train_ds, batch_size=run.config["batch_size"])
    val_dl = get_dl_from_ds(val_ds, batch_size=run.config["batch_size"])
    test_dl = get_dl_from_ds(test_ds, batch_size=run.config["batch_size"])

    logger.info("Initialising model...")
    model = CBOWModel(
        vocab_size=len(vocab),
        embed_dim=run.config["embedding_dimensions"],
        embed_max_norm=run.config["embedding_max_norm"],
    )

    logger.info("Initialising trainer...")
    trainer = Word2VecTrainer(
        model=model,
        epochs=run.config["epochs"],
        batch_size=run.config["batch_size"],
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        learning_rate=run.config["learning_rate"],
        use_scheduler=run.config["use_scheduler"],
        device=get_device(),
        model_dir="weights",
        model_name=MODEL_NAME.format(
            run.config["architecture"],
            run.config["dataset"],
            run.name,
        ),
        wandb_runner=run,
        vocab=vocab,
    )

    logger.info("Starting training...")
    trainer.train()

    # finish run and upload any remaining data
    run.finish()


if __name__ == "__main__":
    train()
