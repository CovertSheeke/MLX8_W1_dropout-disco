import logging
import os

import torch
from torch.utils.data import DataLoader
import wandb

from model import CBOWModel
from trainer import Word2VecTrainer
from utils import get_device
from data import build_cbow_dataset, cbow_collate


logger = logging.getLogger(__name__)


# constants (excluding hyperparameters)
RNG_SEED = 42
CHECKPOINT_FREQUENCY = 5  # how often to save model weights during training
MODEL_NAME = "word2vec_{}_{}"
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
WANDB_TEAM = "freemvmt-london"
WANDB_PROJECT = "word2vec"


# hyperparameters should be encoded and varied from this config constant
CONFIG = {
    "architecture": "cbow",
    "dataset": "text8",
    "context_size": 3,
    "freq_threshold": 5,
    "epochs": 100,
    "batch_size": 1024,
    "learning_rate": 1e-2,  # initial lr for Adam (may want to decrease if not using scheduler)
    "use_scheduler": True,  # whether to step lr down linearly over epochs
    "embedding_dimensions": 100,
    "embedding_max_norm": 1.0,
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

    train_ds, vocab = build_cbow_dataset(
        context_size=CONFIG["context_size"],
        min_freq=CONFIG["freq_threshold"],
        include_hn_titles=False,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # good for GPU use
        collate_fn=cbow_collate,
    )

    # set up wandb to track our experiments (ie. parameter sweeping)
    run = wandb.init(
        entity=WANDB_TEAM,
        project=WANDB_PROJECT,
        config=CONFIG,
    )
    logger.debug(f"WandB run {run.id} initialised with config: {run.config}")

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
        train_steps=1000,  # what is this ??
        # TODO: split validation out from training dataset and pass through here
        val_dl=None,  # to be defined
        val_steps=100,  # to be defined
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        learning_rate=run.config["learning_rate"],
        use_scheduler=run.config["use_scheduler"],
        device=get_device(),
        model_dir="weights",
        model_name=MODEL_NAME.format(
            run.config["architecture"],
            run.config["dataset"],
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
