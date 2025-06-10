import logging
import os

import torch
import wandb

from model import CBOWModel
from trainer import Word2VecTrainer


logger = logging.getLogger(__name__)


# constants (excluding hyperparameters)
VOCAB_SIZE = 30000  # TMP
RNG_SEED = 42
CHECKPOINT_FREQUENCY = 5  # how often to save model weights during training
MODEL_NAME = "word2vec_{}_{}"
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
WANDB_TEAM = "mlx8-dropout-disco"
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

    logger.info("Initialising model...")
    # TODO: vocab size to be determined from when dataset is integrated
    model = CBOWModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=run.config["embedding_dimensions"],
        embed_max_norm=run.config["embedding_max_norm"],
    )

    # TODO: load dataset, create dataloaders to be submitted to trainer
    logger.info("Initialising trainer...")
    trainer = Word2VecTrainer(
        model=model,
        epochs=run.config["epochs"],
        batch_size=run.config["batch_size"],
        train_dl=None,  # to be defined
        train_steps=1000,  # to be defined
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
