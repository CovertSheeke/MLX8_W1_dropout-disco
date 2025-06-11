import json
import os
import logging
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from wandb.sdk.wandb_run import Run


logger = logging.getLogger(__name__)

# list off a few chosen words for eval
CHOICE_WORDS_TO_EVALUATE = [
    "bottle",
    "hot",
    "ferocious",
    "king",
    "jump",
    "debate",
    "politics",
    "science",
    "computer",
    "python",
]


class Word2VecTrainer:
    # TODO: flesh out types
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        batch_size: int,
        train_dl,
        val_dl,
        test_dl,
        checkpoint_frequency,
        learning_rate,
        use_scheduler: bool,
        device,
        model_dir,
        model_name: str,
        wandb_runner: Run,
        vocab: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.checkpoint_frequency = checkpoint_frequency
        self.learning_rate = learning_rate
        self.use_scheduler = (use_scheduler,)
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.wandb_runner = wandb_runner
        self.vocab = vocab

        # ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # some of the below could be exposed as args in order to generalise the class further
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # decay learning rate linearly per epoch (i.e. chase gradient fast early on, and slow down over time)
        if self.use_scheduler:
            self.lr_scheduler = LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda current_epoch: (epochs - current_epoch) / epochs,
            )
        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self._train_epoch()
            self._validate_epoch()
            self._eval_model()  # evaluate model after validation

            # log to console and wandb
            logger.debug(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )
            self.wandb_runner.log(
                {
                    "epoch": epoch,
                    "train_loss": self.loss["train"][-1],
                    "val_loss": self.loss["val"][-1],
                }
            )

            # step the learning rate down after each epoch (if using scheduler)
            if self.use_scheduler:
                self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

        # TODO: implement final test phase with test_dl

        # save final model and loss history
        self.save_model()
        self.save_loss()

    # TODO: could implement negative sampling to speed up training runs?
    def _train_epoch(self):
        self.model.train()
        running_loss = []

        # TODO: make sure this logic (and val) reflect final dataloader setup
        for i, batch_data in tqdm(enumerate(self.train_dl, 1)):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)

            # run back propagation and update weights
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(self.val_dl, 1)):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _eval_model(self):
        """
        Log nearest-neighbour words for a few random+preset probes.
        """
        self.model.eval()
        if self.vocab is None:
            logger.warning("No vocab provided; skipping evaluation.")
            return

        # assemble probe word list
        random_words = random.sample(list(self.vocab), k=min(5, len(self.vocab)))
        probe_words = list(set(random_words) | set(CHOICE_WORDS_TO_EVALUATE))

        # keep only words that survived any pruning in build_vocab
        valid_words = [w for w in probe_words if w in self.vocab]
        if not valid_words:
            logger.warning("No valid probe words found in vocab; skipping evaluation.")
            return

        word_indices = torch.tensor(
            [self.vocab[w] for w in valid_words], device=self.device
        )

        # compute pairwise cosine similarities
        with torch.no_grad():
            all_embeds = torch.nn.functional.normalize(
                self.model.embeddings.weight, p=2, dim=1
            )  # (V, D)
            probe_embs = all_embeds[word_indices]  # (N, D)
            sim = probe_embs @ all_embeds.T  # (N, V)

        # extract neighbours
        k = min(5, len(valid_words) - 1)  # up to 5 neighbours (excludes self)
        for i, word in enumerate(valid_words):
            if k == 0:
                neighbours = []
            else:
                # argsort descending, skip self
                topk_idx = sim[i].topk(k + 1).indices.tolist()
                neighbours = [valid_words[j] for j in topk_idx if j != i][:k]
            logger.info(f"Nearest neighbours for '{word}': {neighbours}")

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        if epoch % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, self.model_name + ".pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, self.model_name + "_loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
