import json
import os
import logging
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
        print model evaluation metrics, these are human readable, produced by comparing a few random words and a list of prechosen words
        which are each dot producted with the entire dataset to find their nearest neighbours
        """
        self.model.eval()
        if self.vocab is None:
            logger.warning(
                "No vocab provided, cannot evaluate model. Please provide a vocab."
            )
            return
        # get a few random words from the vocab
        random_words = np.random.choice(list(self.vocab.keys()), size=5, replace=False)
        words_to_evaluate = list(set(random_words) | set(CHOICE_WORDS_TO_EVALUATE))
        logger.info("Evaluating model on words: {}".format(words_to_evaluate))
        # get the indices of these words in the vocab
        indices = [self.vocab[word] for word in words_to_evaluate if word in self.vocab]
        if not indices:
            logger.warning(
                "No valid words found in vocab for evaluation. Please ensure the vocab is complete."
            )
            return
        # dot product these indices with the model's embedding weights
        with torch.no_grad():
            embeddings = self.model.embedding(torch.tensor(indices).to(self.device))
            # calculate cosine similarity between the embeddings
            cosine_similarities = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            )
            # get the top 5 nearest neighbours for each word
            for i, word in enumerate(words_to_evaluate):
                if word in self.vocab:
                    top_indices = torch.topk(cosine_similarities[i], k=6).indices[1:]
                    nearest_words = [words_to_evaluate[idx] for idx in top_indices]
                    logger.info(
                        "Nearest neighbours for '{}': {}".format(word, nearest_words)
                    )

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
