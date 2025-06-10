import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from wandb.sdk.wandb_run import Run


logger = logging.getLogger(__name__)


class Word2VecTrainer:
    # TODO: flesh out types
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        batch_size: int,
        train_dl,
        train_steps,
        val_dl,
        val_steps,
        checkpoint_frequency,
        learning_rate,
        device,
        model_dir,
        model_name: str,
        wandb_runner: Run,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dl = train_dl
        self.train_steps = train_steps
        self.val_dl = val_dl
        self.val_steps = val_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.learning_rate = learning_rate
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.wandb_runner = wandb_runner

        # some of the below could be exposed as args in order to generalise the class further
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # decay learning rate linearly per epoch (i.e. chase gradient fast early on, and slow down over time)
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

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        # TODO: make sure this logic (and val) reflect final dataloader setup
        for i, batch_data in enumerate(self.train_dl, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)

            # run back propagation and update weights
            loss.backward()
            self.lr_scheduler.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dl, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

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
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
