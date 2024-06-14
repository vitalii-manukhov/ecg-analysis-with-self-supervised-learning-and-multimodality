"""
-*- coding = utf-8 -*-
@File : train.py
@Software : vscode
"""
import logging
import os.path
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.utils import keep_top_files, epoch_log


class StepRunner:
    """
    Description:
        Class for defining a StepRunner object that handles model training steps.

    Args:
        model: The model used for training.
        loss_fn: The loss function for the training.
        stage (str): The stage of training (default is "train").
        metrics_dict (dict): Dictionary of metrics to track during training.
        optimizer: The optimizer used for training.
    """
    def __init__(self, model, loss_fn,
                 stage="train", metrics_dict=None,
                 optimizer=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = model, loss_fn, metrics_dict, stage
        self.optimizer = optimizer

    def step(self, ecg_data, text_data) -> float:
        """
        Description:
            This function performs a training step by forwarding the `ecg_data` and `text_data`
            through the model, computing the loss, and
            optionally updating the model's parameters using the optimizer.

        Args:
            ecg_data: The ECG data used for training.
            text_data: The text data used for training.

        Returns:
            float: The loss value as a float.
        """
        # loss
        ecg_representation, text_representation = self.net(ecg_data, text_data)
        loss = self.loss_fn(ecg_representation, text_representation)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        # step_metrics = {self.stage + "_" + name: metric_fn(preds, labels).item()
        #                 for name, metric_fn in self.metrics_dict.items()}
        # return loss.item(), step_metrics
        return loss.item()

    def train_step(self, ecg_data, text_data) -> float:
        """
        Description:
            This function performs a training step by forwarding the `ecg_data` and `text_data`

        Args:
            ecg_data (torch.Tensor): The ECG data used for training.
            text_data (torch.Tensor): The text data used for training.

        Returns:
            float: The loss value as a float.
        """
        self.net.train()  # train mode, dropout work

        return self.step(ecg_data, text_data)

    @torch.no_grad()
    def eval_step(self, ecg_data, text_data) -> float:
        """
        Description:
            This function performs an evaluation step by forwarding the `ecg_data` and `text_data`

        Args:
            ecg_data (torch.Tensor): The ECG data used for training.
            text_data (torch.Tensor): The text data used for training.

        Returns:
            float: The loss value as a float.
        """

        self.net.eval()  # eval mode, dropout not work
        return self.step(ecg_data, text_data)

    def __call__(self, ecg_data, text_data):
        if self.stage == "train":
            return self.train_step(ecg_data, text_data)
        else:
            return self.eval_step(ecg_data, text_data)


class EpochRunner:
    """
    Description:
        Class for defining an EpochRunner object that handles model training epochs.

    Args:
        steprunner: The StepRunner object used for training.
    """
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def __call__(self, dataloader):
        """
        Description:
            A method that handles the model training epochs based on the given dataloader.

        Args:
            dataloader: The dataloader used for iterating over the dataset.

        Returns:
            The dictionary containing the loss value for the epoch.
        """
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        for i, batch in loop:
            # loss, step_metrics = self.steprunner(*batch)
            # step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            loss = self.steprunner(*batch)
            step_log = dict({self.stage + "_loss": loss})
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                # epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                #                  for name, metric_fn in self.steprunner.metrics_dict.items()}
                # epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)

                epoch_log = dict({self.stage + "_loss": epoch_loss})
                loop.set_postfix(**epoch_log)

                # for name, metric_fn in self.steprunner.metrics_dict.items():
                #     metric_fn.reset()

        return epoch_log


def ssl_train(model, optimizer, loss_fn, metrics_dict,
              train_dataloader, val_dataloader=None,
              epochs=10, save_path='./checkpoint/',
              patience=5, monitor="val_loss", mode="min"):
    """
    Description:
        Function for training a model using Self-Supervised Learning (SSL).

    Args:
        model: The model to be trained.
        optimizer: The optimizer for training the model.
        loss_fn: The loss function used for training.
        metrics_dict: A dictionary containing metrics to evaluate the training.
        train_dataloader: The dataloader for training data.
        val_dataloader: The dataloader for validation data (default is None).
        epochs: The number of epochs for training (default is 10).
        save_path: The path to save the checkpoints (default is './checkpoint/').
        patience: The patience for early stopping (default is 5).
        monitor: The metric to monitor for saving the best model (default is "val_loss").
        mode: The mode to monitor the metric (default is "min").

    Returns:
        DataFrame: A pandas DataFrame containing the training history.
    """
    train_history = {}

    logging.warning("=" * 25 + "Start SSL Training" + "=" * 25)
    for epoch in range(1, epochs + 1):

        epoch_log("Epoch {0} / {1}".format(epoch, epochs))

        # 1 train -------------------------------------------------
        train_step_runner = StepRunner(model=model, stage="train",
                                       loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_dataloader)

        for name, metric in train_metrics.items():
            train_history[name] = train_history.get(name, []) + [metric]

        # 2 validate -------------------------------------------------
        if val_dataloader:
            val_step_runner = StepRunner(model=model, stage="val",
                                         loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_dataloader)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                train_history[name] = train_history.get(name, []) + [metric]

        logging.info("Train Loss: {0:.6f}".format(train_history["train_loss"][epoch - 1]))
        logging.info("Val Loss: {0:.6f}".format(train_history["val_loss"][epoch - 1]))

        # 3 save -------------------------------------------------
        arr_scores = train_history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ckpt_name = "Epoch_{0}_loss_{1:.6f}.pt".format(train_history['epoch'][best_score_idx],
                                                           train_history['val_loss'][best_score_idx])
            ckpt_path = os.path.join(save_path, ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': train_history[monitor][best_score_idx],
            }, ckpt_path)
            keep_top_files(save_path, 5)

            logging.info("<<<<<< reach best {0} : {1} >>>>>>".format(monitor, arr_scores[best_score_idx]))
            logging.info("<<<<<< save the checkpoint {0} >>>>>>".format(ckpt_name))
        if len(arr_scores) - best_score_idx < patience:
            logging.info("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience))
            break

    return pd.DataFrame(train_history)
