import gc
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Any, Iterable
from unittest.mock import MagicMock
from pathlib import Path

import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neural_network import MLP
from matchups import get_matchups
from pipeline import bucketization_of_ranges, bucketization_of_values, unbucketization
from utils import device

train_ranges = torch.load('data/train_ranges.pt').reshape(-1, 2*1326)  # tensor of shape (TRAIN_DATASET_SIZE, 2*1326)
train_boards = torch.load('data/train_boards.pt')  # tensor of shape (TRAIN_DATASET_SIZE, 4)"
train_pots = torch.load('data/train_pots.pt')  # tensor of shape (TRAIN_DATASET_SIZE, 1)
train_stacks = torch.ones_like(train_pots) * 20000  # tensor of shape (TRAIN_DATASET_SIZE, 1)
train_EVs = torch.load('data/train_EVs.pt').reshape(-1, 2*1326)  # tensor of shape (TRAIN_DATASET_SIZE, 2*1326)

valid_ranges = torch.load('data/valid_ranges.pt').reshape(-1, 2*1326)  # tensor of shape (VALID_DATASET_SIZE, 2*1326)
valid_boards = torch.load('data/valid_boards.pt')  # tensor of  shape (VALID_DATASET_SIZE, 4)
valid_pots = torch.load('data/valid_pots.pt')  # tensor of shape (VALID_DATASET_SIZE, 1)
valid_stacks = torch.ones_like(valid_pots) * 20000  # tensor of shape (VALID_DATASET_SIZE, 1)
valid_EVs = torch.load('data/valid_EVs.pt').reshape(-1, 2*1326)  # tensor of shape (VALID_DATASET_SIZE, 2*1326)


@dataclass
class TrainConfig:
    hidden_sizes: Iterable[int]
    lr: float
    dropout: float
    batch_size: int
    epochs: int
    is_ev_based: bool
    unbucketed_loss_freq: int = 10
    neptune_run: Any = None
    n_players: int = 2
    n_buckets: int = 1000  # not used in the code as data are random
    dataset_id: int = 1  # not used in the code as data are random


class TrainSchedule:
    def __init__(self, config: TrainConfig):
        self.config = config
        neptune_run = config.neptune_run
        self.start_time = datetime.now()
        self.run_id = None
        if neptune_run is None:
            neptune_run = defaultdict(MagicMock)
            self.run_id = str(self.start_time)
        else:
            self.run_id = neptune_run["sys/id"].fetch()
        self.neptune_run = neptune_run

        self.bucketed_loss_fun = nn.SmoothL1Loss(reduction="none")
        self.MSELoss = nn.MSELoss(reduction="none")

        self.train_len, self.val_len = len(self._get_trainset()), len(self._get_validset())
        self.n_buckets = 1000  # attached data are generated for this number, however it is a parameter
        self.models_dir = Path("models")
        self.device = device
        self.model = MLP(
            input_size=2*self.n_buckets+1,
            target_size=2*self.n_buckets,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
        ).to(self.device)

        self.neptune_run["time"].log(self.start_time)

        self.logger = self.create_logger()
        self.train_batch_number = 0
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.initialize_model()

    @staticmethod
    def _mask_loss_tensor(loss: Tensor, mask: Tensor, per_sample=False) -> Tensor:
        """ calculates masked loss, so NN is not predicting blocked values """
        if per_sample:
            return (loss * mask).sum(axis=-1) / mask.sum(-1)
        else:
            return (loss * mask).sum() / mask.sum()

    def training(self, epoch: int):
        """ training loop """
        self.logger.info(f"Training at epoch {epoch}")
        self.model.train()
        total_loss = 0
        for i, (ranges, boards, pots, stacks, EVs) in tqdm(enumerate(self.train_dataloader, 1)):
            self.optimizer.zero_grad(set_to_none=True)
            ranges = ranges.to(self.device, non_blocking=True).reshape(-1, 2, 1326)
            pots = pots.to(self.device, non_blocking=True)
            stacks = stacks.to(self.device, non_blocking=True)
            EVs = EVs.to(self.device, non_blocking=True).reshape(-1, 2, 1326)
            matchups = get_matchups(ranges, boards)
            cfvs = EVs * matchups
            bucketed_ranges = bucketization_of_ranges(ranges, boards)
            bucketed_cfvs = bucketization_of_values(cfvs, boards)
            bucketed_EVs = bucketization_of_values(EVs, boards)
            predicted_values = self.model(torch.cat([bucketed_ranges.flatten(start_dim=1), pots / stacks], axis=1)).to(self.device).reshape_as(
                bucketed_EVs)
            if self.config.is_ev_based:
                bucketed_target = bucketed_EVs
            else:
                bucketed_target = bucketed_cfvs

            bucketed_loss = self.bucketed_loss_fun(predicted_values, bucketed_target)
            bucketed_mask = (bucketed_ranges > 0).to(self.device).to(torch.float32)
            bucketed_loss = self._mask_loss_tensor(bucketed_loss, bucketed_mask)
            bucketed_loss.backward()
            self.optimizer.step()
            total_loss += bucketed_loss.item()
            self.neptune_run["train/batch/loss"].log(bucketed_loss, step=self.train_batch_number)
            self.train_batch_number += 1
        mean_loss = total_loss / i
        self.neptune_run["train/epoch/loss"].log(mean_loss, step=epoch)

    def bucketed_validation(self, epoch: int):
        """ loop calculating bucketed loss """
        self.logger.info(f"Bucketed loss at epoch {epoch}")
        self.model.eval()
        total_buck_loss = 0
        n_samples = 0
        with torch.no_grad():
            for ranges, boards, pots, stacks, EVs in tqdm(self.val_dataloader):
                n_samples += ranges.shape[0]
                ranges = ranges.to(self.device, non_blocking=True).reshape(-1, 2, 1326)
                pots = pots.to(self.device, non_blocking=True)
                stacks = stacks.to(self.device, non_blocking=True)
                EVs = EVs.to(self.device, non_blocking=True).reshape(-1, 2, 1326)
                matchups = get_matchups(ranges, boards)
                cfvs = EVs * matchups
                bucketed_ranges = bucketization_of_ranges(ranges, boards)
                bucketed_cfvs = bucketization_of_values(cfvs, boards)
                bucketed_EVs = bucketization_of_values(EVs, boards)

                predicted_values = self.model(
                    torch.cat([bucketed_ranges.flatten(start_dim=1), pots / stacks], axis=1)).to(
                    self.device).reshape_as(
                    bucketed_EVs)
                if self.config.is_ev_based:
                    bucketed_target = bucketed_EVs
                else:
                    bucketed_target = bucketed_cfvs

                bucketed_loss = self.bucketed_loss_fun(predicted_values, bucketed_target)
                bucketed_mask = (bucketed_ranges > 0).to(self.device).to(torch.float32)
                bucketed_loss = self._mask_loss_tensor(bucketed_loss, bucketed_mask, per_sample=True).sum()
                total_buck_loss += bucketed_loss
        mean_buck_loss = total_buck_loss / n_samples
        self.neptune_run["test/epoch/buck_loss"].log(mean_buck_loss, step=epoch)

    def unbucketed_validation(self, epoch: int):
        """ loop calculating unbucketed loss """
        self.logger.info(f"Unbucketed loss at epoch {epoch}")
        self.model.eval()
        total_buck_loss = 0
        n_samples = 0

        total_rmse_unbuck_loss_per_player = torch.zeros(2).to(self.device)
        with torch.no_grad():
            for ranges, boards, pots, stacks, EVs in tqdm(self.val_dataloader):
                n_samples += ranges.shape[0]
                ranges = ranges.to(self.device, non_blocking=True).reshape(-1, 2, 1326)
                pots = pots.to(self.device, non_blocking=True)
                stacks = stacks.to(self.device, non_blocking=True)
                EVs = EVs.to(self.device, non_blocking=True).reshape(-1, 2, 1326)
                matchups = get_matchups(ranges, boards)
                cfvs = EVs * matchups
                bucketed_ranges = bucketization_of_ranges(ranges, boards)
                bucketed_cfvs = bucketization_of_values(cfvs, boards)
                bucketed_EVs = bucketization_of_values(EVs, boards)

                predicted_values = self.model(
                    torch.cat([bucketed_ranges.flatten(start_dim=1), pots / stacks], axis=1)).to(
                    self.device).reshape_as(
                    bucketed_EVs)
                if self.config.is_ev_based:
                    bucketed_target = bucketed_EVs
                else:
                    bucketed_target = bucketed_cfvs

                bucketed_loss = self.bucketed_loss_fun(predicted_values, bucketed_target)
                bucketed_mask = (bucketed_ranges > 0).to(self.device).to(torch.float32)
                bucketed_loss = self._mask_loss_tensor(bucketed_loss, bucketed_mask, per_sample=True).sum()
                total_buck_loss += bucketed_loss

                unbuck_pred_vals = unbucketization(predicted_values, boards)
                if self.config.is_ev_based:
                    unbuck_target = EVs
                else:
                    unbuck_target = cfvs

                unbuck_mask = (ranges > 0).to(self.device).to(torch.float32)

                mse_unbuck_loss_per_sph = self.MSELoss(unbuck_pred_vals / pots * 100, unbuck_target / pots * 100)  # (batch_size, n_players, n_hands)
                mse_unbuck_loss_per_sp = self._mask_loss_tensor(
                    mse_unbuck_loss_per_sph, unbuck_mask, per_sample=True
                )  # (batch_size, n_players)
                rmse_unbuck_loss_per_sp = torch.sqrt(mse_unbuck_loss_per_sp)
                rmse_unbuck_loss_per_p = rmse_unbuck_loss_per_sp.sum(dim=0)
                total_rmse_unbuck_loss_per_player += rmse_unbuck_loss_per_p

        rmse_unbuck_loss_per_p = total_rmse_unbuck_loss_per_player / n_samples
        mean_buck_loss = total_buck_loss / n_samples

        self.neptune_run["test/epoch/buck_loss"].log(mean_buck_loss, step=epoch)
        self.neptune_run["test/epoch/rmse_unbuck_loss"].log(rmse_unbuck_loss_per_p.mean(), step=epoch)
        torch.cuda.empty_cache()

    def fit(self):
        """ main training loop """
        for epoch in range(self.config.epochs):
            gc.collect()
            torch.cuda.empty_cache()
            self.training(epoch)
            self.save_model(epoch)
            if self.config.unbucketed_loss_freq is not None and (
                (epoch % self.config.unbucketed_loss_freq) == 0 or epoch == self.config.epochs - 1
            ):
                self.unbucketed_validation(epoch)
            else:
                self.bucketed_validation(epoch)
            torch.cuda.empty_cache()
            self.neptune_run["finished_epochs"] = epoch + 1

    def save_model(self, epoch):
        model_path = self.models_dir / str(epoch)
        torch.save(self.model, model_path)
        return model_path

    @cached_property
    def train_dataloader(self):
        """ util speeding up data loading """
        kwargs = {
            "num_workers": 1,
            "batch_size": self.config.batch_size,
            "shuffle": True,
            "persistent_workers": True,
            "prefetch_factor": 1
        }
        train_loader = DataLoader(self._get_trainset(), pin_memory=True, **kwargs)
        return train_loader

    @cached_property
    def val_dataloader(self):
        """ util speeding up data loading """
        kwargs = {"num_workers": 1, "batch_size": 1, "shuffle": False, "persistent_workers": True, "prefetch_factor": 1}
        train_loader = DataLoader(self._get_validset(), pin_memory=True, **kwargs)
        return train_loader

    def create_logger(self):
        logger = logging.getLogger("LOGGER")
        logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger

    def initialize_model(self):
        self.logger.info(f"Device is {self.device}")
        if "cuda" in str(self.device):
            self.logger.info("Setting torch backends.cudnn")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        self.logger.info(f"Length of training data = {self.train_len}")
        self.logger.info(f"Length of validation data = {self.val_len}")

    def _get_trainset(self):
        """ initialization of dataset """
        return torch.utils.data.TensorDataset(train_ranges, train_boards, train_pots, train_stacks, train_EVs)

    def _get_validset(self):
        """ initialization of dataset """
        return torch.utils.data.TensorDataset(valid_ranges, valid_boards, valid_pots, valid_stacks, valid_EVs)


