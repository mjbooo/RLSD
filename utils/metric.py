from collections import defaultdict
from typing import Dict, Literal
import math

import torch
from torch.utils.data import DataLoader

class Metric:
    def __init__(self, _config, datamodule) -> None:
        self._config = _config
        self.n_epochs = _config['n_epochs']
        
        self.cum_metrics = defaultdict(lambda: defaultdict(list))
        self.cum_train_step = 0

        self.train_step_per_epoch = datamodule.len_dataloaders['train']
        self.total_train_step = self.n_epochs * self.train_step_per_epoch
        self.logging_interval = self._get_interval(self._config['logging_steps'])
        self.valid_interval = self._get_interval(self._config['valid_steps'])
        
    def __call__(self, mean_loss, metrics: Dict[str, float], optimizer=None, split: Literal["train", "eval", "test"] = "train") -> None:
        self.cum_metrics[split]["loss"].append(mean_loss.detach().item())

        for key, value in metrics.items():
            self.cum_metrics[split][key].append(value)
        
        if split == "train":
            self.cum_metrics[split]["lr"] = optimizer.param_groups[0]['lr']
            self.cum_train_step += 1
            
    def is_logging(self) -> bool:
        """
        Only for training stage, return True if current step is logging step 
        """
        is_logging_step = self.cum_train_step % self.logging_interval == 0
        is_last_step = self.cum_train_step == self.total_train_step
        return is_logging_step or is_last_step
    
    def is_valid(self) -> bool:
        return self.cum_train_step % self.valid_interval == 0

    def get_log(self, split) -> None:
        """return dict for wandb logging"""
        logs = dict()
        
        logs[f'step'] = self.cum_train_step
        logs[f'epoch'] = self.cum_train_step / self.train_step_per_epoch
        if split == "train":
            logs['lr'] = self.cum_metrics[split]["lr"]
        for key, metrics in self.cum_metrics[split].items():
            logs[f"{split}/{key}"] = torch.tensor(metrics).float().mean().item()
        del self.cum_metrics[split]
        
        return logs

    def _get_interval(self, target_step):
        # When step < 1, it means it is a ratio of interval over a single epoch
        if target_step < 1:
            return math.ceil(self.train_step_per_epoch * target_step)
        return target_step