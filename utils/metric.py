from collections import defaultdict
from typing import Dict, Literal
import math

import torch
from torch.utils.data import DataLoader

class Metric:
    def __init__(self, _config, datamodule) -> None:
        self._config = _config
        self.n_epochs = datamodule.n_epochs
        
        self.cum_metrics = defaultdict(lambda: defaultdict(list))
        self.cum_train_step = 0

        self.train_step_per_epoch = datamodule.len_dataloaders['train']
        self.total_train_step = _config['max_training_steps'] if _config['max_training_steps'] else self.n_epochs * self.train_step_per_epoch
        self.logging_interval = self._get_interval(self._config['logging_steps'])
        self.valid_interval = self._get_interval(self._config['valid_steps'])
        self.valid_tiny_interval = self._get_interval(self._config['valid_tiny_steps'])
        self.no_valid_until = self._config['no_valid_until']
        
    def __call__(self, mean_loss=None, metrics: Dict[str, float]=None, optimizer=None, split: Literal["train", "eval", "test"] = "train") -> None:
        if mean_loss is not None:
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
        is_last_step = self.is_last_step()
        return is_logging_step or is_last_step
    
    def is_last_step(self) -> bool:
        return self.cum_train_step == self.total_train_step
    
    def is_valid(self) -> bool:
        is_whole_valid = self._config['whole_valid']
        is_predefined_no_valid = self.get_cum_epoch() >= self.no_valid_until
        is_count = self.cum_train_step % self.valid_interval == 0
        
        return is_count and is_predefined_no_valid and is_whole_valid
    
    def is_valid_tiny(self) -> bool:
        is_predefined_no_valid = self.get_cum_epoch() >= self.no_valid_until
        is_count = self.cum_train_step % self.valid_tiny_interval == 0
        
        return is_count and is_predefined_no_valid

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

    def get_cum_epoch(self):
        # if batch_size < len(dataset)
        train_step_per_epoch = self.train_step_per_epoch
        return math.ceil(self.cum_train_step / train_step_per_epoch)
    
    def state_dict(self):
        return {
            'cum_train_step': self.cum_train_step,
            'total_train_step': self.total_train_step,
            'n_epochs': self.n_epochs,
            'train_step_per_epoch': self.train_step_per_epoch,
        }
    def load_state_dict(self, state_dict):
        self.cum_train_step = state_dict['cum_train_step']
        self.total_train_step = state_dict['total_train_step']
        self.train_step_per_epoch = state_dict['train_step_per_epoch']
        self.n_epochs = state_dict['n_epochs']
