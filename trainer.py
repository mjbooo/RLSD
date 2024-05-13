import os
import warnings
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
# import torch.optim.lr_scheduler as LambdaLR
from torch.optim.lr_scheduler import LambdaLR

from datasets.formatting import get_formatter, query_table, format_table
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    optimization,
)
from accelerate.utils import tqdm
from absl import logging
import wandb

from modules.DistillSpec import DistillSpec
from modules.RL import RL
from modules.SpeculateDecoding import SD
from datamodules.OnPolicyDataModule import OnPolicyDataModule
from utils.util import _save
from utils.metric import Metric


def get_trainer(policy):
    return Trainer

def get_policy(policy):
    policy_mapping = {
        'DistillSpec': DistillSpec,
        'RL': RL,
    }
    return policy_mapping[policy]

def get_data_module(data_module):
    data_module_mapping = {
        'batch': OnPolicyDataModule,
    }
    return data_module_mapping[data_module]

class Trainer(object):
    def __init__(
        self,
        _config,
        drf_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        tgt_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):        
        self._config = _config

        # Speculative Decoding 
        self.drf_model = drf_model
        self.tgt_model = tgt_model
        self.tokenizer = tokenizer
        self.sd = SD(_config, self.drf_model, self.tgt_model, self.tokenizer)
        
        # policy (DS, RL)
        self.policy = get_policy(_config['policy'])(_config, self.sd)

        # DataModule
        self.datamodule = get_data_module(_config['data_gen'])(_config, self.sd)

        # # Save/load, logging
        self.counter = Metric(_config, self.datamodule)
        self.output_dir = _config['output_dir']
        self.debug = _config['debug']

        if not self.debug:
            wandb.init(
                entity="furiosaai",
                project=str(_config['wandb_project_name']),
                config=_config,
                reinit=True
            )
            wandb.run.name = _config['ckpt_save']

    def train(self):
        # get train dataloader with new response
        train_dataloader = self.datamodule.get_dataloader("train")
        self.optimizer, self.lr_scheduler = self.get_optimizers()

        if self._config['initial_valid']:
            self.validate() # Sanity check: validation at the starting point
        for epoch in range(self.datamodule.n_epochs):
            for batch in tqdm(iterable=train_dataloader, desc=f"train: Epoch {epoch + 1}, Steps {self.counter.cum_train_step}"):
                self.drf_model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.policy.get_batch_loss_metrics(self.drf_model, batch, split="train")
                loss.backward()
                self.optimizer.step()
                
                self.counter(loss, metrics, self.optimizer, split="train")
                
                if self.counter.is_logging():
                    self.log()
                if self.counter.is_valid():
                    self.validate()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                if self.counter.is_last_step():
                    return

    @torch.no_grad()
    def measure_block_efficiency(self, split: Literal["valid_tiny", "test"]):
        # Todo: AR for tiny valid data
        tiny_valid_dataloader = self.datamodule.get_dataloader(split)
        self.sd.tgt_model.to(self.sd.drf_model.device).eval()
        for batch in tqdm(iterable=tiny_valid_dataloader, desc=f"[{split}: "):
            del batch['logits']
            # chunk length = 5 by default
            decoded_sample, sd_metrics = self.sd.tgt_model.generate(
                                    **batch,
                                    max_new_tokens=self._config['max_target_length'],
                                    do_sample=True,
                                    assistant_model=self.sd.drf_model,
                                )
            
            # decode_sample = 1 (start token) + cum_n_matches + num_itr
            metrics = {
                'block_efficiency_ratio': 1 + sd_metrics['cum_n_matches'] / sd_metrics['num_itr'],
                'gamma': sd_metrics['gamma'],
                'match_first': sd_metrics['cum_n_matches'],
                'match_first_ratio': sd_metrics['cum_n_matches'] / (sd_metrics['num_itr']*sd_metrics['gamma']),
                }

            self.counter(torch.tensor([-100]), metrics, split=split)
        
        if not self.debug:
            wandb.log(self.counter.get_log(split))
        self.sd.tgt_model.to('cpu').eval()
    
    @torch.no_grad()
    def inference(self, split: Literal["valid", "test"]):
        eval_dataloader = self.datamodule.get_dataloader(split)

        self.drf_model.eval()
        for batch in tqdm(iterable=eval_dataloader, desc=f"[{split}: "):
            loss, metrics = self.policy.get_batch_loss_metrics(self.drf_model, batch, split=split)
            self.counter(loss, metrics, split=split)
        
        split_tiny = "valid_tiny" if split == "valid" else "test"
        self.measure_block_efficiency(split_tiny)

        if not self.debug:
            wandb.log(self.counter.get_log(split))

    def validate(self):
        self.inference("valid")

    def test(self):
        self.inference("test")
    
    @torch.no_grad()
    def log(self):
        if not self.debug:
            wandb.log(self.counter.get_log("train"))
    
    def get_optimizers(self):
        # Todo: Optimizer: Warm-up, cool-down schdule
        
        if self._config['optimizer'] == "adamw":
            optimizer = torch.optim.AdamW(self.drf_model.parameters(), lr=self._config['lr'], weight_decay=0)
        elif self._config['optimizer'] == "adafactor":
            optimizer = optimization.Adafactor(self.drf_model.parameters(), lr=self._config['lr'], relative_step=False, scale_parameter=False, warmup_init=False)
                    
        if self._config['lr_scheduler'] == "fixed":
            return optimizer, None
        elif self._config['lr_scheduler'] == "linear_warmup_cosine_decay":
            warmup_steps = int(1/60 * self.counter.total_train_step)
            cooldown_start = int(1/2 * self.counter.total_train_step)
            cooldown_end = self.counter.total_train_step
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                elif current_step < cooldown_start:
                    return 1.0
                else:
                    progress = float(current_step - cooldown_start) / float(max(1, cooldown_end - cooldown_start))
                    return 0.45 * (1.0 + math.cos(math.pi * progress)) + 0.1
            
            lr_scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Invalid lr_scheduler: {self._config['lr_scheduler']}")
        return optimizer, lr_scheduler
    
    def save_model(self):
        """
        Save the drf_model for iteration
        """
        save_dir = self.output_dir
        logging.info(f"[Saving the drf_model to {save_dir} ...")

        _save(
            model=self.sd.drf_model, 
            save_dir=save_dir,
            config=self._config,)