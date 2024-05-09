import os
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LambdaLR

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
from modules.SpeculateDecoding import SD
from datamodules.OnPolicyDataModule import OnPolicyDataModule
from utils.util import _save
from utils.metric import Metric


def get_trainer(policy):
    return Trainer

def get_policy(policy):
    policy_mapping = {
        'DistillSpec': DistillSpec,
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

        # Sanity check: validation at the starting point
        self.validate()
        for epoch in range(self._config['n_epochs']):
            for batch in tqdm(iterable=train_dataloader, desc=f"train: Epoch {epoch + 1}"):
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

    @torch.no_grad()
    def inference(self, split: Literal["valid", "test"]):
        eval_dataloader = self.datamodule.get_dataloader(split)

        self.drf_model.eval()
        for batch in tqdm(iterable=eval_dataloader, desc=f"[{split}: "):
            loss, metrics = self.policy.get_batch_loss_metrics(self.drf_model, batch, split=split)
            self.counter(loss, metrics, split=split)

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
        
        optimizer = torch.optim.AdamW(self.drf_model.parameters(), lr=self._config['lr'], weight_decay=0)
        if self._config['lr_scheduler'] == "fixed":
            return optimizer, None
        
        num_training_steps = self.counter.total_train_step
        num_warmup_steps = int(0.01 * num_training_steps)  

        if self._config['lr_scheduler'] == "cosine_warmup":
            lr_scheduler = optimization.get_cosine_schedule_with_warmup(
                                            optimizer, 
                                            num_warmup_steps=num_warmup_steps, 
                                            num_training_steps=num_training_steps
                                        )
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