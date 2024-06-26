from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from datasets import Dataset
from transformers import PreTrainedModel
from accelerate.utils import tqdm
from modules.Policy import Policy
from utils.util import disable_dropout_in_model


class RL(Policy):
    def __init__(self, _config, sd, **kwargs):
        super().__init__(_config, sd)
        # disable dropout for draft model (DistillSpec Appendix)
        disable_dropout_in_model(self.sd.drf_model)

        self.improved_reward = _config['improved_reward']

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        split: Literal["train", "valid", "test"] = "train",
    ):
        """
        Compute the KL loss  for distillation and other metrics for the given batch of inputs for train or test.
        batch['labels']: (B, S+1) - S+1: start token added
        """
        # 1. get output for (x, y) pair and offload large model
        # <IMPORTANT> Todo: Check - (1) decoder_input_ids key: doesn't work / (2) labels key: works

        # get grad_fn
        if split=='train':
            outputs_drf = self.drf_model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['labels'][:, :-1], # To keep (B, S), not (B, S+1)
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            outputs_drf = batch
        
        with torch.no_grad():
            self.tgt_model.to(self.drf_model.device).eval()
            outputs_tgt = self.tgt_model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['labels'],
                output_attentions=False,
                output_hidden_states=False,
            )
        self.tgt_model.to('cpu').eval()

        # 2. compute probabilities from logits
        """
        q_drf: (B, S, V) - S: real draft
        p_tgt: (B, S+1, V) - S+1: real draft + bonus token
        mask: (B, S+1) - S+1: start token + real draft
        """
        q_drf, p_tgt, mask = self.get_probability_and_mask(
            outputs_drf = outputs_drf, # from drf_model
            outputs_tgt = outputs_tgt, # from tgt_model
            labels_drf = batch['labels']
        )
        
        # 3. compute loss with probabilities
        losses = self.get_loss(
                            q_drf, 
                            p_tgt[:, :-1, :], # no bonus token for calculating the loss
                            batch['labels'][:, 1:],
                            mask[:, :-1], # no bonus token for calculating the loss
                        )

        # 4. log custom metrics (sd) when evaluate
        metrics = self.get_metrics(
                            q_drf, 
                            p_tgt[:, :-1, :], # p_tgt: (B, S+1) by bonus token
                            batch['labels'][:, 1:], # don't indexing start token
                            mask[:, 1:], # don't indexing start token
                        )
        
        return losses, metrics

    def get_probability_and_mask(
        self, 
        outputs_drf: Dict[str, torch.Tensor],
        outputs_tgt: Dict[str, torch.Tensor],
        labels_drf
    ):
        """
        Compute the probability for the given batch of inputs.
        # Input: two logit tensors from two models
        """

        # mask for pad (True indicates pad token)
        mask_start = torch.zeros(size=(labels_drf.size(0), 1)).bool()
        mask_sequences = (labels_drf[:, 1:] == self.sd.tokenizer.pad_token_id).cpu() 
        mask = torch.cat([mask_start, mask_sequences], dim=1)

        # logits_drf, logits_tgt
        logits_drf = outputs_drf['logits']
        logits_tgt = outputs_tgt.logits
        
        # get loss value
        q_drf = torch.softmax(logits_drf, dim=-1)
        p_tgt = torch.softmax(logits_tgt, dim=-1)

        return  q_drf, p_tgt, mask
    
    def get_loss(
        self,
        q_drf: torch.FloatTensor,
        p_tgt: torch.FloatTensor,
        labels_drf: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the loss for the given batch of inputs.
        # Input: two logit tensors from two models
        """
        # loss = - exact reward
        reward_map = self.get_exact_reward(q_drf, p_tgt, labels_drf, mask)
        
        exact_reward = reward_map['exact_reward_for_loss'] # (B, )

        if self._config['truncation_deg']:
            trunc_reward = self.truncate_exact_reward(reward_map['acceptance_ratio'], self._config['truncation_deg'])
            reward = trunc_reward
        else:
            reward = exact_reward

        if self._config['improved_reward']:
            former_term = reward_map['q_drf_labels'].sum(dim=-1).log()
            latter_term = reward.clone().detach()            
            reward += former_term * latter_term

        losses = - reward

        return losses.mean()
    
    def truncate_exact_reward(self, x, degree):
        x = F.pad(x, (degree-1, 0), value=1)
        # Unroll over the sequence dimension
        windows = x.unfold(1, degree, 1)
        # Multiply elements within each window and sum the products
        return windows.prod(dim=-1).sum(-1)