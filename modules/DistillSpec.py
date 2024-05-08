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


class DistillSpec(Policy):
    def __init__(self, _config, sd, **kwargs):
        super().__init__(_config, sd)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        split: Literal["train", "valid", "test"] = "train",
    ):
        """
        Compute the KL loss  for distillation and other metrics for the given batch of inputs for train or test.
        """
        # 1. get output for (x, y) pair
        # <IMPORTANT> Todo: Check - (1) decoder_input_ids key: doesn't work / (2) labels key: works
        self.tgt_model.to(self.drf_model.device).eval()

        outputs_tgt = self.tgt_model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            output_attentions=False,
            output_hidden_states=False,
        )

        # 2. compute probabilities from logits
        """
        q_drf: (B, S, V) - S: real draft
        p_tgt: (B, S+1, V) - S+1: real draft + bonus token
        mask: (B, S+1) - S+1: start token + real draft
        """
        q_drf, p_tgt, mask = self.get_probability_and_mask(
            outputs_drf = batch, # from drf_model
            outputs_tgt = outputs_tgt, # from tgt_model
        )
        
        # 3. compute loss with probabilities
        losses = self.get_loss(
                            q_drf, 
                            p_tgt[:, :-1, :], # no bonus token for calculating the loss
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
    ):
        """
        Compute the probability for the given batch of inputs.
        # Input: two logit tensors from two models
        """
        # define the loss type for distillation
        label_pad_token_id = -100

        # mask for pad (True indicates pad token)
        labels_drf = outputs_drf['labels']
        mask_start = torch.zeros(size=(labels_drf.size(0), 1)).bool()
        mask_sequences = (labels_drf[:, 1:] == self.sd.tokenizer.pad_token_id).cpu() 
        mask = torch.cat([mask_start, mask_sequences], dim=1)

        # logits_drf, logits_tgt
        logits_drf = outputs_drf['logits_drf']
        logits_tgt = outputs_tgt.logits
        
        # get loss value
        q_drf = torch.softmax(logits_drf, dim=-1)
        p_tgt = torch.softmax(logits_tgt, dim=-1)

        return  q_drf, p_tgt, mask
    
    def get_loss(
        self,
        q_drf: torch.FloatTensor,
        p_tgt: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the loss for the given batch of inputs.
        # Input: two logit tensors from two models
        """

        # KL divergence loss
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        # masking out the padding part
        q_drf_masked = q_drf[~mask]
        p_tgt_masked = p_tgt[~mask]
        
        # get loss value
        loss = criterion(q_drf_masked.log(), p_tgt_masked)

        return loss
    
    def get_metrics(
        self,
        q_drf: torch.FloatTensor,
        p_tgt: torch.FloatTensor,
        labels_drf: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> Dict[str, torch.FloatTensor]:
        metrics = {}
        metric_tensor = {}
        custom_metrics = ['reward_exact']
        

        # 1. exact reward
        q_drf_labels = torch.gather(q_drf, -1, labels_drf.unsqueeze(-1)).squeeze().cpu()
        p_tgt_labels = torch.gather(p_tgt, -1, labels_drf.unsqueeze(-1)).squeeze().cpu()
        mask = mask.cpu()
        probability_ratio = p_tgt_labels / q_drf_labels

        # Don't count the padding tokens for the exact reward
        probability_ratio[mask] = 0
        num_token_drf = (~mask).sum(dim=1) # max 128

        acceptance_ratio = torch.min(probability_ratio, torch.tensor(1))
        metric_tensor['reward_exact'] = torch.cumprod(acceptance_ratio, dim=1).sum(dim=1)

        # gather metrics
        for _m in custom_metrics:
            # get metric itself and in ratio
            metrics[_m] = metric_tensor[_m].mean().item()
            metrics[_m + '_ratio'] = (metric_tensor[_m] / num_token_drf).mean().item()

        return metrics