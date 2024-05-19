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


class DistillSpec(Policy):
    def __init__(self, _config, sd, **kwargs):
        super().__init__(_config, sd)
        # disable dropout for draft model (DistillSpec Appendix)
        disable_dropout_in_model(self.sd.drf_model)

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
        
        self.tgt_model.to(self.drf_model.device).eval()
        with torch.no_grad():
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
        q_drf, p_tgt, mask, log_q_drf = self.get_probability_and_mask(
            outputs_drf = outputs_drf, # from drf_model
            outputs_tgt = outputs_tgt, # from tgt_model
            labels_drf = batch['labels']
        )
        
        # 3. compute loss with probabilities
        losses = self.get_loss(
                            q_drf, 
                            p_tgt[:, :-1, :], # no bonus token for calculating the loss
                            mask[:, :-1], # no bonus token for calculating the loss
                            log_q_drf,
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

        
        # Numerical stability for KL
        if self._config['divergence'] == 'kl':
            return q_drf, p_tgt, mask, torch.log_softmax(logits_drf, dim=-1)

        return q_drf, p_tgt, mask, None
    
    def get_loss(
        self,
        q_drf: torch.FloatTensor,
        p_tgt: torch.FloatTensor,
        mask: torch.BoolTensor,
        log_q_drf: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the loss for the given batch of inputs.
        # Input: two logit tensors from two models
        """
        # masking out the padding part
        if log_q_drf is not None:
            q_drf_masked = log_q_drf[~mask]
        else:
            q_drf_masked = q_drf[~mask]

        p_tgt_masked = p_tgt[~mask]

        if self._config['divergence'] == 'kl':
            # KL divergence loss
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
            # q_drf_masked from torch.logsoftmax
            loss = criterion(q_drf_masked, p_tgt_masked)

        elif self._config['divergence'] == 'tvd':
            # Total Variation Distance loss
            criterion = torch.nn.L1Loss(reduction='mean')
            loss = criterion(q_drf_masked, p_tgt_masked)/2

        return loss