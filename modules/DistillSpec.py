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
        metrics = {}

        # 1. get output for (x, y) pair
        # <IMPORTANT> Todo: Check - (1) decoder_input_ids key: doesn't work / (2) labels key: works
        # _batch = {k: batch.pop(k).to(self.drf_model.device) for k in ["input_ids", "attention_mask", "labels"]}
        
        self.tgt_model.to(self.drf_model.device).eval()
        # Todo: we can skip the forward in some cases (we already use sd to verify)

        outputs_tgt = self.tgt_model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            output_attentions=False,
            output_hidden_states=False,
        )

        # 2. compute loss with logits
        # losses = self.get_loss(logits_drf, logits_tgt[:, :-1, :])
        losses = self.get_loss(
                        outputs_drf = batch, # from drf_model
                        outputs_tgt = outputs_tgt, # from tgt_model
                        )

        # 3. log custom metrics (sd) when evaluate
        # if split in ["valid", "test"]:
        #     for k in self.custom_metrics:
        #         for cls in ["target"]:
        #             _l = batch[f"{cls}_{k}"]
        #             metrics[f"custom/{k}/{cls}"] = sum(_l)/len(_l)
        
        return losses, metrics
    
    def get_loss(
        self,
        outputs_drf: Dict[str, torch.Tensor],
        outputs_tgt: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the loss for the given batch of inputs.
        # Input: two logit tensors from two models
        # Todo: several types of loss (FKL, RKL, JSD...)
        """
        # define the loss type for distillation
        label_pad_token_id = -100
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        # filtering pad with -100 <IMPORTANT>: double check
        labels_drf = outputs_drf['labels']
        mask = labels_drf[:, 1:] == self.sd.tokenizer.pad_token_id
        labels_drf[:, 1:][mask] = label_pad_token_id

        # logits_drf, logits_tgt
        logits_drf = outputs_drf['logits_drf'] # (B, S, V)
        logits_tgt = outputs_tgt.logits[:, :-1, :]  # (B, S, V)

        # masking out the padding part
        logits_drf_masked = logits_drf[~mask]
        logits_tgt_masked = logits_tgt[~mask]
        
        
        # get loss value
        q_drf_flattened = F.log_softmax(logits_drf_masked, dim=-1)
        p_tgt_flattened = torch.softmax(logits_tgt_masked, dim=-1)
        loss = criterion(q_drf_flattened, p_tgt_flattened)

        return loss