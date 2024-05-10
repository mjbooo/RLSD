from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from datasets import Dataset
from transformers import PreTrainedModel
from trl.trainer.utils import pad_to_length
from accelerate.utils import tqdm


class Policy(object):
    def __init__(self, _config, sd, **kwargs):
    # def __init__(self, _config, sd, ref_model, custom_metrics):
        self.sd = sd
        self.drf_model = sd.drf_model
        self.tgt_model = sd.tgt_model
        self.is_encoder_decoder = sd.drf_model.config.is_encoder_decoder


        self.tokenizer = sd.tokenizer
        self.padding_value = self.tokenizer.pad_token_id

        self.max_prompt_length = _config['max_prompt_length']
        self.max_chunk_length = _config['max_chunk_length']
        self.max_target_length = _config['max_target_length']
        self.temperature = _config['temperature']

        self.custom_metrics = _config['custom_metrics']

    def get_loss(
        self,
    ):
        raise NotImplementedError

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        split: Literal["train", "valid", "test"] = "train",
    ):
        raise NotImplementedError

    def get_exact_reward(
        self,
        q_drf: torch.FloatTensor,
        p_tgt: torch.FloatTensor,
        labels_drf: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        # 1. exact reward
        q_drf_labels = torch.gather(q_drf, -1, labels_drf.unsqueeze(-1)).squeeze(-1).cpu()
        p_tgt_labels = torch.gather(p_tgt, -1, labels_drf.unsqueeze(-1)).squeeze(-1).cpu()

        mask = mask.cpu()
        probability_ratio = p_tgt_labels / q_drf_labels

        # Don't count the padding tokens for the exact reward
        probability_ratio[mask] = 0

        acceptance_ratio = torch.min(probability_ratio, torch.tensor(1))

        return torch.cumprod(acceptance_ratio, dim=1).sum(dim=1)