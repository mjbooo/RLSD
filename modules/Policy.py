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
        self.sd = sd
        self._config = _config

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
        """
        No start token in input
        """
        map_reward = dict()

        # 0. num_token_drf
        map_reward['num_token_drf'] = (~mask).sum(dim=1).cpu() # max 128

        # 1. exact reward
        q_drf_labels = torch.gather(q_drf, -1, labels_drf.unsqueeze(-1)).squeeze(-1)
        p_tgt_labels = torch.gather(p_tgt, -1, labels_drf.unsqueeze(-1)).squeeze(-1)
        map_reward['q_drf_labels'] = q_drf_labels # save for improved reward

        probability_ratio = p_tgt_labels / q_drf_labels
        mask = mask.to(probability_ratio.device)

        # Don't count the padding tokens for the exact reward
        probability_ratio[mask] = 0
        
        # 2. acceptance_rate_alpha
        acceptance_ratio = torch.min(probability_ratio, torch.tensor(1))
        map_reward['acceptance_ratio'] = acceptance_ratio
        map_reward['acceptance_ratio_alpha'] = (acceptance_ratio.cpu().detach().sum(-1)/ map_reward['num_token_drf']).mean()

        # 3. first block efficiency (gamma = 5)
        _cumprod = torch.cumprod(acceptance_ratio, dim=-1)
        
        # exact
        gammas = self._config['gammas']
        exact_reward_first_chunk = _cumprod[..., :max(gammas)].cpu().detach().cumsum(-1)
        # random
        acceptance_ratio_first_chunk = acceptance_ratio[..., :max(gammas)].cpu().detach()
        is_accepted = torch.rand_like(acceptance_ratio_first_chunk) < acceptance_ratio_first_chunk
        random_reward_first_chunk = ((~is_accepted).cumsum(dim=-1) < 1).cumsum(-1)  # this is `n` in algorithm 1

        for g in gammas:
            # added 1 token from the target model
            map_reward[f'first_block_efficiency_{g}_exact'] = exact_reward_first_chunk[..., g-1].mean() + 1
            map_reward[f'first_block_efficiency_{g}_random'] = random_reward_first_chunk[..., g-1].float().mean() + 1

        # 4. exact_reward
        exact_reward = _cumprod.sum(dim=-1)
        map_reward['exact_reward'] = exact_reward

        return map_reward

    def gather_metrics(self, metric_tensor: Dict[str, torch.Tensor]):
        metrics = {}
        metrics['num_token_drf'] = metric_tensor['num_token_drf'].float().mean().item()
        for _m in self.custom_metrics:
            # get metric itself and in ratio
            if _m == 'first_block_efficiency':
                for g in self._config['gammas']:
                    metrics[f'{_m}_{g}_exact'] = metric_tensor[f'{_m}_{g}_exact'].mean().item()
                    metrics[f'{_m}_{g}_random'] = metric_tensor[f'{_m}_{g}_random'].mean().item()
                continue

            metrics[_m] = metric_tensor[_m].mean().item()
            if not 'ratio' in _m:
                metrics[_m + '_ratio'] = (metric_tensor[_m] / metrics['num_token_drf']).mean().item()
        
        return metrics