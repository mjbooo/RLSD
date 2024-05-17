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
        if self._config['weighted_logit']:
            # use weighted expectation over full vocab: (B, S, V)
            
            # 1. acceptance_ratio_mean
            min_p_q = torch.min(p_tgt, q_drf)
            min_p_q[mask] = 0 # Don't count the padding tokens for the exact reward
            acceptance_ratio_mean = min_p_q.sum(-1)

            # 2. acceptance_ratio_labels
            q_drf_labels = torch.gather(q_drf, -1, labels_drf.unsqueeze(-1)).squeeze(-1)
            p_tgt_labels = torch.gather(p_tgt, -1, labels_drf.unsqueeze(-1)).squeeze(-1)

            probability_ratio_labels = p_tgt_labels / q_drf_labels
            probability_ratio_labels[mask] = 0 # Don't count the padding tokens for the exact reward
            acceptance_ratio_labels = torch.min(probability_ratio_labels, torch.tensor(1))

            # 3. get weighted reward
            # logarithm
            acceptance_ratio_history = F.pad(acceptance_ratio_labels[..., :-1].cumprod(-1), (1, 0), value=1) # (B, S)
            _p_tgt_labels_history_log = F.pad(p_tgt_labels[..., :-1].log().cumsum(-1), (1, 0), value=1)
            _p_tgt_labels_history_log[mask] = - float('inf')
            p_tgt_labels_history = _p_tgt_labels_history_log.softmax(-1) # (B, S)

            acceptance_ratio_history[mask] = 0
            mat = acceptance_ratio_history.log()[..., None, :] - acceptance_ratio_history.log()[..., None]
            mask_triu = torch.triu(torch.ones_like(mat, dtype=torch.bool))
            mat[~mask_triu] = - float('inf')
            mat[mask] = - float('inf')
            mat = mat.exp()

            weighted_reward = (
                                mat # B, S, S
                                * acceptance_ratio_mean[..., None, :] # B, 1, S
                                * p_tgt_labels_history[..., None] # B, S, 1 
                            ).sum(dim=(1, 2))

            assert mat.isnan().sum().item() == 0
        
        elif self._config['full_logit']:
            # use expectation over full vocab: (B, S, V)
            q_drf_probs = q_drf
            p_tgt_probs = p_tgt

            min_p_q = torch.min(p_tgt_probs, q_drf_probs)

            # Don't count the padding tokens for the exact reward
            # mask = mask.to(probability_ratio.device)
            min_p_q[mask] = 0
            acceptance_ratio = min_p_q.sum(-1)
        else:
            # use specific (sampled) vocab: (B, S)

            q_drf_probs = torch.gather(q_drf, -1, labels_drf.unsqueeze(-1)).squeeze(-1)
            p_tgt_probs = torch.gather(p_tgt, -1, labels_drf.unsqueeze(-1)).squeeze(-1)
            # map_reward['q_drf_labels'] = q_drf_labels # save for improved reward

            probability_ratio = p_tgt_probs / q_drf_probs
            # mask = mask.to(probability_ratio.device)

            # Don't count the padding tokens for the exact reward
            probability_ratio[mask] = 0
            
            # 2. acceptance_rate_alpha
            acceptance_ratio = torch.min(probability_ratio, torch.tensor(1))

        # map_reward['acceptance_ratio'] = acceptance_ratio
        map_reward['acceptance_ratio_mean'] = acceptance_ratio_mean
        map_reward['acceptance_ratio_alpha_mean'] = (acceptance_ratio_mean.cpu().detach().sum(-1)/ map_reward['num_token_drf']).mean()
        map_reward['acceptance_ratio_labels'] = acceptance_ratio_labels
        map_reward['acceptance_ratio_alpha_labels'] = (acceptance_ratio_labels.cpu().detach().sum(-1)/ map_reward['num_token_drf']).mean()

        # 3. first block efficiency (gamma = 5)
        gammas = self._config['gammas']

        for _expectation in ['mean', 'labels']:
            acceptance_ratio = map_reward[f'acceptance_ratio_{_expectation}']
            # exact
            _cumprod = torch.cumprod(acceptance_ratio, dim=-1)
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
            if _expectation == 'mean':
                if self._config['weighted_logit']:
                    map_reward[f'exact_reward_{_expectation}'] = weighted_reward
            else: # labels
                exact_reward = _cumprod.sum(dim=-1)
                map_reward[f'exact_reward_{_expectation}'] = exact_reward

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
            elif _m == 'acceptance_ratio_alpha':
                metrics['acceptance_ratio_alpha_mean'] = metric_tensor['acceptance_ratio_alpha_mean'].item()
                metrics['acceptance_ratio_alpha_labels'] = metric_tensor['acceptance_ratio_alpha_labels'].item()
            else:
                _keys_exact_reward = ['exact_reward_labels']
                if self._config['weighted_logit']:
                    _keys_exact_reward += ['exact_reward_mean']
                for _key in _keys_exact_reward:
                    metrics[_key] = metric_tensor[_key].mean().item()
                    if not 'ratio' in _m:
                        metrics[_key + '_ratio'] = (metric_tensor[_key] / metrics['num_token_drf']).mean().item()
        return metrics