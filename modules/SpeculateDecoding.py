import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple
from datasets.formatting import get_formatter, query_table, format_table
from transformers import PreTrainedModel
from einops import rearrange, reduce, repeat
import numpy as np
from absl import logging
from collections import defaultdict
import datasets

from typing import Any, Dict, List, Optional, Union
from accelerate.utils import tqdm

class SD(object):
    """
    Module for Speculate Decoding
    """
    def __init__(
            self,
            _config,
            drf_model,
            tgt_model,
            tokenizer_drf,
        ):
        self._config = _config

        self.drf_model = drf_model
        self.tgt_model = tgt_model
        self.tokenizer = tokenizer_drf
        self.padding_value = tokenizer_drf.pad_token_id
        #Todo: check label_pad_token_id
        # self.label_pad_token_id = _config['label_pad_token_id']

        self.is_encoder_decoder = self.drf_model.config.is_encoder_decoder

        self.policy = _config['policy']
        
        # DPO preprocessing
        self.max_prompt_length = _config['max_prompt_length']
        self.max_target_length = _config['max_target_length']
        self.max_chunk_length = _config['max_chunk_length']
    
    def sample(self, p):
        return np.random.choice(np.arange(p.shape[-1]), p=p)

    def max_fn(self, fx):
        # Apply max(0, f(x)) to zero out negative values
        positive_fx = torch.clamp(fx, min=0)

        # Normalize the positive values
        sum_positive_fx = positive_fx.sum()
        return positive_fx / sum_positive_fx
    
    @torch.no_grad
    def batch_sd(self, inputs_prompts):
        """
        Reference
        (1) https://github.com/lucidrains/speculative-decoding
        (2) HF: transforemrs.generation.utils.GenerationMixin.assisted_decoding
        
        N is the total tokens to generate
        K is the total tokens to generate by draft model for each chunk

        """
        N = self.max_target_length
        K = self.max_chunk_length
        
        task_prefix = "summarize: "

        articles = inputs_prompts['article']

        input_ids = self.tokenizer([task_prefix + article for article in articles], 
                        max_length=self.max_prompt_length,
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        add_special_tokens=True
                        )
        
        decoder_input_ids = torch.full((len(inputs_prompts), 1), self.tokenizer.pad_token_id, dtype=torch.long).to(self.device)

        n = 0
        accepted_tokens = 0

        while n < N:
            # Step 1: auto-regressive decode K tokens from draft model
            outputs_drf = self.drf_model.generate(
                input_ids=input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=K,
                do_sample=True,
                temperature=1,
                output_logits=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_tokens = outputs_drf.sequences[:, -K:]

            draft_sequences = outputs_drf.sequences[:, 1:]
            p = [logits.softmax(dim=1).topk(dim=1, k=1).values.squeeze().cpu().item() for logits in outputs_drf.logits]

            p_dist = [logits.softmax(dim=1).cpu() for logits in outputs_drf.logits]

            # Step 2: target model forward passes on x_draft
            with torch.no_grad():
                target_logits = self.tgt_model(
                    **inputs_prompts,
                    decoder_input_ids=draft_sequences,
                    output_attentions=False,
                    output_hidden_states=False
                )

            q = [logits.unsqueeze(dim=0).softmax(dim=1).topk(dim=1, k=1).values.squeeze().cpu().item() for logits in target_logits.logits[0]]
            q_dist = [logits.unsqueeze(dim=0).softmax(dim=1).cpu() for logits in target_logits.logits[0]]

            # Step 3: append draft tokens based on rejection criterion and resample
            all_accepted = True
            for t in range(K):
                rand = np.random.random()
                if rand < min(1, q[t] / p[t]):  # accepted
                    # update decoder inputs here, there will be a different number of tokens accepted
                    decoder_input_ids = torch.cat([decoder_input_ids, generated_tokens], dim=-1)
                    accepted_tokens += 1
                    n += 1
                else:  # rejected
                    resampled_token = self.sample(self.max_fn(q_dist[t] - p_dist[t]))  # resample from difference
                    # update decoder input with the resampled token
                    n += 1
                    all_accepted = False
                    break
            
            # Step 4: if all draft tokens were accepted, sample a final token from target model
            if all_accepted:
                decoder_input_ids = torch.cat([decoder_input_ids, generated_tokens], dim=-1)

        acceptance_rate = accepted_tokens / T
        return self.tokenizer.batch_decode(inputs_prompts["input_ids"]), acceptance_rate