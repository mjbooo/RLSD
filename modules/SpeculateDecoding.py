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
    
    def yogesh():
        """
        Reference
        (1) https://github.com/lucidrains/speculative-decoding
        (2) HF: transforemrs.generation.utils.GenerationMixin.assisted_decoding

        """
        raise NotImplementedError