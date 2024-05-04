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