from collections import defaultdict
from typing import Optional, Union, Dict, List, Any
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from transformers import PreTrainedModel, DataCollatorForSeq2Seq
from accelerate.utils import tqdm
from datamodules.DataLoader import PromptIterator, BatchWiseDataLoader

from modules.SpeculateDecoding import SD
from utils.util import get_task_prompt


class DataModule:
    def __init__(self, _config, sd: SD):
        self._config = _config
        
        self.sd = sd
        self.is_encoder_decoder = self.sd.is_encoder_decoder
        
        # dataset
        self.dataset_name = _config['dataset']
        self.datasets = self.load_dataset()
        self.len_dataloaders = {k: len(v)/_config['batch_train'] for k, v in self.datasets.items()}
        self.task_prompt = get_task_prompt(self.dataset_name)

        self.max_prompt_length = _config['max_prompt_length']
        self.max_chunk_length = _config['max_chunk_length']
        self.max_target_length = _config['max_target_length']
        self.temperature = _config['temperature']

    def load_dataset(self):
        dataset = load_from_disk(f"./datamodules/dataset/{self.dataset_name}")
        
        if self._config['tiny_data']:
            return dict(
                train=Dataset(dataset['train']._data[:20]),
                valid=Dataset(dataset['validation']._data[:5]),
                test=Dataset(dataset['test']._data[:10]),
            )
        elif self._config['simple_setup']:
            return dict(
                train=Dataset(dataset['train']._data[:1000]),
                valid=Dataset(dataset['validation']._data[:100]),
                test=Dataset(dataset['test']._data[:100]),
            )
        else:
            return dict(
                train=dataset['train'],
                valid=dataset['validation'],
                test=dataset['test'],
            )

    def get_prompt_dataloaders(self, tokenize=False) -> DataLoader:
        datasets = self.load_dataset()

        data_loaders = dict()        
        for split in ['train', 'valid', 'test']:
            data_loaders[split] = PromptIterator(
                datasets[split],
                batch_size=self._config['batch_prompt'],
                shuffle=False)
        
        return data_loaders
    
    def get_collator(self):
        raise NotImplementedError
    
    def get_dataloader(self, split) -> DataLoader:
        raise NotImplementedError

    def add_task_prompt(self, features: List[Union[str, Dict]], is_collate: bool = False):
        if is_collate:
            return [self.task_prompt + f['prompt'] for f in features]
        return [self.task_prompt + p for p in features]

    @torch.no_grad()
    def generate_draft_from_batch(self, prompts: List[str], split: str) -> Dict[str, torch.Tensor]:
        self.sd.drf_model.eval()
        self.sd.tgt_model.eval()

        inputs_prompts = self.sd.tokenizer(
            prompts, 
            max_length=self.max_prompt_length, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            add_special_tokens=True
        ).to(self.sd.drf_model.device)
        
        # DistillSpec, RL
        outputs = self.generate_sequence(self.sd.drf_model, **inputs_prompts)

        return outputs, inputs_prompts

    def generate_sequence(self, model, **inputs_prompts):
        if self.is_encoder_decoder:
            return model.generate(
                **inputs_prompts,
                max_new_tokens=self.max_target_length,
                do_sample=True, 
                temperature=self.temperature, 
                output_logits=True, 
                return_dict_in_generate=True, 
                output_attentions=True,
            )
        else:
            raise NotImplementedError