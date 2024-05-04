from collections import defaultdict
from typing import Optional, Union, Dict, List, Any
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from transformers import PreTrainedModel, DataCollatorForSeq2Seq
from accelerate.utils import tqdm
from datamodules.collator import DataCollatorForSeq2SeqCustom
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
        self.task_prompt = get_task_prompt(self.dataset_name)
        # self.label_pad_token_id = _config['label_pad_token_id']

        self.max_prompt_length = _config['max_prompt_length']
        self.max_chunk_length = _config['max_chunk_length']
        self.max_target_length = _config['max_target_length']
        self.temperature = _config['temperature']

    def load_dataset(self):
        dataset = load_from_disk(f"./datamodules/dataset/{self.dataset_name}")
        
        if self._config['tiny_data']:
            return dict(
                train=Dataset(dataset['train']._data[:100]),
                valid=Dataset(dataset['validation']._data[:5]),
                test=Dataset(dataset['test']._data[:10]),
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

            dataset = datasets[split]

            if tokenize:
                """
                Todo: Tokenize the prompts
                """
                pass

            data_loaders[split] = PromptIterator(
                dataset,
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

    
    @torch.no_grad()
    def get_target_batch(self, prompts: List[str], split) -> List[Dict[str, torch.Tensor]]:
        """
        batch-wise target generation from prompt x
        """

        # generate drafts (on-the-fly while training)
        self.sd.drf_model.eval()
        self.sd.tgt_model.eval()
        outputs_drf, inputs_prompts = self.generate_draft_from_batch(prompts, split=split)
        
        # judge drafts (on-the-fly while training)
        self.sd.tgt_model.to('cuda')
        results = self.sd.judge_draft(outputs_drf=outputs_drf, inputs_prompts=inputs_prompts)

        d_datasets = dict(
            prompt=prompts,
        )
        
        for cls in self.target_cls:
            d_datasets[cls] = results[f"{cls}_batch"]
            # if split in ["valid", "test"]:
            for _metric in self.custom_metrics:
                d_datasets[f"{cls}_{_metric}"] = results[f"{cls}_{_metric}_batch"]

        # tokenizer row (on-the-fly while training)
        dataset_text = Dataset.from_dict(d_datasets)
        dataset_tensor = dataset_text.map(self.tokenize_row, dataset_text)  # dataset_num_proc
        
        # offloading
        inputs_prompts['input_ids'] = inputs_prompts['input_ids'].to('cpu')
        inputs_prompts['attention_mask'] = inputs_prompts['attention_mask'].to('cpu')
        outputs_drf['sequences'] = outputs_drf['sequences'].to('cpu')
        
        # build input for collator
        _features = [{k: _d[k] for k in dataset_tensor.column_names} for _d in dataset_tensor] 

        # Rollback the model to the original device
        self.sd.drf_model.train()

        return _features