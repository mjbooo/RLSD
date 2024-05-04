from collections import defaultdict
from typing import Optional, Union, Dict, List, Any
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from transformers import PreTrainedModel, DataCollatorForSeq2Seq
from accelerate.utils import tqdm
from datamodules.collator import DPODataCollatorWithPaddingCustom, DataCollatorForSeq2SeqCustom
from datamodules.DataLoader import PromptIterator, ItrWiseDataLoader, BatchWiseDataLoader

from modules.SpeculateDecoding import SD
from modules.Preference import RewardSetter
from utils.util import get_task_prompt, get_target_cls


class Datamodule:
    def __init__(self, _config, sd: SD):
        self._config = _config
        # dependency injection: model, tokenizer, judge_draft from SD
        self.sd = sd
        
        self.is_encoder_decoder = self.sd.is_encoder_decoder
        
        # dataset
        self.dataset_name = _config['dataset']
        self.task_prompt = get_task_prompt(self.dataset_name)
        self.prompt_dataloaders = self.get_prompt_dataloaders()
        # self.label_pad_token_id = _config['label_pad_token_id']

        # get_target_itr_dataset
        self.target_cls = get_target_cls(_config)
        self.draft_from = _config['draft_from']
        self.max_prompt_length = _config['max_prompt_length']
        self.max_target_length = _config['max_target_length']
        self.temperature = _config['temperature']

        self.custom_metrics = _config['custom_metrics']

        

    def load_dataset(self):
        dataset = load_from_disk(f"./datamodules/dataset/{self.dataset_name}")
        
        if self._config['tiny_data']:
            return dict(
                train=Dataset(dataset['train']._data[:100]),
                valid=Dataset(dataset['validation']._data[:5]),
                test=Dataset(dataset['test']._data[:10]),
            )
        elif self._config['num_same_train_valid']:
            n = self._config['num_same_train_valid']
            return dict(
                train=Dataset(dataset['train']._data[:n]),
                valid=Dataset(dataset['train']._data[:n]),
                test=Dataset(dataset['test']._data[:10]),
            )
        else:
            return dict(
                train=dataset['train'],
                valid=dataset['validation'],
                test=dataset['test'],
            )

    def get_prompt_dataloaders(self) -> DataLoader:
        datasets = self.load_dataset()

        data_loaders = dict()        
        for split in ['train', 'valid', 'test']:
            data_loaders[split] = PromptIterator(
                datasets[split],
                batch_size=self._config['batch_prompt'],
                shuffle=False)
        
        return data_loaders
    
    def get_collator(self):
        # RL, DistillSpec
        # elif is_encoder_decoder: True (seq2seq model)
        # Todo: Set arugments e.g., max length 
        return DataCollatorForSeq2SeqCustom(
            tokenizer=self.sd.tokenizer,
            padding=True,
            label_pad_token_id=self.label_pad_token_id,
        )
    
    def get_dataloader(self, split) -> DataLoader:
        shuffle = True if split == 'train' else False
        prompt_dataloader = self.prompt_dataloaders[split]
        data_collator = self.get_collator()

        kwargs_dataloader = dict(
            prompt_iterator=prompt_dataloader,
            batch_size=self._config['batch_train'],
            split=split,
            shuffle=shuffle,
            num_workers=0,            
        )
            
        if self.data_gen == "batch":
            return BatchWiseDataLoader(
                get_target_batch=self.get_target_batch,
                add_task_prompt=self.add_task_prompt,
                itrwise_collate_fn=data_collator,
                multiply_draft_pair=self.multiply_draft_pair,
                **kwargs_dataloader,
            )

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
        if self.draft_from == "drf":
            outputs = {
                "drf": self.generate_sequence(self.sd.drf_model, **inputs_prompts)
            }
        else:
            outputs = {
                "tgt": self.generate_sequence(self.sd.tgt_model, **inputs_prompts)
            }

        return outputs, inputs_prompts

    
    @torch.no_grad()
    def get_target_batch(self, prompts: List[str], split) -> List[Dict[str, torch.Tensor]]:

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