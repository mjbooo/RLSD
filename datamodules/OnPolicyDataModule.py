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

from datamodules.DataModule import DataModule
from modules.SpeculateDecoding import SD

class OnPolicyDataModule(DataModule):
    def __init__(self, _config, sd: SD):
        super().__init__(_config, sd)
  
    def generate_sequence(self, model, **inputs_prompts):
        return super().generate_sequence(model, **inputs_prompts)
    
    @torch.no_grad()
    def generate_draft_from_batch(self, prompts: List[str], split: str) -> Dict[str, torch.Tensor]:
        return super().generate_draft_from_batch(prompts, split)
    
    def get_dataloader(self, split) -> DataLoader:
        shuffle = True if split == 'train' else False
        prompt_dataloader = self.get_prompt_dataloaders(tokenize=True)
        data_collator = self.get_collator()

        kwargs_dataloader = dict(
            prompt_iterator=prompt_dataloader,
            batch_size=self._config['batch_train'],
            split=split,
            shuffle=shuffle,
            num_workers=0,            
        )
            
        return BatchWiseDataLoader(
            get_target_batch=self.get_target_batch,
            add_task_prompt=self.add_task_prompt,
            itrwise_collate_fn=data_collator,
            **kwargs_dataloader,
        )
    
    def get_collator(self):
        # RL, DistillSpec
        # elif is_encoder_decoder: True (seq2seq model)
        # Todo: Set arugments e.g., max length 
        return DataCollatorForSeq2SeqCustom(
            tokenizer=self.sd.tokenizer,
            padding=True,
            label_pad_token_id=self.label_pad_token_id,
        )
        
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

        d_datasets = dict(
            prompt=prompts,
        )

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