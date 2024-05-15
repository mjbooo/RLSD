from collections import defaultdict
from typing import Optional, Union, Dict, List, Any
import copy
import random
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from einops import rearrange

from transformers import PreTrainedModel, DataCollatorForSeq2Seq
from accelerate.utils import tqdm
from datamodules.DataLoader import PromptIterator, BatchWiseDataLoader


from datamodules.DataModule import DataModule
from modules.SpeculateDecoding import SD

class OnPolicyDataModule(DataModule):
    def __init__(self, _config, sd: SD):
        super().__init__(_config, sd)
        # max_training_steps overrides n_epochs
        if _config['max_training_steps']:
            self.n_epochs = math.ceil(_config['max_training_steps']/self.len_dataloaders['train'])
        else:
            self.n_epochs = _config['n_epochs']
    
    def get_dataloader(self, split) -> DataLoader:
        shuffle = True if split == 'train' else False
        
        if split != 'valid_tiny':
            dataset_text = self.datasets[split]
        else:
            # sampling subset from valid set
            n = self._config['num_valid_tiny'] if not self._config['tiny_data'] else 3
            random_ids = random.sample(range(len(self.datasets['valid'])), n)
            dataset_text = self.datasets['valid'].select(random_ids)
        batch_size = 1 if split in ['valid_tiny', 'test'] else self._config['batch_train']


        kwargs_dataloader = dict(
            dataset_text=dataset_text,
            batch_size=batch_size,
            split=split,
            shuffle=shuffle,
            num_workers=0,            
        )
            
        return BatchWiseDataLoader(
            data_generation_policy=self.get_target_onpolicy,
            add_task_prompt=self.add_task_prompt,
            **kwargs_dataloader,
        )
        
    @torch.no_grad()
    def get_target_onpolicy(self, prompts: List[str], split) -> List[Dict[str, torch.Tensor]]:
        """
        batch-wise target generation from prompt x
        """

        # generate drafts (on-the-fly while training)
        self.sd.drf_model.eval().to('cuda')
        outputs_drf, inputs_prompts = self.generate_draft_from_batch(prompts, split=split)

        # Rollback the model to the original state
        self.sd.drf_model.train()

        # build the batched features
        _features = {
            'input_ids': inputs_prompts['input_ids'],
            'attention_mask': inputs_prompts['attention_mask'],
            'labels': outputs_drf['sequences'],
            'logits': rearrange(torch.stack(outputs_drf.logits), 's b v -> b s v')
        }
        return _features