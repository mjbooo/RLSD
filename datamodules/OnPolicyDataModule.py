from collections import defaultdict
from typing import Optional, Union, Dict, List, Any
import copy

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
    
    def get_dataloader(self, split) -> DataLoader:
        shuffle = True if split == 'train' else False
        
        dataset_text = self.datasets[split]
        batch_size = self._config['batch_train'] if split!='valid_tiny' else 1


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