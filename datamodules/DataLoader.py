from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from datasets import Dataset
from accelerate.utils import tqdm

class PromptIterator(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(
            dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
        )

class BatchWiseDataLoader(DataLoader):
    def __init__(
        self, 
        dataset_text,
        data_generation_policy, 
        add_task_prompt,
        batch_size, 
        split, 
        shuffle,
        num_workers,
        **kwargs,
        ):
        self.data_generation_policy = data_generation_policy

        self.add_task_prompt = add_task_prompt
        self.split = split

        self.label_pad_token_id = -100

        super().__init__(
            dataset=dataset_text,
            batch_size=batch_size, 
            collate_fn=self.collate_fn_onpolicy,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def collate_fn_onpolicy(self, features: List[Dict[str, Any]]):
        prompts = self.add_task_prompt(features, is_collate=True)
        
        return self.data_generation_policy(prompts, self.split)
    



