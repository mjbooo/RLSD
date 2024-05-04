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


class ItrWiseDataLoader(DataLoader):
    def __init__(
        self, 
        prompt_iterator, 
        get_target_itr_dataset, 
        tokenize_row, 
        batch_size, 
        split, 
        collate_fn,
        shuffle,
        num_workers,
        ):
        # At the beginning of each iteration, get the target (response) from prompt for whole D
        dataset_text = get_target_itr_dataset(prompt_iterator, split=split)
        dataset_tensor = dataset_text.map(tokenize_row, dataset_text)

        super().__init__(
            dataset_tensor,
            batch_size=batch_size, 
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
        )

class BatchWiseDataLoader(DataLoader):
    def __init__(
        self, 
        get_target_batch,
        prompt_iterator, 
        add_task_prompt,
        batch_size, 
        split, 
        itrwise_collate_fn,
        shuffle,
        num_workers,
        multiply_draft_pair,
        **kwargs,
        ):
        self.get_target_batch = get_target_batch

        self.add_task_prompt = add_task_prompt
        self.itrwise_collate_fn = itrwise_collate_fn
        self.split = split

        batch_size = batch_size * multiply_draft_pair if split!='train' else batch_size

        dataset_text = prompt_iterator.dataset
        super().__init__(
            dataset_text,
            batch_size=batch_size, 
            collate_fn=self.batchwise_collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def batchwise_collate_fn(self, features):
        # Todo: {DistillSpec/DPO}*{itreration/batch}
        prompts = self.add_task_prompt(features, is_collate=True)
        _features = self.get_target_batch(prompts, self.split)
        return self.itrwise_collate_fn.__call__(_features)