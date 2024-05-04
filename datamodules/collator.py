from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from trl.trainer.utils import DPODataCollatorWithPadding
from transformers import DataCollatorForSeq2Seq


class DPODataCollatorWithPaddingCustom(DPODataCollatorWithPadding):
    def __init__(self, **kwargs):
        self.custom_metrics = kwargs.pop("custom_metrics")
        super().__init__(**kwargs)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = super().__call__(features)

        # Collating custom metrics
        for k in features[0].keys():
            if any([k.endswith(_metric) for _metric in self.custom_metrics]):
                padded_batch[k] = [f[k] for f in features]
        return padded_batch

class DataCollatorForSeq2SeqCustom(DataCollatorForSeq2Seq):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        column_map = {
            "prompt_input_ids": "input_ids",
            "prompt_attention_mask": "attention_mask",
            "target_labels": "labels",
        }
        _features = [{v: f[k] for k, v in column_map.items()} for f in features]
        batch = super().__call__(_features)
        
        # To add the other custom metrics
        for k in features[0].keys():
            if k not in column_map.keys():
                batch[k] = [f[k] for f in features]

        return batch
