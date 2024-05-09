import os, glob, sys, random, pickle, copy, resource, logging
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
import yaml

map_name_task = {
    # model
    "google-t5/t5-small": "T5small",
    "google-t5/t5-base": "T5base",
    "google-t5/t5-large": "T5large",
    "google-t5/t5-3b": "T5xl",
    "google-t5/t5-11b": "T5xxl",

    "google/t5-small-lm-adapt": "T5lm-small",
    "google/t5-base-lm-adapt": "T5lm-base",
    "google/t5-large-lm-adapt": "T5lm-large",
    "google/t5-xl-lm-adapt": "T5lm-xl",
    "google/t5-xxl-lm-adapt": "T5lm-xxl",


    # dataset
    "cnn_dailymail": ("cnndm", "summarization"),
    "xsum": ("xsum", "summarization"),
}

map_prompt = {
    "summarization": "summarize: ",
}
    
def get_short_name(name_obj: str):
    if name_obj not in map_name_task:
        return name_obj
    mapped = map_name_task[name_obj]
    # (Dataset, Task)
    if isinstance(mapped, tuple):
        return map_name_task[name_obj][0]
    return mapped

def get_task_name(name_obj: str):
    if name_obj not in map_name_task:
        return name_obj
    return map_name_task[name_obj][1]

def get_task_prompt(name_obj: str):
    task_name = get_task_name(name_obj)
    return map_prompt[task_name]

def _save(model, save_dir, config):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()
    model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=True)
    model.save_pretrained(save_dir)
    with open(os.path.join(save_dir, 'config_sacred.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)