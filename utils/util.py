import os, glob, sys, random, pickle, copy, resource, logging
import torch
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
    "wmt14": ("wmt", "translation")
}

map_prompt = {
    "summarization": "summarize: ",
    "translation": "translate English to German: ",
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

def _save(model, optimizer, lr_scheduler, metric, save_dir, config):
    # Todo: optimzier state, scheduler state 
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()
    
    # save model
    model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=True)
    
    # save optimizer, scheduler
    torch.save({
        'metric_state_dict': metric.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict()
    }, os.path.join(save_dir, "optimizers.pt"))
    
    # save expt config
    with open(os.path.join(save_dir, 'config_sacred.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0