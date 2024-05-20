import os, random, sys
from absl import app
from absl import flags
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from collections import defaultdict

import torch
import torch.utils.data
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from accelerate.utils import tqdm


from DataLoader import PromptIterator
# add import path 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.util import get_task_prompt
from main import set_seed



FLAGS = flags.FLAGS
flags.DEFINE_string("root", "/pvc/home-mjlee", "root path")
flags.DEFINE_string("dataset", None, "The name of dataset")
flags.DEFINE_string("option", None, "only if option is needed")
flags.DEFINE_boolean("tgt_response", False, "get draft from the target model")
flags.DEFINE_string("tgt", "google/t5-xl-lm-adapt", "target model")
flags.DEFINE_integer("max_prompt_length", 1024, "max prompt length")
flags.DEFINE_integer("max_target_length", 64, "max target length")
flags.DEFINE_integer("batch_size", 48, "batch size")

flags.DEFINE_integer("seed", 2024, "seed")
flags.DEFINE_boolean("tiny_data", False, "for debugging")

map_to_prompt = {
'xsum': 'document',
'cnn_dailymail': 'article',
'wmt/wmt14': 'en',
}

@torch.no_grad()
def main(argv):
    set_seed(FLAGS.seed)
    dataset_name = FLAGS.dataset if FLAGS.dataset != 'wmt/wmt14' else 'wmt14'
    save_path = f'./datamodules/dataset/{dataset_name}' 

    if not FLAGS.tgt_response:
        dataset = load_dataset(FLAGS.dataset, FLAGS.option)
        # dpo_dataset_dict: "prompt", "chosen", "rejected"
        if FLAGS.dataset != 'wmt/wmt14':
            dataset = dataset.rename_column(map_to_prompt[FLAGS.dataset], 'prompt')
            dataset.save_to_disk(save_path)
        
        elif FLAGS.dataset == 'wmt/wmt14':
            datasets = {}
            for split in ['train', 'validation', 'test']:
                features = {'prompt': []}
                for i, item in enumerate(dataset[split]):
                    features['prompt'].append(item['translation']['en'])
                datasets[split] = Dataset.from_dict(features)
            dataset = DatasetDict(datasets)
            dataset.save_to_disk(save_path)
    
    else:
        datasets = load_from_disk(f"./datamodules/dataset/{dataset_name}")

        if FLAGS.tiny_data:
            datasets = DatasetDict(
                            dict(
                                train=Dataset(datasets['train']._data[:20]),
                                valid=Dataset(datasets['validation']._data[:5]),
                                test=Dataset(datasets['test']._data[:10]),
                            )
                        )

        tgt_model_name = FLAGS.tgt.split('/')[-1]

        save_path = os.path.join(save_path, f"{tgt_model_name}-{FLAGS.seed}")
        local_path = f"{FLAGS.root}/data/ImprovedSD/checkpoint/opensource/{FLAGS.tgt}"

        tgt_model = AutoModelForSeq2SeqLM.from_pretrained(local_path).to('cuda').eval()
        tgt_tokenizer = AutoTokenizer.from_pretrained(FLAGS.tgt)
        task_prompt = get_task_prompt(dataset_name)
        
        def preprocess_function(examples):
            model_inputs = tgt_tokenizer(
                        [task_prompt + doc for doc in examples["prompt"]],
                        max_length=FLAGS.max_prompt_length, 
                        return_tensors="pt",
                        padding=True, 
                        truncation=True, 
                        add_special_tokens=True,
                    )
            return model_inputs
        
        def collate_fn(batch):
            return {k: torch.stack([torch.LongTensor(example[k]) for example in batch]).to(tgt_model.device) for k in ['input_ids', 'attention_mask']}

        for split in ['train', 'valid', 'test']:
            tokenized_datasets = datasets[split].map(preprocess_function, batched=True)
            data_loader = PromptIterator(
                tokenized_datasets,
                batch_size=FLAGS.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                )

            decoded_samples = []
            for batch in tqdm(iterable=data_loader, desc=f"Generating {split} samples"):

                decoded_outputs = tgt_model.generate(
                                        **batch,
                                        max_new_tokens=FLAGS.max_target_length,
                                        do_sample=True,
                                        temperature=1,
                                        )
                decoded_sample = tgt_tokenizer.batch_decode(decoded_outputs, skip_special_tokens=True)
                decoded_samples.extend(decoded_sample)
            datasets[split] = datasets[split].add_column(f'{FLAGS.tgt}', decoded_samples)
        
        final_dataset = DatasetDict(datasets)
        final_dataset.save_to_disk(save_path)


if __name__ == '__main__':
    app.run(main)
"""
python3 datamodules/preprocess.py --dataset=cnn_dailymail --option=3.0.0
python3 datamodules/preprocess.py --dataset=xsum
python3 datamodules/preprocess.py --dataset=wmt/wmt14 --option='de-en'

# tgt_response
CUDA_VISIBLE_DEVICES=2 python3 datamodules/preprocess.py --dataset=xsum --tgt_response=True --seed=2024 ;\
CUDA_VISIBLE_DEVICES=2 python3 datamodules/preprocess.py --dataset=xsum --tgt_response=True --seed=2023 ;\
CUDA_VISIBLE_DEVICES=2 python3 datamodules/preprocess.py --dataset=xsum --tgt_response=True --seed=2022
CUDA_VISIBLE_DEVICES=1 python3 datamodules/preprocess.py --dataset=xsum --tgt_response=True --seed=2024 --batch_size=96

# debug
CUDA_VISIBLE_DEVICES=1 python3 datamodules/preprocess.py --dataset=xsum --tgt_response=True --seed=2024 --batch_size=96 --tiny_data=True
"""