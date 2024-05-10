from absl import app
from absl import flags
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "The name of dataset")
flags.DEFINE_string("option", None, "only if option is needed")

map_to_prompt = {
'xsum': 'document',
'cnn_dailymail': 'article',
'wmt/wmt14': 'en',
}

def main(argv):
    dataset = load_dataset(FLAGS.dataset, FLAGS.option)
    # dpo_dataset_dict: "prompt", "chosen", "rejected"
    if FLAGS.dataset != 'wmt/wmt14':
        dataset = dataset.rename_column(map_to_prompt[FLAGS.dataset], 'prompt')
        dataset.save_to_disk(f'./datamodules/dataset/{FLAGS.dataset}')
    
    elif FLAGS.dataset == 'wmt/wmt14':
        datasets = {}
        for split in ['train', 'validation', 'test']:
            features = {'prompt': []}
            for i, item in enumerate(dataset[split]):
                features['prompt'].append(item['translation']['en'])
            datasets[split] = Dataset.from_dict(features)
        DatasetDict(datasets).save_to_disk(f'./datamodules/dataset/wmt14')


if __name__ == '__main__':
    app.run(main)
"""
python3 datamodules/preprocess.py --dataset=cnn_dailymail --option=3.0.0
python3 datamodules/preprocess.py --dataset=xsum
python3 datamodules/preprocess.py --dataset=wmt/wmt14 --option='de-en'
"""