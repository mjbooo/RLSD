# Load model directly
import os
from absl import flags, logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import ex

def load_models(_config):
	models = []
	for role in ['drf', 'tgt']:
		model_name = _config[role]
		if _config['ckpt_dir'] is None:
			local_path = f"{_config['root']}/data/ImprovedSD/checkpoint/opensource/{model_name}"
		else:
			logging.info(f"Load selected model checkpoint from {_config['ckpt_dir']}...")
			local_path = f"{_config['ckpt_dir']}/{model_name}"
		
		if not os.path.exists(local_path):
			logging.info(f"No model found in local_path! Download & Save {model_name}...")
			model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
			model.save_pretrained(local_path)

		model = AutoModelForSeq2SeqLM.from_pretrained(local_path)		
		models.append(model)

	return models[0], models[1].eval()

def load_tokenizers(_config, max_target_length):
	drf_tokenizer = AutoTokenizer.from_pretrained(_config['drf'], model_max_length=max_target_length)
	tgt_tokenizer = AutoTokenizer.from_pretrained(_config['tgt'], model_max_length=max_target_length)

	return drf_tokenizer, tgt_tokenizer