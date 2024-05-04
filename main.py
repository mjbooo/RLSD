import os, glob, sys, random, pickle, copy, resource

import torch
import torch.multiprocessing as mp
import numpy as np

from absl import logging
from config import ex
import wandb

from modules.model import load_models, load_tokenizers
from trainer import get_trainer

def set_seed(seed):
	mp.set_sharing_strategy('file_system')
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False    

@ex.automain
def main(_config):
	# Set seed, and checkpoint name
	set_seed(_config['seed'])

	# Load models, tokenizers, and dataset
	drf_model, tgt_model = load_models(_config) 

	"""Todo: <Important> compatability b/w two tokenizers: when sd 
	Refer Ludcidrain SD"""

	drf_tokenizer, tgt_tokenizer = load_tokenizers(
										_config, 
										max_target_length=_config['max_target_length'],
									)

	trainer = get_trainer(_config['policy'])(
		_config,
		drf_model=drf_model,
		tgt_model=tgt_model,
		tokenizer=drf_tokenizer,
	)

	if not _config['eval']:
		logging.info(f"[Overall] Training Starts ...")
		trainer.train()
		
		logging.info(f"[Overall] Saving the model to {trainer.output_dir} ...")
		trainer.save_model()	
	  
	logging.info(f"[Overall] Final Evaluation Starts ...") 
	trainer.test()
	
	if not _config['debug']:
		wandb.finish(0)