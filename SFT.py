import numpy as np
import math

from absl import app, flags
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_scheduler
from transformers.optimization import Adafactor, AdafactorSchedule
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_from_disk, Dataset
import evaluate
import wandb

from utils.util import get_task_prompt, disable_dropout_in_model

FLAGS = flags.FLAGS
flags.DEFINE_string("root", "/pvc/home-mjlee", "root path")
flags.DEFINE_string("model_repo", "google/t5-small-lm-adapt", "The name of model repository") # "google/t5-small-lm-adapt", "google/t5-xl-lm-adapt"
flags.DEFINE_string("dataset", "xsum", "The name of dataset")

# Task option
flags.DEFINE_string("task", "summarization", "The name of task")
flags.DEFINE_integer("max_prompt_length", 1024, "The maximum length of prompt")
flags.DEFINE_integer("max_target_length", 64, "The maximum length of target")

# Training option
flags.DEFINE_float("learning_rate", 3e-4, "The learning rate of model")
flags.DEFINE_integer("max_training_steps", 100000, "batch size for training") # 100000, 20000

flags.DEFINE_integer("batch_train", 32, "batch size for training")
flags.DEFINE_integer("batch_eval", 128, "batch size for evaluation")
flags.DEFINE_string("wandb_project_name", "240512_SFT", "The name of wandb project")
flags.DEFINE_boolean("tiny_data", False, "Use tiny data for debugging")


def main(argv):
    
    def compute_metrics(eval_pred):
        rouge = evaluate.load("rouge")

        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    
    datasets = load_from_disk(f"./datamodules/dataset/{FLAGS.dataset}")
    if FLAGS.tiny_data: # For debugging
        datasets['train'] = Dataset.from_dict(datasets['train'][:100])
        datasets['validation'] = Dataset.from_dict(datasets['validation'][:20])
        datasets['test'] = Dataset.from_dict(datasets['test'][:50])

    
    # For loading model
    local_path = f"{FLAGS.root}/data/ImprovedSD/checkpoint/opensource/{FLAGS.model_repo}"
    model = AutoModelForSeq2SeqLM.from_pretrained(local_path)		
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_repo)
    # Fix wrong dropout probability of google/t5-small-lm-adapt
    disable_dropout_in_model(model)
    
    # For saving
    model_name = FLAGS.model_repo.split("/")[-1]
    run_name = f"{model_name}-{FLAGS.dataset}-{FLAGS.learning_rate}"
    output_dir = f"{FLAGS.root}/data/ImprovedSD/checkpoint/SFT/{run_name}"

    def preprocess_function(examples):
        prefix = get_task_prompt(FLAGS.task)
        
        inputs = [prefix + doc for doc in examples["prompt"]]
        model_inputs = tokenizer(inputs, max_length=FLAGS.max_prompt_length, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=FLAGS.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    wandb.init(
        entity="furiosaai",
        project=str(FLAGS.wandb_project_name),
        config=FLAGS,
        reinit=True
    )
    wandb.run.name = run_name

    optimizer = Adafactor(
                    model.parameters(), 
                    lr=FLAGS.learning_rate,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
                )

    # Define the custom learning rate schedule function
    warmup_steps = int(1/60 * FLAGS.max_training_steps)
    cooldown_start = int(1/2 * FLAGS.max_training_steps)
    cooldown_end = FLAGS.max_training_steps

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < cooldown_start:
            return 1.0
        else:
            progress = float(current_step - cooldown_start) / float(max(1, cooldown_end - cooldown_start))
            return 0.45 * (1.0 + math.cos(math.pi * progress)) + 0.1

    # Initialize optimizers
    scheduler = LambdaLR(optimizer, lr_lambda)
  
    # Cares the save
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=100,
        per_device_train_batch_size=FLAGS.batch_train,
        per_device_eval_batch_size=FLAGS.batch_eval,
        weight_decay=0,
        save_total_limit=3,
        max_steps=FLAGS.max_training_steps,
        predict_with_generate=True,
        report_to=["wandb"],
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, scheduler),
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    
    wandb.finish(0)

if __name__ == '__main__':
    app.run(main)
