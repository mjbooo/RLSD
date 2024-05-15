import os

from sacred import Experiment

from utils.util import get_short_name

ex = Experiment("METER", save_git_info=False)

@ex.config
def config():
    # Model config
    drf = "google/t5-small-lm-adapt" # drf model
    tgt = "google/t5-xl-lm-adapt" # target model
    
    # policy: DS, RL
    policy = "DistillSpec"
    improved_reward = False
    truncation_deg = None

    # data generation
    data_gen = "batch" # "data generation method" 
    
    # DistillSpec
    divergence = "kl" # "divergence for distillation" Literal["kl", "tvd"]

    # Iterative setting config
    max_prompt_length = 2048 # len(x)
    max_target_length = 128 # N
    max_chunk_length = 5 # K

    temperature = 1.0 # "temperature for sampling during drf_model.generate")
    seed = 2024 # "setting seed"

    # Optimizer config
    optimizer = "adafactor"
    lr_scheduler = "fixed" # "learning rate scheduler" Literal["fixed", "cosine_warmup"]
    lr = 3e-4 #, "learning rate"

    # Training config
    device = 0
    dataset = "xsum" # "choosing dataset") # Todo: Dataset pvc, ckpt pvc
    num_valid_tiny = 500 # " len(valid_tiny) for measuring block efficiency"
    batch_train = 2 # "batch size"
    n_epochs = 3 # "The number of total epochs"
    max_training_steps = None # "The number of total training steps". This will over ride the n_epochs
    eval = False # "enable eval mode"
    debug = False # enable debug mode (no wandb logging)
    simple_setup = False # use simple setup for massive experiments
    num_simple = 1000 # number of samples in train set for simple setup
    tiny_data = False # use small data for debugging
    initial_valid = True # disable validation for step=0

    # load
    ckpt_dir = None # load checkpoint model
    resume_training_steps = None # the number of steps numberwhich you resume training from.

    # Logging config
    custom_metrics = ['exact_reward', 'acceptance_ratio_alpha']
    logging_steps = 0.01 # if the value <1, then it works as a ratio for a single epoch
    valid_steps = 0.5 # if the value <1, then it works as a ratio for a single epoch
    wandb_project_name = "RLSD" # wandb project name

    # Path config
    root = "/pvc/home-mjlee" # root path 
    if 'RL' in policy:
        prefix = ''
        if improved_reward:
            prefix += "Improved-"
        if truncation_deg:
            prefix += f"Trun-{truncation_deg}-"

        model_ckpt = prefix + policy
    elif 'DistillSpec' in policy:
        model_ckpt = f"{policy}-{divergence}"

    factors = [
        model_ckpt,
        get_short_name(drf),
        get_short_name(tgt),
        get_short_name(dataset),
        f"mtl-{max_target_length}",
        f"temp-{temperature}",
        f"lr-{lr}",
        seed,
    ]
    if tiny_data:
        factors.append('tiny_data')
    ckpt_save = "_".join(map(str, factors))
    output_dir = f"{root}/data/ImprovedSD/checkpoint/{ckpt_save}"

    
@ex.named_config
def Simple():
    num_valid_tiny = 10
    simple_setup = True
    max_training_steps = 2000

# Policy
@ex.named_config
def DS():
    policy = "DistillSpec"

    wandb_project_name = "240513DistillSpec"
    
    max_training_steps = 300000 # "The number of total training steps". This will over ride the n_epochs
    batch_train=32
    optimizer = "adafactor"
    lr = 3e-4
    lr_scheduler = "linear_warmup_cosine_decay"

@ex.named_config
def RL():
    policy = "RL"

    wandb_project_name = "240513DistillSpec"
    
    max_training_steps = 300000 # "The number of total training steps". This will over ride the n_epochs
    batch_train=32
    optimizer = "adafactor"
    lr = 3e-4
    lr_scheduler = "linear_warmup_cosine_decay"

@ex.named_config
def Improved_RL():
    policy = "RL"
    improved_reward = True
    
    wandb_project_name = "240513DistillSpec"
    
    max_training_steps = 300000 # "The number of total training steps". This will over ride the n_epochs
    batch_train=32
    optimizer = "adafactor"
    lr = 3e-4
    lr_scheduler = "linear_warmup_cosine_decay"

@ex.named_config
def Truncated_RL():
    policy = "RL"
    truncation_deg = 3
    
    wandb_project_name = "240513_Truncation"

    batch_train=32
    optimizer = "adafactor"
    lr = 3e-4
    lr_scheduler = "linear_warmup_cosine_decay"

# Dataset
@ex.named_config
def Xsum():
    dataset = "xsum"
    max_prompt_length=1024
    max_target_length=64

# Debug
@ex.named_config
def DS_debug():
    policy = "DistillSpec"
    wandb_project_name = "240513DistillSpec"
    dataset = "xsum"
    optimizer = "adafactor"
    n_epochs = 3 # "The number of total epochs"
    valid_steps = 0.5 # if the value <1, then it works as a ratio for a single epoch
    max_training_steps = 30 # "The number of total training steps". This will over ride the n_epochs
    max_target_length=64
    batch_train=1
    lr = 3e-4
    lr_scheduler = "linear_warmup_cosine_decay"
    # tgt = "google/t5-small-lm-adapt" # target model
    # debug = True # enable debug mode (no wandb logging)
    tiny_data = True # use small data for debugging
    # initial_valid = False # disable validation for step=0
    

@ex.named_config
def Debug():
    tgt = "google/t5-small-lm-adapt" # target model
    debug = True # enable debug mode (no wandb logging)
    tiny_data = True # use small data for debugging
    initial_valid = False # disable validation for step=0