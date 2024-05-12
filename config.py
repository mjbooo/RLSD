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

    # data generation
    data_gen = "batch" # "data generation method" 
    
    # Iterative setting config
    max_prompt_length = 2048 # len(x)
    max_target_length = 128 # N
    max_chunk_length = 5 # K

    temperature = 1.0 # "temperature for sampling during drf_model.generate")
    seed = 2024 # "setting seed"

    # Training config
    device = 0
    dataset = "cnn_dailymail" # "choosing dataset") # Todo: Dataset pvc, ckpt pvc
    lr = 5e-4 #, "learning rate"
    lr_scheduler = "fixed" # "learning rate scheduler" Literal["fixed", "cosine_warmup"]
    batch_train = 2 # "batch size"
    n_epochs = 3 # "The number of total epochs"
    eval = False # "enable eval mode"
    debug = False # enable debug mode (no wandb logging)
    tiny_data = False # use small data for debugging
    ckpt_load = None # load checkpoint model
    initial_valid = True # disable validation for step=0

    # Logging config
    custom_metrics = ['reward_exact']
    logging_steps = 0.01 # if the value <1, then it works as a ratio for a single epoch
    valid_steps = 0.2 # if the value <1, then it works as a ratio for a single epoch
    wandb_project_name = "RLSD" # wandb project name

    # Path config
    root = "/pvc/home-mjlee" # root path 
    factors = [
        ("Improved-"+policy if improved_reward else policy),
        get_short_name(drf),
        get_short_name(tgt),
        get_short_name(dataset),
        f"mtl-{max_target_length}",
        f"temp-{temperature}",
        f"lr-{lr}",
        seed,
    ]
    ckpt_save = "_".join(map(str, factors))
    output_dir = f"{root}/data/ImprovedSD/checkpoint/{ckpt_save}"

    
# DistillSpec
@ex.named_config
def DS():
    policy = "DistillSpec"
    wandb_project_name = "DistillSpec"

@ex.named_config
def RL():
    policy = "RL"

@ex.named_config
def Improved_RL():
    policy = "RL"
    improved_reward = True

@ex.named_config
def Debug():
    tgt = "google/t5-small-lm-adapt" # target model
    debug = True # enable debug mode (no wandb logging)
    tiny_data = True # use small data for debugging
    initial_valid = False # disable validation for step=0