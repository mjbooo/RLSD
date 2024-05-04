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

    # Iterative setting config
    max_prompt_length = 2048 # len(x)
    max_target_length = 128 # N
    max_chunk_length = 10 # K

    temperature = 1.0 # "temperature for sampling during drf_model.generate")
    seed = 2024 # "setting seed"

    # Training config
    device = 0
    dataset = "cnn_dailymail" # "choosing dataset") # Todo: Dataset pvc, ckpt pvc
    lr = 5e-4 #, "learning rate"
    lr_scheduler = "fixed" # "learning rate scheduler" Literal["fixed", "cosine_warmup"]
    batch_train = 8 # "batch size"
    n_epochs = 2 # "The number of total epochs"
    eval = False # "enable eval mode"
    debug = False # "enable debug mode (no wandb logging)"
    tiny_data = False # "use small data for debugging"

    # Logging config
    # logging_steps = 0.01 # if the value <1, then it works as a ratio for a single epoch
    # valid_steps = 0.05 # if the value <1, then it works as a ratio for a single epoch
    wandb_project_name = "RLSD" # wandb project name

    # Path config
    root = "/pvc/home-mjlee" # root path 
    factors = [
        get_short_name(drf),
        get_short_name(tgt),
        get_short_name(dataset),
        f"mtl-{max_target_length}",
        f"temp-{temperature}",
        seed,
    ]
    ckpt_save = "_".join(map(str, factors))
    output_dir = f"{root}/data/ImprovedSD/checkpoint/{ckpt_save}"

    
@ex.named_config
def RL():
    raise NotImplementedError