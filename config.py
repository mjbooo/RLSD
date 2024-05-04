import os

from sacred import Experiment

from utils.util import get_short_name

ex = Experiment("METER", save_git_info=False)

@ex.config
def config():
    # Model config
    drf = "google-t5/t5-small" # drf model
    tgt = "google-t5/t5-base" # target model

    # IterativeTrainer config
    ckpt_load = None # "evaluation with pretrained checkpoint"
    is_sft = False # "enable SFT"
    num_itr = 3 # "number of DPO iterations"
    
    # DPOSDTrainer config
    policy = "IterativeDPO" 
    loss_type = "sigmoid" # Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
    beta = 0.1 # "beta for DPO"
    label_smoothing = 0.1 # "label smoothing for DPO"
    label_pad_token_id = -100 # label_pad_token_id  for the prompt tokens.
    train_match_first = False # "train only match-first part of drafts")
    reference_free = False # "reference-free for DPO: sanity check") 

    # draft generation config
    draft_from = "drf" # Literal["drf", "mix"]
    num_same_train_valid = 0 # train == valid & len(D)==num_same_train_valid
    multiply_draft_pair = 1 # getting multiple draft-pair from each prompt (# drafts = 2 * multiply_draft_pair)
    largest_gap_pair = 1 # getting draft-pair to the top-k largest reward gap outof multiple draft pair

    # Preference option
    preference = "first-total" # How to set the preference by class RewardSetter (casted by _config['preference'].split("-") )
    tie_option = "truncate" # "tie option for RewardSetter" ["random", "truncate", "drop"]
    
    # Iterative setting config
    data_gen = "iteration" # Literal["iteration", "batch"]
    max_length = 2048 # "max length > max_prompt_length + max_target_length")
    max_prompt_length = 256 # "max length of prompt x for drf_model.generate")
    max_target_length = 10 # "max length for the output y of drf_model.generate")
    temperature = 1.0 # "temperature for sampling during drf_model.generate")
    seed = 2024 # "setting seed"

    # Training config
    device = 0
    dataset = "cnn_dailymail" # "choosing dataset") # Todo: Dataset pvc, ckpt pvc
    lr = 5e-4 #, "learning rate"
    lr_scheduler = "fixed" # "learning rate scheduler" Literal["fixed", "cosine_warmup"]
    batch_prompt = 8 # "batch size"
    batch_train = 8 # "batch size"
    n_epochs = 2 # "The number of total epochs"
    patience = 20 # "patience for EarlyStoppingCallback"
    eval = False # "enable eval mode"
    debug = False # "enable debug mode (no wandb logging)"
    tiny_data = False # "use small data for debugging"
    phenomenon = False # "To reproduce the strange 'phenomenon'"

    # Logging config
    logging_steps = 0.01 # if the value <1, then it works as a ratio for a single epoch
    valid_steps = 0.05 # if the value <1, then it works as a ratio for a single epoch
    wandb_project_name = "ImprovedSD" # wandb project name
    custom_metrics = ["num_token_drf"] + [_metric + postfix for _metric in ["reward_first", "reward_total", "reward_exact"] for postfix in ["", "_ratio"]]

    # Path config
    root = "/pvc/home-mjlee" # root path 
    factors = [
        get_short_name(drf),
        get_short_name(tgt),
        get_short_name(dataset),
        f"{'SFT-' if is_sft else ''}{num_itr}DPO" if policy == "IterativeDPO" else f"{'SFT-' if is_sft else ''}{num_itr}{policy}",
        f"datagen-{data_gen}",
        f"mtl{max_target_length}",
        f"pf-{preference}",
        "tr-first" if train_match_first else "tr-total",
        # "scrat" if (eval & ~ckpt) else "pretr",
        f"mult-drf-{multiply_draft_pair}",
        f"num-same-tr-val-{num_same_train_valid}",
        f"temp{temperature}",
        seed,
    ]
    ckpt_save = "_".join(map(str, factors))
    output_dir = f"{root}/data/ImprovedSD/checkpoint/{ckpt_save}"

    
@ex.named_config
def batch_itr():
    raise NotImplementedError

@ex.named_config
def Placeholder():
    NotImplementedError