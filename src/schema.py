from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),

    "transformer_seed": merge(tinteger, nullable, default(None))
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

training_schema = {
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "n_points": merge(tinteger, nullable, default(None)),

    "task_kwargs": merge(tdict, nullable, default(None)),
    "data_kwargs": merge(tdict, nullable, default(None)),
    "prompt_kwargs": merge(tdict, nullable, default(None)),
    "val_kwargs": merge(tdict, nullable, default(None)),

    "weight_type": merge(tstring, allowed(["standard", "normalized"])),
    "prompt_type": merge(tstring, allowed(["standard", "model_one", "model_two", "model_three", "model_four"])),
    "val_type": merge(tstring, allowed(["standard", "current_tasks", "provided_tasks", "current_plus_provided_tasks"])),

    "data_split": merge(tstring, allowed(["steps_based",  "blocks_based"])),
    "data_schedule": merge(tstring, allowed(["sequential", "random", "mixed_sequential"])),
    "task_schedule": merge(tstring, allowed(["sequential", "random", "mixed_sequential"])),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
