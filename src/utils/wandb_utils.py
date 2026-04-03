"""Optional Weights & Biases logging utilities. All calls are no-ops when disabled."""

_wandb_enabled = False
_wandb = None


def init_wandb(
    enabled=False,
    project="microplastic-da",
    entity=None,
    name=None,
    config=None,
    notes=None,
):
    global _wandb_enabled, _wandb

    _wandb_enabled = enabled

    if not enabled:
        return config or {}

    try:
        import wandb

        _wandb = wandb

        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config or {},
            notes=notes,
        )
        return wandb.config
    except ImportError:
        print("Warning: wandb not installed, logging disabled")
        _wandb_enabled = False
        return config or {}


def log(metrics):
    if _wandb_enabled and _wandb is not None:
        _wandb.log(metrics)


def watch(model, log_freq=100):
    if _wandb_enabled and _wandb is not None:
        _wandb.watch(model, log_freq=log_freq)


def update_config(config_dict, allow_val_change=True):
    if _wandb_enabled and _wandb is not None:
        _wandb.config.update(config_dict, allow_val_change=allow_val_change)


def finish():
    if _wandb_enabled and _wandb is not None:
        _wandb.finish()


def is_enabled():
    return _wandb_enabled
