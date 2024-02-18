from torch import optim
from torch.optim import lr_scheduler 

def get_LR_scheduler(name: str, kwargs : dict):
    """Returns the LR Scheduler identified by string name"""
    return getattr(lr_scheduler, name)(**kwargs)

def get_optimizer(name: str, kwargs: dict):
    """Returns the LR Scheduler identified by string name"""
    return getattr(optim, name)(**kwargs)
