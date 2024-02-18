from torch import optim
from torch.optim import lr_scheduler 

def get_LR_scheduler_from_name(Lr_scheduler_name: str):
    """Returns the LR Scheduler identified by string name"""
    return getattr(lr_scheduler, Lr_scheduler_name)

def get_optimizer_from_name(optimizer_name: str):
    """Returns the LR Scheduler identified by string name"""
    return getattr(optim, optimizer_name)
