def get_now_time():
    """Get the current time in YYYY-MM-DD_HH-MM-SS format.
    """
    import datetime
    now = datetime.datetime.now()

    # Get the current time in YYYY-MM-DD_HH-MM-SS format
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    return now

def seed_everything(seed: int):
    """Set seeds for almost everything.

    Source: Modified version of https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964

    Args:
        seed (int): manually set seed for reproducibility
    """
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)