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

def check_for_differences_between_df(df1, df2, columns_to_exclude: list=None):
    """Checks for differences between two dataframes, returns the number of differences"""
    # Find the intersection of the columns
    intersecting_columns = list(df1.columns.intersection(df2.columns))

    # Remove columns_to_exclude from intersecting_columns
    if columns_to_exclude is not None:
        intersecting_columns = [column for column in intersecting_columns if column not in columns_to_exclude]
    
    # Compare the values in the intersecting columns
    # See here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html 
    differences = df1[intersecting_columns].compare(df2[intersecting_columns])

    # Return the number of differences
    return len(differences)