import datetime
import os
import random
from pathlib import Path, PosixPath

import numpy as np
import torch
from PIL import Image




def get_now_time():
    """Get the current time in YYYY-MM-DD_HH-MM-SS format.
    """
    
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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_for_differences_between_df(df1, df2, columns_to_exclude: list=None):
    """Checks for differences between two dataframes, returns the number of differences"""
    # Find the intersection of the columns
    intersecting_columns = list(df1.columns.intersection(df2.columns))

    print(f"Number of intersecting columns: {len(intersecting_columns)}")
    print(f"Checking for differences accross the following columns: {intersecting_columns}")

    try:
        # Remove columns_to_exclude from intersecting_columns
        if columns_to_exclude is not None:
            intersecting_columns = [column for column in intersecting_columns if column not in columns_to_exclude]
        
        # Compare the values in the intersecting columns
        # See here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html 
        differences = df1[intersecting_columns].compare(df2[intersecting_columns])
        return len(differences)
    except Exception as e:
        print(f"Error: {e}")
        print("Couldn't compare via pandas.DataFrame.compare, trying via lengths...")
        # Compare the lengths of the dataframes
        if len(df1) != len(df2):
            differences = abs(len(df1) - len(df2))
            print(f"Difference in dataframe lengths: {differences} (aboslute value of {len(df1)} - {len(df2)})")
            return differences

def sort_dict_by_values(dict_to_sort):
    sorted_dict = dict(sorted(dict_to_sort.items(), key=lambda x:x[1], reverse=True))
    return sorted_dict

def open_image(image_path_or_PIL):
    if isinstance(image_path_or_PIL, str) or isinstance(image_path_or_PIL, os.PathLike) or isinstance(image_path_or_PIL, PosixPath):
        return Image.open(image_path_or_PIL).convert("RGB")
    else:
        return image_path_or_PIL.convert("RGB")