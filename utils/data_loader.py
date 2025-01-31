import os
import pandas as pd
import yaml
from tqdm import tqdm  # Import tqdm for progress bar
import dask.dataframe as dd
from concurrent.futures import ProcessPoolExecutor

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../configs/config.yaml")  # Adjusted path for config.yaml

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

def load_dataset(filepath, file_type="csv", delimiter=","):
    """
    Load a dataset from a given file path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load the dataset based on the file type
    if file_type == "csv":
        return pd.read_csv(filepath, sep=delimiter, low_memory=False, engine='c')
    elif file_type == "excel":
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type!")    

def read_file(file_path, delimiter=";", skiprows=0):
    """
    Read a single CSV file. This function must be top-level for parallel execution.
    """
    return pd.read_csv(file_path, sep=delimiter, low_memory=False, skiprows=skiprows)

def load_in_parallel(file_paths, delimiter=";", skiprows=0):
    """
    Load multiple CSV files in parallel using ProcessPoolExecutor.
    """
    from functools import partial  # To pass additional arguments to `read_file`
    read_func = partial(read_file, delimiter=delimiter, skiprows=skiprows)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(read_func, file_paths), total=len(file_paths), desc="Loading files in parallel"))

    df_combined = pd.concat(results, ignore_index=True)
    return df_combined

def load_with_dask(filepath, delimiter=";", use_tqdm=False, dtype=None):
    """
    Load large CSV files using Dask with optional tqdm progress bar.
    """
    # Load CSV using Dask with the provided dtype
    ddf = dd.read_csv(filepath, sep=delimiter, dtype=dtype)

    if use_tqdm:
        total_rows = ddf.shape[0].compute()
        with tqdm(total=total_rows, desc="Loading with Dask") as pbar:
            ddf = ddf.compute()  # Trigger computation
            pbar.update(total_rows)
    else:
        ddf = ddf.compute()

    return ddf


def inspect_data(file_key, delimiter=";", show_progress=True, loading_method="normal", skiprows=0):
    """
    Load and inspect a dataset based on its key in the config file.
    Optionally show a progress bar with tqdm.
    Allows different loading methods: 'normal', 'dask', 'parallel'.
    """
    # Construct the file path using the config file paths
    filepath = os.path.join(config["paths"]["raw_data"], config["files"][file_key])
    print(f"Loading {file_key} from {filepath}...")

       # Display a progress bar over rows while inspecting
    if show_progress:
         # Load the dataset
        df = load_dataset(filepath, delimiter=delimiter)
        for _ in tqdm(df.iterrows(), desc=f"Processing {file_key} rows", total=df.shape[0], disable=not show_progress):
            pass  # Just iterating to show progress bar
        
    # Determine the loading method
    if loading_method == "normal":
        df = load_dataset(filepath, delimiter=delimiter)
    elif loading_method == "dask":
        df = load_with_dask(filepath, delimiter=delimiter, use_tqdm=show_progress)
    elif loading_method == "parallel":
        df = load_in_parallel([filepath], delimiter=delimiter, skiprows=skiprows)
    else:
        raise ValueError(f"Unsupported loading method: {loading_method}")
    
    df.columns = df.columns.str.strip()
    
    # Display basic information
    print(f"Shape of {file_key}: {df.shape}")
    print(f"Columns in {file_key}:\n{df.columns}")

    return df

