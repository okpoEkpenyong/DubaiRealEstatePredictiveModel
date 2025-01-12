import os
import yaml
import pytest
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ""))  # This should point to the project root

config_path = os.path.join(current_dir, "../configs/config.yaml")  # Adjusted path for config.yaml


# Helper function to check if a path exists
def validate_paths(config, base_path_key="paths", files_key="files"):
    paths = config.get(base_path_key, {})
    files = config.get(files_key, {})

    errors = []
    print(f"DEBUG: Project root is {project_root}")  # Debug
    
    print(f"DEBUG: current dir is {current_dir}")  # Debug
    
    # Validate directory paths
    for key, path in paths.items():
        absolute_path = os.path.join(project_root, path)  # Resolve relative paths correctly
        print(f"DEBUG: Checking directory path {absolute_path} for key {key}")  # Debug
        if not os.path.isdir(absolute_path):
            errors.append(f"Directory does not exist: {absolute_path} (key: {key})")
    
    # Validate file paths
    for key, file in files.items():
        raw_data_path = os.path.join(project_root, paths["raw_data"])  # Resolve raw_data path correctly
        full_file_path = os.path.join(raw_data_path, file)
        print(f"DEBUG: Checking file path {full_file_path} for key {key}")  # Debug
        if not os.path.isfile(full_file_path):
            errors.append(f"File does not exist: {full_file_path} (key: {key})")

    return errors


# Test for config.yaml validity and paths
def test_config_yaml():
    print(f"DEBUG: Config file path is {config_path}")  # Debug
    
    # Check if the config file exists
    assert os.path.isfile(config_path), f"Config file not found at: {config_path}"

    # Load the YAML
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML file: {config_path}\nError: {e}")

    # Validate paths
    errors = validate_paths(config)
    if errors:
        pytest.fail("\n".join(errors))


# Test for file headers
def test_file_headers():
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    raw_data_path = os.path.join(project_root, config["paths"]["raw_data"])  # Resolve raw_data path correctly
    file_name = config["files"]["01_real_estate_rents"]
    file_path = os.path.join(raw_data_path, file_name)

    # Debug prints
    print(f"DEBUG: Raw data path is {raw_data_path}")
    print(f"DEBUG: File name is {file_name}")
    print(f"DEBUG: Full file path is {file_path}")

    # Ensure file exists before loading
    assert os.path.isfile(file_path), f"File not found: {file_path}"

    try:
        # Load the file and check headers
        df = pd.read_csv(file_path, delimiter=";",low_memory=False)
    except pd.errors.ParserError as e:
        pytest.fail(f"Error reading {file_path}: {e}")
        
    
    required_columns = ['Property Type','Property Sub Type',
                           'Nearest Mall', 'Nearest Landmark', 'Nearest Metro', 'Area', 'Usage']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
