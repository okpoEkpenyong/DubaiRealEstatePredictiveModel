import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.insert(1, os.getcwd())



def display_dataframe_info(df):
    # Print the shape of the dataframe
    print('Shape:', df.shape)
    
    # Print the column names with their data types
    dtypes_list = [f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    print('Dtypes:', dtypes_list)
    
    # Print the column names with the percentage of missing values
    missings_list = [f"{col}:{missing:.2f}%" for col, missing in zip(df.columns, df.isnull().mean()*100)]
    print('Missings (%):', missings_list)
    
    # Calculate total missing percentage
    total_missing_percentage = (df.isnull().sum().sum() / df.size) * 100
    print(f'Total missings (%): {total_missing_percentage:.2f}%')

    
    # Print total duplicates
    print('total duplicates:', df.duplicated(keep=False).sum())
    
    # Print the total memory consumed by the dataframe
    memory_usage = df.memory_usage().sum() / 1024**2  # Convert bytes to megabytes (MB)
    print(f"Total memory usage: {memory_usage:.2f} MB")
    
    # Display the first 2 rows of the dataframe
    return df.head(2)

def optimize_dataframe_memory(df):
    df = df.copy()
    print(f"Original memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")
    
    # Downcasting integers
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    
    # Downcasting floats
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    
    # Converting object columns to category if there are fewer unique values than 50% of total rows
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        num_unique_values = df[col].nunique()
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    
    print(f"Optimized memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")
    return df

def encode_cat_cols(df):
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    
    label_encoders = {}

    for col in non_numeric_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return label_encoders, df        


def format_and_clean_table(df, datetime_columns=None, numeric_columns=None, datetime_format="%d/%m/%Y %H:%M:%S", missing_threshold=0.9):
    """
    Format and clean a DataFrame for better display and usability.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        datetime_columns (list, optional): List of column names to convert to datetime. Default is None.
        numeric_columns (list, optional): List of column names to convert to numeric. Default is None.
        datetime_format (str, optional): Datetime format for conversion. Default is "%d/%m/%Y %H:%M:%S".
        missing_threshold (float, optional): Maximum allowed fraction of missing values per column. Default is 0.9.
    
    Returns:
        pd.DataFrame: The cleaned and formatted DataFrame.
    """
    try:
        df = df.copy()
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Drop rows with all values missing
        df.dropna(how="all", inplace=True)
        
        # Drop columns with >90% missing values
        threshold = len(df) * (1 - missing_threshold)
        df.dropna(axis=1, thresh=threshold, inplace=True)

        # Strip whitespace from string data
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert specified columns to datetime
        if datetime_columns:
            for col in datetime_columns:
                if col in df.columns:
                    try:
                        # Attempt conversion to datetime with error coercion
                        df[col] = pd.to_datetime(df[col], format=datetime_format, errors='coerce')
                        # Verify the conversion
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            print(f"Warning: {col} could not be fully converted to datetime. Check data for inconsistencies.")
                    except Exception as dt_err:
                        print(f"Error converting column {col} to datetime: {dt_err}")

                
        # Convert specified columns to numeric
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
        df.drop_duplicates(inplace=True)         
        
        # Display basic info
        print("Formatted and Cleaned Data Overview:")
        print(df.info())
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        raise
