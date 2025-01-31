from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder, TargetEncoder
import yaml
warnings.filterwarnings(
    action='ignore'
)
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../configs/config.yaml")  # Adjusted path for config.yaml

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    

sys.path.insert(1, os.getcwd())


def dynamic_aggregation(group):
    mean_value = group.mean()
    threshold = mean_value * 2  # setting threshold to 2 times the mean value
    max_value = group.max()

    # If the maximum value exceeds the dynamic threshold, return the max value
    if max_value > threshold:
        return max_value
    return mean_value

def get_column_types(df):
    """
    Identifies and returns numerical and categorical columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - dict: A dictionary containing numerical and categorical column names.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols
    }

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

def le_encode_cat_cols(df):
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
        # print("Formatted and Cleaned Data Overview:")
        # print(df.info())
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

def remove_negative_values(df):
    """
    Remove rows with negative values in numerical columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with no negative values in numerical columns.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    return df[~(df[numerical_columns] < 0).any(axis=1)]

def preprocess_data(df, dataset_type="sales"):
    """
    Preprocess the data by dropping irrelevant columns and handling missing values.

    Parameters:
        df (pd.DataFrame): Input DataFrame to preprocess.
        dataset_type (str): Specify either 'sales' or 'rentals' to apply dataset-specific preprocessing.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    drop_features = ['Year', 'Month', 'Day', 'Hour', 'YearMonth', 'Ejari Contract Number']
    if dataset_type == "rentals":
        drop_features = ['Ejari Contract Number']
    
    df = df.drop(columns=drop_features, errors='ignore')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    if dataset_type == "sales":
        for col in ['Avg Price Last Month', 'Avg Price Last Week','Price per sq.m']:
            if col in df.columns:
                df[col] = imputer.fit_transform(df[[col]])
    
    return df

def scale_features(df, numerical_columns):
    """
    Scale numerical features while retaining unscaled values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        numerical_columns (list): List of numerical column names to scale.

    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    scaler = StandardScaler()
    # scaled_df = df.copy()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def feature_engineering(df, dataset_type="sales", outliers_info=None):
    """
    Add derived features and calculate proximity scores for modeling.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        dataset_type (str): Specify either 'sales' or 'rentals' to apply dataset-specific engineering.
        outliers_info (dict): Dictionary containing outlier thresholds for specific features.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Derived features
    if dataset_type == "sales":
        df['Area Premium'] = df['Sale Price'] / df.groupby('Area', observed=False)['Sale Price'].transform('mean')
        if outliers_info:
            df['Luxury Indicator'] = (
                (df['Sale Price'] > outliers_info['Sale Price']['upper_percentile']) |
                (df['Property Size (sq.m)'] > outliers_info['Property Size (sq.m)']['upper_percentile'])
            ).astype(int)
        df['Luxury Adjusted Price'] = df['Sale Price'] * df.get('Luxury Indicator', 1)
        df['Sales Price Log'] = np.log1p(df['Sale Price'])
    
    elif dataset_type == "rentals":
        df['Price per sq.m'] = df['Annual Rental Price'] / (df['Property Size (sq.m)'] + 1e-6)
        df['Freehold Indicator'] = df['Is Free Hold?'].apply(lambda x: 1 if x == "Free Hold" else 0)

    # Proximity Score
    df['Nearest Metro Encoded'] = df['Nearest Metro'].cat.codes if pd.api.types.is_categorical_dtype(df['Nearest Metro']) else df['Nearest Metro']
    df['Nearest Mall Encoded'] = df['Nearest Mall'].cat.codes if pd.api.types.is_categorical_dtype(df['Nearest Mall']) else df['Nearest Mall']
    df['Nearest Landmark Encoded'] = df['Nearest Landmark'].cat.codes if pd.api.types.is_categorical_dtype(df['Nearest Landmark']) else df['Nearest Landmark']

    df['Proximity Score'] = (
        df['Nearest Metro Encoded'] +
        df['Nearest Mall Encoded'] +
        df['Nearest Landmark Encoded']
    )
    
    return df

def oe_encode_categorical_features(df, categorical_columns):
    """
    Encode categorical features using ordinal encoding.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        categorical_columns (list): List of categorical columns to encode.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    encoder = OrdinalEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    return df

def remove_negative_values(df):
    """
    Remove rows with negative values in numerical columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with no negative values in numerical columns.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    return df[~(df[numerical_columns] < 0).any(axis=1)]

def pipeline(df, dataset_type="sales", outliers_info=None):
    """
    Complete pipeline to preprocess, engineer features, encode, and clean the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        dataset_type (str): Either 'sales' or 'rentals'.
        outliers_info (dict): Outlier thresholds for feature engineering (if available).

    Returns:
        pd.DataFrame: Fully processed and scaled DataFrame.
        pd.DataFrame: Fully processed but unscaled/unencoded DataFrame.
    """
    
    # Remove negative values
    df = remove_negative_values(df)
    
    # Preprocess data
    df = preprocess_data(df, dataset_type)
    
    # Feature engineering
    df = feature_engineering(df, dataset_type, outliers_info)
    
    df_copy = df.copy()
    # Identify categorical columns for encoding
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = 'Sale Price' if dataset_type == "sales" else 'Annual Rental Price'
    
    # Encode categorical features
    df = oe_encode_categorical_features(df, target_col, categorical_columns)
    
    # Scale numerical features
    numerical_columns = df.select_dtypes(include=['number']).columns
    scaled_df = scale_features(df, numerical_columns)
    
    return scaled_df, df_copy

def process_dates(transactions_df, date_col):
    """
    Processes the 'Transaction Date' column in transactions_df:
    - Converts to datetime format
    - Extracts additional date-related features
    
    Parameters:
        transactions_df (pd.DataFrame): Input dataframe containing 'Transaction Date'.
    
    Returns:
        pd.DataFrame: Updated dataframe with processed date features.
    """
    # Convert 'Transaction Date' to datetime
    transactions_df[date_col] = pd.to_datetime(
        transactions_df[date_col], format='%d/%m/%Y %H:%M', errors='coerce'
    )
    
    # Handle missing values (optional: drop rows or fill missing dates)
    missing_dates = transactions_df[date_col].isna().sum()
    if missing_dates > 0:
        print(f"Warning: {missing_dates} invalid date entries found. Replacing with NaT.")
        # Optional: Drop rows with invalid dates
        # transactions_df.dropna(subset=['Transaction Date'], inplace=True)
    

    # Extract useful components
    transactions_df['Year'] =  transactions_df[date_col].dt.year
    transactions_df['Month'] = transactions_df[date_col].dt.month
    transactions_df['Day'] =   transactions_df[date_col].dt.day
    transactions_df['Hour'] =  transactions_df[date_col].dt.hour
    
    # Extract Year-Week and Year-Month from Transaction Date
    # transactions_df['YearWeek'] = transactions_df['Transaction Date'].dt.strftime('%Y-%U')  # Year-Week format
    # transactions_df['YearMonth'] = transactions_df['Transaction Date'].dt.to_period('M').astype(str)  # Year-Month format


    return transactions_df

def compute_summary_statistics(df):
    """
    Display summary statistics for a given DataFrame.
    """
    
    num_features = df.select_dtypes(include="number").columns

    # Summary statistics
    print("\n--- Summary Statistics ---")
    summary = df[num_features].describe(include="all").transpose()
    print(summary.to_string())

    # Missing values
    print("\n--- Missing Value Statistics ---")
    missing_stats = (df.isnull().sum() / len(df) * 100).to_frame(name="Missing Percentage")
    print(missing_stats.to_string())

def identify_extremes(df, column):
    """
    Identifies the maximum and minimum values of the given column,
    and returns the corresponding rows with labels.
    """
    # Find max and min values
    max_value_row = df.loc[df[column].idxmax()]
    min_value_row = df.loc[df[column].idxmin()]

    # Return max and min with labels
    max_label = f"Max {column}: {max_value_row[column]:,.2f} (Year {max_value_row['Year']})"
    min_label = f"Min {column}: {min_value_row[column]:,.2f} (Year {min_value_row['Year']})"

    return {
        'max': (max_value_row['Year'], max_value_row[column], max_label),
        'min': (min_value_row['Year'], min_value_row[column], min_label)
    }

def detect_mixed_and_abnormal_types(df, threshold=0.01):
    """
    Detects mixed data types and abnormal values in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - threshold (float): Proportion of rows with abnormal types or values for flagging a column (default: 0.01).

    Returns:
    - pd.DataFrame: A summary of detected issues in the dataset.
    """
    results = []

    for col in df.columns:
        # Identify mixed types
        mixed_types = df[col].apply(type).nunique() > 1

        # Check for abnormal types (non-numeric in numeric columns, etc.)
        if pd.api.types.is_numeric_dtype(df[col]):
            abnormal_types = df[col].apply(lambda x: not isinstance(x, (int, float)) and x is not np.nan).sum()
        elif pd.api.types.is_string_dtype(df[col]):
            abnormal_types = df[col].apply(lambda x: not isinstance(x, str) and x is not np.nan).sum()
        else:
            abnormal_types = 0

        abnormal_types_percent = abnormal_types / len(df)

        # Detect abnormal values (extreme outliers)
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            abnormal_values = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            abnormal_values_percent = abnormal_values / len(df)
        else:
            abnormal_values = 0
            abnormal_values_percent = 0

        # Append results
        results.append({
            "Column": col,
            "Mixed Types": mixed_types,
            "Abnormal Types Count": abnormal_types,
            "Abnormal Types (%)": abnormal_types_percent * 100,
            "Abnormal Values Count": abnormal_values,
            "Abnormal Values (%)": abnormal_values_percent * 100,
        })

    # Create a summary DataFrame
    results_df = pd.DataFrame(results)
    flagged_columns = results_df[
        (results_df["Mixed Types"]) |
        (results_df["Abnormal Types (%)"] > threshold * 100) |
        (results_df["Abnormal Values (%)"] > threshold * 100)
    ]
    
    return results_df, flagged_columns

def missingness_pipeline(df, target_column="Value", impute_method="mean", test_size=0.2, random_state=42):
    """
    A complete pipeline to handle missingness analysis, modeling, correlation, and imputation.
    
    Parameters:
    - df: The input DataFrame with missing values.
    - target_column: The column to predict missingness (default is "Value").
    - impute_method: The method to use for imputing missing values ('mean', 'median', 'ffill', 'bfill').
    - test_size: The size of the test set for the logistic regression model (default is 0.2).
    - random_state: Random state for reproducibility (default is 42).
    
    Returns:
    - model: The trained logistic regression model for missingness prediction.
    - accuracy: Accuracy score of the model.
    - corr_matrix: The correlation matrix of the variables.
    - df_imputed: The DataFrame with imputed missing values.
    """
    
    # Step 1: Encode and prepare data for missingness modeling
    def prepare_missingness_data(df, target_column):
        df_encoded = pd.get_dummies(df.drop(columns=[target_column, "Missing"]), drop_first=True)
        df_encoded.fillna(df_encoded.mean(), inplace=True)  # Fill NaNs in numeric columns
        X = df_encoded
        y = df["Missing"]
        return X, y
    
    # Step 2: Train and evaluate logistic regression model for missingness prediction
    def train_missingness_model(X, y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = LogisticRegression(max_iter=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy
    
    # Step 3: Plot the correlation matrix
    def plot_missingness_correlation(df):
        corr_matrix = df.drop(columns=["Country Name", "Country Code", "Indicator Name", "Indicator Code"]).corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Missingness with Other Variables")
        plt.show()
        return corr_matrix
    
    # Step 4: Impute missing values based on the chosen method
    def impute_missing_values(df, method):
        df_imputed = df.copy()
        if method == "mean":
            df_imputed["Value"].fillna(df_imputed["Value"].mean(), inplace=True)
        elif method == "median":
            df_imputed["Value"].fillna(df_imputed["Value"].median(), inplace=True)
        elif method == "ffill":
            df_imputed["Value"].fillna(method="ffill", inplace=True)
        elif method == "bfill":
            df_imputed["Value"].fillna(method="bfill", inplace=True)
        else:
            raise ValueError("Invalid imputation method. Choose from 'mean', 'median', 'ffill', 'bfill'.")
        return df_imputed
    
    # Execute Steps
    X, y = prepare_missingness_data(df, target_column)
    model, accuracy = train_missingness_model(X, y, test_size, random_state)
    corr_matrix = plot_missingness_correlation(df)
    df_imputed = impute_missing_values(df, method=impute_method)
    
    # Return results
    return model, accuracy, corr_matrix, df_imputed


