

import os
import sys
from sklearn.linear_model import LinearRegression
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import ttest_rel, wilcoxon

from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import time
    
    
# from pyemd import EEMD
# from pyemd import emd
# from PyEMD import EMD
import emd


sys.path.insert(1, os.getcwd())



# Define the EEMD-SD-SVM Model
# from PyEMD import EEMD


class EEMD_SD_SVM:
    def __init__(self, n_imfs=5, C=1.0, epsilon=0.1, kernel='rbf'):
        self.n_imfs = n_imfs
        self.svm_models = [SVR(C=C, epsilon=epsilon, kernel=kernel) for _ in range(n_imfs)]
        self.emd = emd()

    def fit(self, X, y):
        y = y.ravel()  # Ensure y is a 1D array
        t = np.arange(len(y))  # Create time indices
        
        # Apply EEMD decomposition
        imfs = self.eemd.eemd(y, t, max_imf=self.n_imfs)

        # Train a separate SVM for each IMF
        for i, model in enumerate(self.svm_models):
            model.fit(X, imfs[i])
        return self

    def predict(self, X):
        # Predict using each SVM and sum the contributions
        predictions = np.sum([model.predict(X) for model in self.svm_models], axis=0)
        return predictions


# class EEMD_SD_SVM:
#     def __init__(self, n_imfs=5, C=1.0, epsilon=0.1, kernel='rbf'):
#         self.n_imfs = n_imfs
#         self.svm_models = [SVR(C=C, epsilon=epsilon, kernel=kernel) for _ in range(n_imfs)]
#         self.eemd = emd()

#     def fit(self, X, y):
#         # Apply EEMD decomposition
#         imfs = self.eemd.eemd(y, np.arange(len(y)), max_imf=self.n_imfs)
        
#         # Train a separate SVM for each IMF
#         for i, model in enumerate(self.svm_models):
#             model.fit(X, imfs[i])
#         return self

#     def predict(self, X):
#         # Predict using each SVM and sum the contributions
#         predictions = np.sum([model.predict(X) for model in self.svm_models], axis=0)
#         return predictions

# def train_model_pipelines_with_early_stopping(df, target, scale=True, shuffle=True, output_dir=None):
#     """
#     Train multiple models on the given dataset, including comparisons of models with and without early stopping.

#     Args:
#         df (pd.DataFrame): The input dataset.
#         target (str): Target column name.
#         scale (bool): Whether to scale features and target variable.
#         shuffle (bool): Whether to shuffle the data before splitting.
#         output_dir (str): Directory to save model checkpoints.

#     Returns:
#         Tuple: Training, validation, and testing scores, and staged predictions for visualization.
#     """
#     from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
#     from xgboost import XGBRegressor
#     from lightgbm import LGBMRegressor

#     # Scalers for features and target
#     x_scaler = MinMaxScaler()
#     y_scaler = MinMaxScaler()

#     predictors = df.drop(target, axis=1)

#     # Scale features and target variable
#     if scale:
#         predictors = pd.DataFrame(x_scaler.fit_transform(predictors), columns=predictors.columns)
#         target = pd.DataFrame(y_scaler.fit_transform(df[[target]]), columns=[target])
#     else:
#         predictors = pd.DataFrame(predictors, columns=predictors.columns)
#         target = pd.DataFrame(df[[target]], columns=[target])

#     # Split into train, validation, and test sets
#     train_size = int(len(predictors) * 0.7)
#     val_size = int(len(predictors) * 0.15)

#     if shuffle:
#         x_train, x_temp, y_train, y_temp = train_test_split(predictors, target, test_size=0.3, random_state=42)
#         x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
#     else:
#         x_train, x_temp = predictors[:train_size], predictors[train_size:]
#         y_train, y_temp = target[:train_size], target[train_size:]
#         val_end = train_size + val_size
#         x_val, x_test = predictors[train_size:val_end], predictors[val_end:]
#         y_val, y_test = target[train_size:val_end], target[val_end:]

#     # Models with and without early stopping
#     models = [
#         ("Gradient Boosting (Full)", GradientBoostingRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42)),
#         ("Gradient Boosting (Early Stopping)", GradientBoostingRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4)),
#         ("HistGradient Boosting (Full)", HistGradientBoostingRegressor(max_iter=1000, random_state=42)),
#         ("HistGradient Boosting (Early Stopping)", HistGradientBoostingRegressor(max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1, tol=1e-4)),
#         ("XGBoost (Full)", XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)),
#         ("XGBoost (Early Stopping)", XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0, early_stopping_rounds=10, eval_set=[(x_val, y_val)])),
#         ("LightGBM (Full)", LGBMRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42)),
#         ("LightGBM (Early Stopping)", LGBMRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42, early_stopping_round=10, eval_set=[(x_val, y_val)])),
#     ]
#     metrics = ['mean_absolute_error', 'r2_score', 'mean_squared_error']

#     # Initialize dictionaries for scores and predictions
#     train_scores, val_scores, test_scores = {}, {}, {}
#     staged_predictions = {"train": {}, "val": {}}

#     train_df = y_train.copy()
#     val_df = y_val.copy()
#     test_df = y_test.copy()

#     # Train and evaluate models
#     for model_name, model in tqdm(models, total=len(models), desc="Evaluating Models"):
#         print(f"\nTraining {model_name}...")
#         model_output_dir = os.path.join(output_dir, model_name)
#         os.makedirs(model_output_dir, exist_ok=True)

#         # Train the model
#         model.fit(x_train, y_train)

#         # Save predictions
#         train_df[model_name] = model.predict(x_train).flatten()
#         val_df[model_name] = model.predict(x_val).flatten()
#         test_df[model_name] = model.predict(x_test).flatten()

#         # Save staged predictions for visualization (if supported)
#         if hasattr(model, "staged_predict"):
#             staged_predictions["train"][model_name] = list(model.staged_predict(x_train))
#             staged_predictions["val"][model_name] = list(model.staged_predict(x_val))

#         # Save the trained model
#         model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
#         joblib.dump(model, model_path)

#         # Calculate scores
#         train_scores[model_name] = []
#         val_scores[model_name] = []
#         test_scores[model_name] = []

#         for metric in metrics:
#             train_scores[model_name].append(eval(metric)(train_df[model_name], y_train))
#             val_scores[model_name].append(eval(metric)(val_df[model_name], y_val))
#             test_scores[model_name].append(eval(metric)(test_df[model_name], y_test))

#     # Convert scores to DataFrames
#     train_scores_df = pd.DataFrame(train_scores, index=metrics).T
#     val_scores_df = pd.DataFrame(val_scores, index=metrics).T
#     test_scores_df = pd.DataFrame(test_scores, index=metrics).T

#     return train_scores_df, val_scores_df, test_scores_df, staged_predictions

def train_model_pipelines_with_early_stopping(
    df, models, target, scale=True, category=None, shuffle=True, output_dir=None
):
    """
    Train multiple models on the given dataset, including comparisons of models with and without early stopping.

    Args:
        df (pd.DataFrame): The input dataset.
        models (list): List of tuples with model names and model instances.
        target (str): Target column name.
        scale (bool): Whether to scale features and target variable.
        category (str): Category of models (used for tqdm description).
        shuffle (bool): Whether to shuffle the data before splitting.
        output_dir (str): Directory to save model checkpoints.

    Returns:
        dict: A dictionary containing training/validation/testing scores, staged predictions,
              training times, and number of estimators for each model.
    """


    # Scalers for features and target
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    predictors = df.drop(target, axis=1)

    # Scale features and target variable
    if scale:
        predictors = pd.DataFrame(x_scaler.fit_transform(predictors), columns=predictors.columns)
        target = pd.DataFrame(y_scaler.fit_transform(df[[target]]), columns=[target])
    else:
        predictors = pd.DataFrame(predictors, columns=predictors.columns)
        target = pd.DataFrame(df[[target]], columns=[target])

    # Split into train, validation, and test sets
    train_size = int(len(predictors) * 0.7)
    val_size = int(len(predictors) * 0.15)

    if shuffle:
        x_train, x_temp, y_train, y_temp = train_test_split(predictors, target, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    else:
        x_train, x_temp = predictors[:train_size], predictors[train_size:]
        y_train, y_temp = target[:train_size], target[train_size:]
        val_end = train_size + val_size
        x_val, x_test = predictors[train_size:val_end], predictors[val_end:]
        y_val, y_test = target[train_size:val_end], target[val_end:]
    
    metrics = ['mean_absolute_error', 'r2_score', 'mean_squared_error']

    # Initialize dictionaries for scores, predictions, training times, and estimators
    train_scores, val_scores, test_scores = {}, {}, {}
    staged_predictions = {"train": {}, "val": {}}
    training_times = {}
    n_estimators = {}

    train_df = y_train.copy()
    val_df = y_val.copy()
    test_df = y_test.copy()

    # Train and evaluate models
    for model_name, model in tqdm(models, total=len(models), desc=f"Evaluating {category} Models"):
        if isinstance(model, (XGBRegressor, LGBMRegressor)):
            model.set_params(eval_set=[(x_val, y_val)])  # Set validation data for early stopping
        print(f"\nTraining {model_name}...")
        model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
        if model_output_dir:
            os.makedirs(model_output_dir, exist_ok=True)

        # Measure training time
        start_time = time.time()
        model.fit(x_train, y_train)
        training_time = time.time() - start_time
        training_times[model_name] = training_time

        # Track the number of estimators (if applicable)
        if hasattr(model, "n_estimators_"):
            n_estimators[model_name] = model.n_estimators_
        else:
            n_estimators[model_name] = len(list(getattr(model, "staged_predict", lambda x: [])(x_train)))

        # Save predictions
        train_df[model_name] = model.predict(x_train).flatten()
        val_df[model_name] = model.predict(x_val).flatten()
        test_df[model_name] = model.predict(x_test).flatten()
        
        if hasattr(model, "staged_predict"):
            staged_predictions["train"][model_name] = list(model.staged_predict(x_train))
            staged_predictions["val"][model_name] = list(model.staged_predict(x_val))
            n_estimators[model_name] = len(staged_predictions["train"][model_name])
        else:
            n_estimators[model_name] = len(list(getattr(model, "staged_predict", lambda x: [])(x_train)))


        # Save the trained model
        if model_output_dir:
            model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)

        # Calculate scores
        train_scores[model_name] = []
        val_scores[model_name] = []
        test_scores[model_name] = []

        for metric in metrics:
            train_scores[model_name].append(eval(metric)(train_df[model_name], y_train))
            val_scores[model_name].append(eval(metric)(val_df[model_name], y_val))
            test_scores[model_name].append(eval(metric)(test_df[model_name], y_test))

    # Convert scores to DataFrames
    train_scores_df = pd.DataFrame(train_scores, index=metrics).T
    val_scores_df = pd.DataFrame(val_scores, index=metrics).T
    test_scores_df = pd.DataFrame(test_scores, index=metrics).T
    
    # Collect reusable return parameters into a dictionary
    results = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "predictors": predictors,
        "train_scores_df": train_scores_df,
        "val_scores_df": val_scores_df,
        "test_scores_df": test_scores_df,
        "staged_predictions": staged_predictions,
        "training_times": training_times,
        "n_estimators": n_estimators,
    }

    return results


# def train_model_pipelines_with_early_stopping(
#     df, models, target, scale=True, category=None, shuffle=True, output_dir=None
# ):
#     """
#     Train multiple models on the given dataset, including comparisons of models with and without early stopping.

#     Args:
#         df (pd.DataFrame): The input dataset.
#         models (list): List of tuples with model names and model instances.
#         target (str): Target column name.
#         scale (bool): Whether to scale features and target variable.
#         category (str): Category of models (used for tqdm description).
#         shuffle (bool): Whether to shuffle the data before splitting.
#         output_dir (str): Directory to save model checkpoints.

#     Returns:
#         dict: A dictionary containing training/validation/testing scores, staged predictions,
#               training times, and number of estimators for each model.
#     """

#     # Initialize scalers
#     x_scaler = MinMaxScaler()
#     y_scaler = MinMaxScaler()

#     predictors = df.drop(target, axis=1)

#     # Scale features and target variable
#     if scale:
#         predictors = pd.DataFrame(x_scaler.fit_transform(predictors), columns=predictors.columns)
#         target = pd.DataFrame(y_scaler.fit_transform(df[[target]]), columns=[target])
#     else:
#         predictors = pd.DataFrame(predictors, columns=predictors.columns)
#         target = pd.DataFrame(df[[target]], columns=[target])

#     # Split into train, validation, and test sets
#     train_size = int(len(predictors) * 0.7)
#     val_size = int(len(predictors) * 0.15)

#     if shuffle:
#         x_train, x_temp, y_train, y_temp = train_test_split(predictors, target, test_size=0.3, random_state=42)
#         x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
#     else:
#         x_train, x_temp = predictors[:train_size], predictors[train_size:]
#         y_train, y_temp = target[:train_size], target[train_size:]
#         val_end = train_size + val_size
#         x_val, x_test = predictors[train_size:val_end], predictors[val_end:]
#         y_val, y_test = target[train_size:val_end], target[val_end:]
    
#     # ✅ Fix: Convert y values to 1D NumPy arrays to avoid DataConversionWarning
#     y_train = y_train.values.ravel()
#     y_val = y_val.values.ravel()
#     y_test = y_test.values.ravel()

#     metrics = ['mean_absolute_error', 'r2_score', 'mean_squared_error']

#     # Initialize dictionaries for results
#     train_scores, val_scores, test_scores = {}, {}, {}
#     staged_predictions = {"train": {}, "val": {}}
#     training_times = {}
#     n_estimators = {}

#     train_df = pd.DataFrame(y_train, columns=[target])
#     val_df = pd.DataFrame(y_val, columns=[target])
#     test_df = pd.DataFrame(y_test, columns=[target])

#     # Train and evaluate models
#     for model_name, model in tqdm(models, total=len(models), desc=f"Evaluating {category} Models"):
#         if isinstance(model, (XGBRegressor, LGBMRegressor)):
#             model.set_params(eval_set=[(x_val, y_val)])  # Set validation data for early stopping

#         print(f"\nTraining {model_name}...")
#         model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
#         if model_output_dir:
#             os.makedirs(model_output_dir, exist_ok=True)

#         # Measure training time
#         start_time = time.time()
#         model.fit(x_train, y_train)  # ✅ y_train is now a 1D array
#         training_time = time.time() - start_time
#         training_times[model_name] = training_time

#         # Save predictions
#         train_df[model_name] = model.predict(x_train).flatten()
#         val_df[model_name] = model.predict(x_val).flatten()
#         test_df[model_name] = model.predict(x_test).flatten()
        
#         # Handle models with staged prediction
#         if hasattr(model, "staged_predict"):
#             staged_predictions["train"][model_name] = list(model.staged_predict(x_train))
#             staged_predictions["val"][model_name] = list(model.staged_predict(x_val))
#             n_estimators[model_name] = len(staged_predictions["train"][model_name])
#         else:
#             n_estimators[model_name] = len(list(getattr(model, "staged_predict", lambda x: [])(x_train)))

#         # Save the trained model
#         if model_output_dir:
#             model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
#             joblib.dump(model, model_path)

#         # Calculate scores
#         train_scores[model_name] = []
#         val_scores[model_name] = []
#         test_scores[model_name] = []

#         for metric in metrics:
#             train_scores[model_name].append(eval(metric)(train_df[model_name], y_train))
#             val_scores[model_name].append(eval(metric)(val_df[model_name], y_val))
#             test_scores[model_name].append(eval(metric)(test_df[model_name], y_test))

#     # Convert scores to DataFrames
#     train_scores_df = pd.DataFrame(train_scores, index=metrics).T
#     val_scores_df = pd.DataFrame(val_scores, index=metrics).T
#     test_scores_df = pd.DataFrame(test_scores, index=metrics).T
    
#     # Return all results
#     results = {
#         "x_train": x_train,
#         "x_val": x_val,
#         "x_test": x_test,
#         "y_train": y_train,
#         "y_val": y_val,
#         "y_test": y_test,
#         "predictors": predictors,
#         "train_scores_df": train_scores_df,
#         "val_scores_df": val_scores_df,
#         "test_scores_df": test_scores_df,
#         "staged_predictions": staged_predictions,
#         "training_times": training_times,
#         "n_estimators": n_estimators,
#     }

#     return results


def train_model_pipelines(df, target, scale=True, shuffle=True,output_dir=None):

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    predictors = df.drop(target, axis=1)

    # Scale features and target variable
    if scale:
        predictors = pd.DataFrame(x_scaler.fit_transform(predictors), columns=predictors.columns)
        target = pd.DataFrame(y_scaler.fit_transform(df[[target]]), columns=[target])
    else:
        predictors = pd.DataFrame(predictors, columns=predictors.columns)
        target = pd.DataFrame(df[[target]], columns=[target])

    train_size = int(len(predictors) * 0.7)  # 70% for training
    val_size = int(len(predictors) * 0.15)   # 15% for validation

    # Shuffle and split into train, validation, and test sets
    if shuffle:
        x_train, x_temp, y_train, y_temp = train_test_split(predictors, target, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    else:
        x_train, x_temp = predictors[:train_size], predictors[train_size:]
        y_train, y_temp = target[:train_size], target[train_size:]
        val_end = train_size + val_size
        x_val, x_test = predictors[train_size:val_end], predictors[val_end:]
        y_val, y_test = target[train_size:val_end], target[val_end:]

    # Models to evaluate
    models = [
        LinearRegression(),
        DecisionTreeRegressor(random_state=42),
        # HistGradientBoostingRegressor(random_state=42)
        # RandomForestRegressor(random_state=42),
        # ExtraTreesRegressor(random_state=42),
    ]
    models_names = [ 'Linear Model','Decision Tree']
    # models_names = ['Linear Model', 'Decision Tree', 'Histogram-based Gradient','Random Forest', 'Extra Trees']

    metrics = ['mean_absolute_error', 'r2_score', 'mean_squared_error']

    # Initialize dictionaries to store scores and predictions
    train_scores = {}
    val_scores = {}
    test_scores = {}

    # Copy dataframes to store predictions
    train_df = y_train.copy()
    val_df = y_val.copy()
    test_df = y_test.copy()

    # Evaluate models with tqdm progress bar
    for model, model_name in tqdm(zip(models, models_names), total=len(models), desc="Evaluating Models"):
    # for model, model_name in zip(models, models_names):
        if model_name not in models_names:
            print(f"Model {model_name} not found. Skipping...")
            continue
        print(f"\nTraining {model_name} model...")
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        model.fit(x_train, y_train)
        
        train_df[model_name] = model.predict(x_train).flatten()
        val_df[model_name] = model.predict(x_val).flatten()
        test_df[model_name] = model.predict(x_test).flatten()

        # Save the trained model pipeline
        model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        # joblib.dump(pipeline, model_path)

        train_scores[model_name] = []
        val_scores[model_name] = []
        test_scores[model_name] = []

        for metric in metrics:
            train_scores[model_name].append(eval(metric)(train_df[model_name], y_train))
            val_scores[model_name].append(eval(metric)(val_df[model_name], y_val))
            test_scores[model_name].append(eval(metric)(test_df[model_name], y_test))

    # Sort dataframes by index for consistency
    train_df.sort_index(inplace=True)
    val_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)

    # Convert scores to DataFrames
    train_scores_df = pd.DataFrame(train_scores, index=metrics).T
    val_scores_df = pd.DataFrame(val_scores, index=metrics).T
    test_scores_df = pd.DataFrame(test_scores, index=metrics).T

    # Inverse transform predictions if scaling is applied
    if scale:
        for c in train_df:
            train_df[c] = y_scaler.inverse_transform(train_df[[c]]).flatten()
        for c in val_df:
            val_df[c] = y_scaler.inverse_transform(val_df[[c]]).flatten()
        for c in test_df:
            test_df[c] = y_scaler.inverse_transform(test_df[[c]]).flatten()
            

    return train_scores_df, val_scores_df, test_scores_df

def train_multiple_models(X_train, X_val, y_train, y_val, categorical_features, numerical_features, models, output_dir, n_estimators=100):
    """
    Trains multiple models with progress bar and saves the model pipelines.
    """
    # Define the preprocessing pipeline for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = Pipeline(steps=[
        ('num', numerical_transformer),
        ('cat', categorical_transformer)
    ])
    
    # Check for infinity values in numerical columns
    # inf_cols_train = numerical_features[np.isinf(X_train[numerical_features]).any()]
    # inf_cols_val = numerical_features[np.isinf(X_val[numerical_features]).any()]
    
    metrics = []
    feature_importance = {}
    trained_models = {}


    # Ordinal Encoding for categorical features
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded = encoder.fit_transform(X_train[categorical_features])
    X_val_encoded = encoder.transform(X_val[categorical_features])

    # Create the output directory for saving models
    os.makedirs(output_dir, exist_ok=True)
    
    # fig, axs = plt.subplots(len(models), 3, figsize=(18, 6 * len(models)))
    # plt.subplots_adjust(wspace=0.4)

    # Loop through each model and train with progress tracking
    for model_name, model in models.items():
        if model_name not in models:
            print(f"Model {model_name} not found. Skipping...")
            continue
        print(f"\nTraining {model_name} model...")
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Initialize the model and pipeline with preprocessing
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Apply n_estimators only for tree-based and ensemble models
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "AdaBoost", "Extra Trees"]:
            model.set_params(n_estimators=n_estimators)

        # Progress bar for model training
        progress_bar = tqdm(range(1, n_estimators + 1), desc=f"Training {model_name}", unit="tree")

        for i in progress_bar:
            if model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "AdaBoost", "Extra Trees"]:
                model.set_params(n_estimators=i)  # Incrementally add trees
            pipeline.fit(X_train_encoded, y_train)
            
            trained_models[model_name] = model

        # Predict and evaluate the model
        y_pred = pipeline.predict(X_val_encoded)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Save the trained model pipeline
        model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
        joblib.dump(pipeline, model_path)

        print(f"\n{model_name} Model - RMSE: {rmse:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")
        print(f"Model saved to {model_path}")
        
        # Get feature importances if model is tree-based
        if hasattr(model, 'feature_importances_'):
            feature_importance[model_name] = model.feature_importances_
            importances = model.feature_importances_
            sorted_indices = np.argsort(importances)
            
            # Print feature importances for debugging or reference
            print(f"\nFeature Importances for {model_name}:") 
            print(feature_importance)
            
            
            # Handle plotting based on whether axs is 1D or 2D
            # if len(axs.shape) == 1:  # axs is a 1D array
            #     ax = axs[i]  # Use the i-th axis
            # else:  # axs is a 2D array
            #     ax = axs[i, 1]  # Use the correct subplot

            # x_col = importances[sorted_indices]
            # y_col = np.array(X)[sorted_indices]
            # sns.barplot(x=x_col,y=y_col,palette="viridis",hue=y_col, legend=False, ax=ax)
            # ax.set_title(f"Feature Importance - {model_name}")
            # ax.set_xlabel("Importance")
        else:
            feature_importance = None

        metrics.append({
                'Model': model_name,
                # "Feature Importance": feature_importances,
                'rmse':rmse, # Sensitive to large errors.
                'mse':mse,
                'r2': r2, # Model explanation power
                # mape  #  Normalized to actual values.
        })
        print("-"*100)    
        
    # Metrics comparison
    metrics_df = pd.DataFrame(metrics)
    plt.figure(figsize=(14, 4))
    sns.barplot(data=metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Metric', y='Score', hue='Model')
    plt.title("Model Performance Metrics")
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()
        
    return metrics_df, trained_models, y_pred, feature_importance, X_train_encoded, X_val_encoded

def prepare_data(model_features_df, target, categorical_features, numerical_features, test_size=0.2, val_size=0.25):
    """
    Prepares and splits the data for both sale price and rental price prediction.
    Returns the training and validation splits for both targets.
    """
    
    # Split data into features (X) and targets (y)
    X = model_features_df.drop(columns=[target])  # X is the features
    
    y = model_features_df[target]
    # y_rental = model_features_df[target]
        
    
    # Split into train, validation, and test sets for Sale Price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    
    # Split into train, validation, and test sets for Rental Price
    # X_train_rental, X_test_rental, y_train_rental, y_test_rental = train_test_split(X, y_rental, test_size=test_size, random_state=42)
    # X_train_rental, X_val_rental, y_train_rental, y_val_rental = train_test_split(X_train_rental, y_train_rental, test_size=val_size, random_state=42)

    
    # Return all splits and processed data
    return X, X_train, X_val, X_test, y_train, y_val, y_test

def get_feature_importances(model, feature_names, target_name):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    return importances


def tune_models(models, param_grids, X_train, y_train, search_type="grid", cv=5, scoring="neg_mean_squared_error", n_iter=10, random_state=42):
    """
    Tune models using Grid Search or Randomized Search with cross-validation.
    
    Args:
        models (dict): Dictionary of model names and their instances.
        param_grids (dict): Dictionary of hyperparameter grids for each model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        search_type (str): "grid" for GridSearchCV, "random" for RandomizedSearchCV.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric for evaluation.
        n_iter (int): Number of parameter settings sampled (only for RandomizedSearchCV).
        random_state (int): Random state for reproducibility.
    
    Returns:
        dict: Best models, their parameters, and cross-validation scores.
    """
    best_models = {}
    results = []
    
    for model_name, model in models.items():
        print(f"Tuning {model_name}...")
        param_grid = param_grids.get(model_name, {})
        
        if search_type == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        elif search_type == "random":
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state)
        else:
            raise ValueError("search_type must be 'grid' or 'random'")
        
        search.fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        
        # Store results
        results.append({
            "Model": model_name,
            "Best Params": search.best_params_,
            "Best Score": search.best_score_
        })
    
    results_df = pd.DataFrame(results)
    return best_models, results_df


def compare_models(models, X_test, y_test):
    """
    Perform statistical comparisons between models.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target values.
    
    Returns:
        pd.DataFrame: Statistical comparison of model performance.
    """
    scores = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        scores[model_name] = {
            "MSE": np.mean((y_test - y_pred) ** 2),
            "MAE": np.mean(np.abs(y_test - y_pred)),
            "R2": model.score(X_test, y_test)
        }
    
    scores_df = pd.DataFrame(scores).T
    
    # Pairwise statistical tests
    model_names = list(models.keys())
    p_values = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            
            mse1, mse2 = scores[m1]["MSE"], scores[m2]["MSE"]
            
            # Perform statistical tests (t-test & Wilcoxon signed-rank test)
            t_stat, t_p = ttest_rel([mse1], [mse2])
            w_stat, w_p = wilcoxon([mse1], [mse2])
            
            p_values.append({
                "Model 1": m1,
                "Model 2": m2,
                "T-test p-value": t_p,
                "Wilcoxon p-value": w_p
            })
    
    p_values_df = pd.DataFrame(p_values)
    
    return scores_df, p_values_df


def train_and_tune_models(
    X_train, X_val, y_train, y_val, categorical_features, numerical_features, models, tuning="grid",model_group=None, scoring=None, early_stopping=True, cv=3, output_dir="../models/checkpoints/tuned"
):
    """
    Trains multiple models with hyperparameter tuning (Grid Search, Randomized Search, or Bayesian) and optional early stopping.

    Args:
        X_train (DataFrame): Training features.
        X_val (DataFrame): Validation features.
        y_train (Series): Training target.
        y_val (Series): Validation target.
        categorical_features (list): Categorical feature names.
        numerical_features (list): Numerical feature names.
        models (dict): Dictionary of models with default hyperparameters.
        tuning (str): 'grid', 'random', or 'bayesian' for hyperparameter tuning.
        early_stopping (bool): Whether to apply early stopping for applicable models.
        output_dir (str): Directory to save trained models.

    Returns:
        metrics_df (DataFrame): Model evaluation metrics.
        tuned_models (dict): Dictionary of trained models with best hyperparameters.
    """

    
    
    os.makedirs(output_dir, exist_ok=True)

    # Metrics storage
    metrics = []
    tuned_models = {}

    # Tuning configurations for each model
 
    random_param_grids = {
    "LightGBM": {
        "regressor__n_estimators": randint(20, 100),  # Fewer estimators
        "regressor__learning_rate": uniform(0.05, 0.1),  # Narrow learning rate
        "regressor__max_depth": randint(3, 10),  # Shallower trees
        "regressor__num_leaves": randint(15, 31),  # Simplify trees
    },
    "RandomForest": {
        "regressor__n_estimators": randint(20, 100),  # Fewer estimators
        "regressor__max_depth": randint(3, 10),  # Limit tree depth
        "regressor__min_samples_split": randint(5, 15),  # Fewer splits
        "regressor__min_samples_leaf": randint(5, 10),
    },
    "XGBoost": {
        "regressor__n_estimators": randint(20, 100),  # Fewer estimators
        "regressor__learning_rate": uniform(0.05, 0.1),  # Narrow range
        "regressor__max_depth": randint(3, 6),  # Shallower trees
        "regressor__gamma": uniform(0, 0.1),  # Regularization
        "regressor__subsample": uniform(0.7, 0.3),  # Smaller subsample
    },
    "GradientBoosting": {
        "regressor__n_estimators": randint(20, 100),  # Fewer estimators
        "regressor__learning_rate": uniform(0.05, 0.1),  # Narrow range
        "regressor__max_depth": randint(3, 6),  # Shallower trees
        "regressor__subsample": uniform(0.7, 0.3),  # Smaller subsample
    },
    "AdaBoost": {
        "regressor__n_estimators": randint(20, 50),  # Fewer iterations
        "regressor__learning_rate": uniform(0.05, 0.1),  # Narrow range
        "regressor__loss": ["linear", "square"],  # Fewer options
    },
    "SVR": {
        "regressor__C": uniform(0.1, 5),  # Smaller range
        "regressor__epsilon": uniform(0.05, 0.2),  # Narrow range
        "regressor__kernel": ["linear", "rbf"],  # Simpler kernels
    },
    "KNN": {
        "regressor__n_neighbors": randint(3, 10),  # Smaller range
        "regressor__weights": ["uniform", "distance"],
        "regressor__metric": ["minkowski", "euclidean"],
    },
    "LinearRegression": {
        "regressor__fit_intercept": [True, False],
        "regressor__normalize": [True, False],
    },
    "ExtraTrees": {
        "regressor__n_estimators": randint(20, 100),  # Fewer estimators
        "regressor__max_depth": randint(3, 10),  # Shallower trees
        "regressor__min_samples_split": randint(5, 15),
        "regressor__min_samples_leaf": randint(5, 10),
    },
    "DecisionTree": {
        "regressor__max_depth": randint(3, 10),  # Shallower trees
        "regressor__min_samples_split": randint(5, 15),
        "regressor__min_samples_leaf": randint(5, 10),
    },
}
    
    grid_param_grids = {
    "LightGBM": {
        "regressor__n_estimators": [20, 50, 100],  # Fewer estimators
        "regressor__learning_rate": [0.05, 0.1],  # Narrow range
        "regressor__max_depth": [3, 5, 10],  # Shallower trees
        "regressor__num_leaves": [15, 31],
    },
    "RandomForest": {
        "regressor__n_estimators": [20, 50, 100],  # Fewer estimators
        "regressor__max_depth": [3, 5, 10],
        "regressor__min_samples_split": [5, 10],
        "regressor__min_samples_leaf": [5, 10],
    },
    "XGBoost": {
        "regressor__n_estimators": [20, 50, 100],  # Fewer estimators
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5],
        "regressor__gamma": [0, 0.1],
        "regressor__subsample": [0.7, 1.0],
    },
    "GradientBoosting": {
        "regressor__n_estimators": [20, 50, 100],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5],
        "regressor__subsample": [0.7, 1.0],
    },
    "AdaBoost": {
        "regressor__n_estimators": [20, 50],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__loss": ["linear", "square"],
    },
    "SVR": {
        "regressor__C": [0.1, 1, 5],
        "regressor__epsilon": [0.05, 0.1],
        "regressor__kernel": ["linear", "rbf"],
    },
    "KNN": {
        "regressor__n_neighbors": [3, 5],
        "regressor__weights": ["uniform", "distance"],
        "regressor__metric": ["minkowski", "euclidean"],
    },
    "LinearRegression": {
        "regressor__fit_intercept": [True, False],
        "regressor__normalize": [True, False],
    },
    "ExtraTrees": {
        "regressor__n_estimators": [20, 50, 100],
        "regressor__max_depth": [3, 5, 10],
        "regressor__min_samples_split": [5, 10],
        "regressor__min_samples_leaf": [5, 10],
    },
    "DecisionTree": {
        "regressor__max_depth": [3, 5, 10],
        "regressor__min_samples_split": [5, 10],
        "regressor__min_samples_leaf": [5, 10],
    },
}

    bayesian_param_spaces = {
    "LightGBM": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
        "num_leaves": lambda trial: trial.suggest_int("num_leaves", 15, 63),
    },
    "RandomForest": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 200),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
    },
    # Add other models here if needed
   }
    
    print(f"Model 1: Estimating property {model_group} prices using real estate transaction data only.")
    # Loop over models
    for model_name, model in models.items():
        print(f"\nTraining and tuning {model_name} model with {cv}-fold cross-validators...")
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Apply n_estimators only for tree-based and ensemble models
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "AdaBoost", "Extra Trees"]:
            model.set_params(n_estimators=cv) # Incrementally add trees

        # Progress bar for model training
        progress_bar = tqdm(range(1, cv + 1), desc=f"Training {model_name}", unit="tree")

        for i in progress_bar:
            
            # Select tuning method
            # param_grid = param_grids.get(model_name, {})
            if tuning == "grid":
                param_grid = grid_param_grids.get(model_name, {})
                print(f"Performing Grid Search for {model_name} and {tuning} tuning...")
                tuner = GridSearchCV(
                    model, param_grid, scoring=scoring, n_jobs=-1, cv=cv, verbose=0
                )
            elif tuning == "random":
                param_grid = random_param_grids.get(model_name, {})
                print(f"Performing Randomized Search for {model_name} and {tuning} tuning...")
                tuner = RandomizedSearchCV(
                    model, param_grid, n_iter=30, cv=cv, scoring=scoring, n_jobs=-1, verbose=0, random_state=42
                )
                
            elif tuning == "bayesian" and model_name in bayesian_param_spaces:
                print(f"Performing Bayesian Search for {model_name}...")
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                param_space = bayesian_param_spaces[model_name]
                
                def objective(trial):
                    # Set parameters from the defined space
                    params = {f"regressor__{key}": func(trial) for key, func in param_space.items()}
                    
                    model.set_params(**params)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    return mean_squared_error(y_val, y_pred)
                
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=20, timeout=3600)
                best_params = study.best_params
                
                 # Update the pipeline with best parameters
                model.set_params(**best_params)
                best_model = model.fit(X_train, y_train)
                best_score = -study.best_value
                # tuner = pipeline  # No tuner for Bayesian; model is trained directly
                
            else:
                print(f"No tuning for {model_name}. Skipping...")
                continue

            # Perform hyperparameter tuning
            if tuning != "bayesian":
                tuner.fit(X_train, y_train)
                best_model = tuner.best_estimator_
                best_params = tuner.best_params_
                best_score =  tuner.best_score_
            # else:
                # best_model = tuner  # For Bayesian, model is already tuned
                # best_params = study.best_params if tuning == "bayesian" else {}
                # best_score = study.best_score if tuning == "bayesian" else {}

        # Save the trained model
        model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Best {model_name} saved at {model_path}.")
        
        # Evaluate the model
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
                # Cross-validation scores
        # cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring)
        
        # store tuned models
        # tuned_models[model_name] = best_model
        
        metrics.append({
            "Model": model_name,
            "RMSE": rmse,
            "MSE": mse,
            "RSquared": r2,
            # "Best Params": str(best_params),
            "Best Params": best_params,
            "Best Score": best_score,
            # 'CV Accuracy': cv_scores.mean()
        })
                
        # print(f"\n{model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f}")
        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best {scoring} score: {best_score:.4f}\n")
        
        print("-"*120)

    # Visualize metrics
    metrics_df = pd.DataFrame(metrics)
    
    # print("Performance metrics:")
    # print(metrics_df)
    # print("-"*120)
    
    # metrics_df.set_index('Model')[['RMSE', 'MSE', 'RSquared']].plot(kind='bar', figsize=(14, 4))
    # plt.ylabel('Scores')
    # plt.title('Model Performance Comparison')
    # plt.show()


    return metrics_df, tuned_models


def tune_models_with_comparison(models, param_grids, x_train, y_train, x_val, y_val, search_type="grid", cv=5, scoring="neg_mean_squared_error", random_iter=10, output_dir=None,category=None):
    """
    Tune models using GridSearchCV or RandomizedSearchCV and perform statistical comparisons.

    Args:
        models (dict): Dictionary of models with names as keys and model instances as values.
        param_grids (dict): Dictionary of hyperparameter grids for each model.
        x_train, y_train: Training data.
        x_val, y_val: Validation data.
        search_type (str): "grid" for GridSearchCV, "random" for RandomizedSearchCV.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric for model selection.
        random_iter (int): Number of iterations for RandomizedSearchCV.

    Returns:
        dict: Best models, their scores, and statistical comparisons.
    """
    results = {}
    best_models = {}

    # for model_name, model in models.items():
    for model_name, model in tqdm(models, total=len(models), desc=f"Evaluating {category} Models"):
        print(f"Tuning {model_name}... for the {category} catgory")
        model_output_dir = os.path.join(output_dir, model_name) if output_dir else None
        if model_output_dir:
            os.makedirs(model_output_dir, exist_ok=True)

        if search_type == "grid":
            search = GridSearchCV(model, param_grids[model_name], cv=cv, scoring=scoring, n_jobs=-1)
        elif search_type == "random":
            search = RandomizedSearchCV(model, param_grids[model_name], n_iter=random_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=42)
        else:
            raise ValueError("search_type must be 'grid' or 'random'")
        
        
        search.fit(x_train, y_train)
        
        y_pred = best_model.predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        docs_results_df = pd.DataFrame(search.cv_results_)
        docs_results_df = results_df.sort_values(by=["rank_test_score"])
        docs_results_df = results_df.set_index(
            results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("kernel")
        docs_results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

        best_model = search.best_estimator_
        best_params = search.best_params_
        val_score = best_model.score(x_val, y_val)
        
            # "Model": model_name,
            # "RMSE": rmse,
            # "MSE": mse,
            # "RSquared": r2,
        
        best_models[model_name] = best_model
        results[model_name] = \
        {"Best Params": best_params, "Validation Score": val_score,"RMSE": rmse,"MSE": mse,"RSquared": r2}
        
        model_path = os.path.join(model_output_dir, f"{model_name}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Best {model_name} saved at {model_path}.")
        
        # create df of model scores ordered by performance
        model_scores = docs_results_df.filter(regex=r"split\d*_test_score")

        # plot 30 examples of dependency between cv fold and AUC scores
        fig, ax = plt.subplots()
        import seaborn as sns
        sns.lineplot(
            data=model_scores.transpose().iloc[:30],
            dashes=False,
            palette="Set1",
            marker="o",
            alpha=0.5,
            ax=ax,
        )
        ax.set_xlabel("CV test fold", size=12, labelpad=10)
        ax.set_ylabel("Model AUC", size=12)
        ax.tick_params(bottom=True, labelbottom=False)
        plt.show()

        # print correlation of AUC scores across folds
        print(f"Correlation of models:\n {model_scores.transpose().corr()}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Statistical Comparison (Pairwise t-tests and Wilcoxon tests)
    model_names = list(best_models.keys())
    model_scores = {name: best_models[name].score(x_val, y_val) for name in model_names}

    comparisons = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            scores1, scores2 = model_scores[model1], model_scores[model2]

            # Perform statistical tests
            t_stat, t_p = ttest_rel([scores1], [scores2])
            w_stat, w_p = wilcoxon([scores1], [scores2])

            comparisons.append({
                "Model 1": model1, "Model 2": model2,
                "T-Test P-Value": t_p, "Wilcoxon P-Value": w_p
            })

    comparisons_df = pd.DataFrame(comparisons)

    return {
        "best_models": best_models,
        "results_df": results_df,
        "comparisons_df": comparisons_df,
        "docs_results_df": docs_results_df
    }


# Define a function for EEMD-SD-SVM
def eemd_sd_svm_model(x, y, test_data=None):
    """
    EEMD-SD-SVM Pipeline.

    Args:
        x (pd.DataFrame): Training input features.
        y (pd.Series): Training target.
        test_data (pd.DataFrame): Test input features (optional).

    Returns:
        Pipeline: Fitted EEMD-SD-SVM pipeline.
    """
    # eemd = EEMD()
    
    # Decompose each feature using EEMD and extract statistical features
    def eemd_features(data):
        decomposed_features = []
        for feature in data.T:
            imfs = emd(feature)
            stats = [
                np.mean(imfs, axis=1),  # Mean of IMFs
                np.std(imfs, axis=1),  # Standard deviation of IMFs
                np.max(imfs, axis=1),  # Maximum value of IMFs
            ]
            decomposed_features.append(np.hstack(stats))
        return np.hstack(decomposed_features)
    
    x_transformed = eemd_features(x.values.T)
    if test_data is not None:
        test_transformed = eemd_features(test_data.values.T)
    else:
        test_transformed = None

    # SVM Model
    svm_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

    pipeline = Pipeline([
        ("eemd_transform", eemd_features),
        ("svm", svm_model),
    ])
    pipeline.fit(x_transformed, y)
    return pipeline

# Define a stacked model
def stacked_model(x_train, y_train, base_models, meta_model):
    """
    Create a stacked model pipeline.

    Args:
        x_train (pd.DataFrame): Training input features.
        y_train (pd.Series): Training target.
        base_models (list): List of base models.
        meta_model (estimator): Meta-model.

    Returns:
        StackingRegressor: Fitted stacking regressor.
    """
    estimators = [(f"model_{i}", model) for i, model in enumerate(base_models)]
    stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=meta_model)
    stacking_regressor.fit(x_train, y_train)
    return stacking_regressor

    