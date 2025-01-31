import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, learning_curve
import seaborn as sns
import warnings
import yaml

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../configs/config.yaml")  # Adjusted path for config.yaml

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    
import data_loader, data_processor, plots, visualizations, insights



def plot_highlighted_trends(yearly_data, y_columns, titles, xlabel, ylabel, color_map=None, highlight_threshold=None):
    """
    Plots trends for multiple columns in a dataset with an option to highlight certain values.

    Parameters:
    - yearly_data (pd.DataFrame): The dataframe containing the yearly data.
    - y_columns (list): List of column names for which trends need to be plotted.
    - titles (list): List of titles for the subplots.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - color_map (list, optional): List of colors for each line.
    - highlight_threshold (float, optional): The value above which points will be highlighted in the plot.
    """
    plt.figure(figsize=(10, 6))

    for i, column in enumerate(y_columns):
        plt.subplot(len(y_columns), 1, i + 1)
        color = color_map[i] if color_map else None
        plt.plot(yearly_data['Year'], yearly_data[column], label=f'{column} Trend', color=color)
        
        if highlight_threshold:
            # Highlight points above the threshold in red
            highlighted = yearly_data[yearly_data[column] > highlight_threshold]
            plt.scatter(highlighted['Year'], highlighted[column], color='red', label='Highlighted (>11B AED)', zorder=5)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titles[i])
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()


def visualize_training_errors_by_algorithm(
    staged_predictions, 
    y_train, 
    y_val, 
    training_times, 
    n_estimators,
    figsize=(18, 5),
    colors=None,
    title_fontsize=16,
    pad = 0,
    h_pad = 0,
    w_pad = 0,
    rect = 0.92,  
):
    """
    Visualize training/validation errors and comparison of training times and estimators, grouped by algorithm.

    Args:
        staged_predictions (dict): Staged predictions for models during training.
        y_train (pd.DataFrame): True training target values.
        y_val (pd.DataFrame): True validation target values.
        training_times (dict): Training times for each model.
        n_estimators (dict): Number of estimators for each model.
        figsize (tuple): Size of each group of subplots (default: (18, 5)).
        colors (list): Colors for the bar charts (default: None).
        title_fontsize (int): Font size for the group title (default: 16).
    """
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    # Remove tuple wrapping from training_times and n_estimators
    training_times = training_times[0]
    n_estimators = n_estimators[0]

    # Default colors for bar charts
    if colors is None:
        colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'cyan']

    # Group models by algorithm prefix
    algorithms = {
        "Gradient Boosting": ["Gradient Boosting (Full)", "Gradient Boosting (Early Stopping)"],
        "HistGradient Boosting": ["HistGradient Boosting (Full)", "HistGradient Boosting (Early Stopping)"],
        "MLP Regressor": ["MLP Regressor (Full)", "MLP Regressor (Early Stopping)"],
        "Random Forest": ["Random Forest"],
        # "EEMD-SD-SVM": ["EEMD-SD-SVM"],
        # "Stacked Model":["Stacked Model"]
        }
    

    for algo_name, model_names in algorithms.items():
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        fig.suptitle(f"{algo_name} Performance", fontsize=title_fontsize)

        # Training and Validation Errors
        for model_name in model_names:
            if model_name in staged_predictions["train"]:
                train_errors = [mean_squared_error(y_train, pred) for pred in staged_predictions["train"][model_name]]
                val_errors = [mean_squared_error(y_val, pred) for pred in staged_predictions["val"][model_name]]

                # Explicitly set labels to ensure legend works
                axes[0].plot(train_errors, label=model_name)
                axes[1].plot(val_errors, label=model_name)

        axes[0].set_title("Training Error")
        axes[0].set_xlabel("Boosting Iterations")
        axes[0].set_ylabel("MSE (Training)")
        axes[0].legend()

        axes[1].set_title("Validation Error")
        axes[1].set_xlabel("Boosting Iterations")
        axes[1].set_ylabel("MSE (Validation)")
        axes[1].legend()

        # Bar Chart: Training Time vs. Estimators
        labels = [model for model in model_names if model in training_times]
        times = [training_times[model] for model in labels]
        estimators = [n_estimators[model] for model in labels]

        bars = axes[2].bar(labels, times, color=colors[:len(labels)])
        axes[2].set_ylabel("Training Time (s)")
        axes[2].set_title("Training Time vs Estimators")

        # Annotate bars with the number of estimators
        for bar, est in zip(bars, estimators):
            height = bar.get_height()
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"Estimators: {est}",
                ha="center",
                va="bottom",
            )
          
        # Adjust layout to avoid blank spaces
        plt.tight_layout(rect=[pad, h_pad, w_pad, rect])
        plt.show()

    
# def visualize_training_errors(staged_predictions, y_train, y_val):
#     """
#     Visualize training and validation errors over boosting iterations.

#     Args:
#         staged_predictions (dict): Staged predictions for models during training.
#         y_train (pd.DataFrame): True training target values.
#         y_val (pd.DataFrame): True validation target values.
#     """
#     from sklearn.metrics import mean_squared_error

#     fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

#     for model_name, predictions in staged_predictions["train"].items():
#         train_errors = [mean_squared_error(y_train, pred) for pred in predictions]
#         val_errors = [mean_squared_error(y_val, pred) for pred in staged_predictions["val"][model_name]]

#         axes[0].plot(train_errors, label=model_name)
#         axes[1].plot(val_errors, label=model_name)

#     axes[0].set_title("Training Error")
#     axes[0].set_xlabel("Boosting Iterations")
#     axes[0].set_ylabel("MSE (Training)")
#     axes[1].set_yscale("log")
#     axes[0].legend()

#     axes[1].set_title("Validation Error")
#     axes[1].set_xlabel("Boosting Iterations")
#     axes[1].set_ylabel("MSE (Validation)")
#     axes[1].set_yscale("log")
#     axes[1].legend()

#     plt.tight_layout()
#     plt.show()


#     plt.tight_layout()
#     plt.show()

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[sorted_idx], align='center')
        plt.xticks(range(len(importance)), feature_names[sorted_idx], rotation=90)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance is not available for this model.")

def visualize_predictions(y_true, y_pred, title="Actual vs. Predicted"):
    """Plot actual vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_heatmap(y_true, y_pred, target_name):
    plt.figure(figsize=(14, 5))
    heatmap_data = np.vstack([y_true, y_pred]).T
    sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', cbar=True)
    plt.title(f'Heatmap of Actual vs Predicted for {target_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()

def plot_cv_results(model, X, y, target_name):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    plt.figure(figsize=(14, 4))
    sns.boxplot(y=cv_scores, color='lightblue')
    plt.title(f'Cross-Validation Results for {target_name}')
    plt.ylabel('Negative RMSE')
    plt.show()

def plot_learning_curve(model, X_train, y_train, target_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], cv=5, n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(14, 4))
    plt.plot(train_sizes, train_mean, label="Training Score", color='blue')
    plt.plot(train_sizes, val_mean, label="Validation Score", color='orange')
    plt.title(f'Learning Curve for {target_name}')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def plot_feature_importances(model, feature_names, top_n=None, title="Feature Importance"):
    """
    Plot and return feature importances from a trained model.
    
    Parameters:
        model: Trained model with `feature_importances_` attribute.
        feature_names: List or Index of feature names corresponding to the model's input.
        top_n: (Optional) Number of top features to display. If None, display all features.
        title: Title for the plot.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Importance'] sorted by importance (descending).
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort indices by importance in descending order

    # Optionally limit to top_n features
    if top_n is not None:
        indices = indices[:top_n]

    # Create DataFrame for feature importances
    importance_df = pd.DataFrame({
        "Feature": np.array(feature_names)[indices],
        "Importance": importances[indices]
    })

    # Plot feature importances
    plt.figure(figsize=(14, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    return importance_df

def plot_actual_vs_predicted(y_true, y_pred, target_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='green')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title(f'Actual vs. Predicted for {target_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def plot_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot for {target_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()
   
def plot_missingness_no_labels(df, title="Percentage of Missing Values by Feature"):
    """
    Calculates and visualizes the percentage of missing values for each feature in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - title (str): The title for the visualization.

    Returns:
    - pd.DataFrame: A DataFrame with missing percentages for each feature.
    """
    # Calculate missing values and percentages
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Feature': missing_values.index,
        'MissingPercent': missing_percent
    }).sort_values(by='MissingPercent', ascending=False)

    # Visualize missing values
    plt.figure(figsize=(14, 6))
    sns.barplot(data=missing_df, y='Feature', x='MissingPercent', palette='viridis',hue='Feature', legend=False)
    plt.title(title)
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Feature")
    plt.show()
    
    return missing_df

def plot_missingness_labels(data, feature_col='Features',rotation=15,percent_col='Perc(%)', plot_size=(15,5), font_size=15):
    """
    Plots missing values by column.

    Parameters:
    - data (DataFrame): Data containing feature names and their corresponding missing percentage.
    - feature_col (str): Column name representing feature names.
    - percent_col (str): Column name representing the percentage of missing values.

    Returns:
    - None (Displays the plot)
    """
    # Setting style and palette
    sns.set(style="ticks", palette="pastel")

    # Creating the plot
    plt.figure(figsize=plot_size)

    # Adding background color and grid
    plt.gca().set_facecolor('#f0f0f0')  # Replace with the desired background color (e.g., light gray)
    plt.grid(axis='y', linestyle='--', alpha=0.8)

    # Building the bar plot
    ax = sns.barplot(x=data[feature_col], y=data[percent_col])

    # Adding percentage labels inside the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.xticks(rotation=rotation)  # Rotating X-axis labels by 25 degrees
    plt.title('Columns with Missing Values', fontsize=font_size)
    plt.xlabel('Columns', fontsize=font_size)
    plt.ylabel('% of Missing Data', fontsize=font_size)
    plt.show()

def preprocess_real_estate_data(df):
    """
    Preprocesses the merged real estate DataFrame by imputing missing values, 
    calculating average prices per sq.m by month and week, and deriving new features.
    """
    # Convert Transaction Date to datetime and derive YearMonth and YearWeek
    # df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    # df['YearMonth'] = df['Transaction Date'].dt.to_period('M')
    # df['YearWeek'] = df['Transaction Date'].dt.strftime('%Y-%U')

    # Calculate global medians for fallback imputation
    global_median_amount = df['Amount'].median()
    global_median_contract_amount = df['Contract Amount'].median()
    global_median_annual_amount = df['Annual Amount'].median()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Impute Amount, Contract Amount, and Annual Amount first
        df['Amount'] = df.groupby('Property Sub Type')['Amount'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median_amount)
        )
        df['Contract Amount'] = df.groupby('Property Sub Type')['Contract Amount'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median_contract_amount)
        )
        df['Annual Amount'] = df.groupby('Property Sub Type')['Annual Amount'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median_annual_amount)
        )

    # Calculate price per sq.m
    df['Price per sq.m'] = df['Amount'] / df['Transaction Size (sq.m)']

    # Average prices by month
    avg_price_month = df.groupby(['Property Type', 'Area', 'YearMonth'])['Price per sq.m'].mean().reset_index()
    avg_price_month.rename(columns={'Price per sq.m': 'Avg Price Last Month'}, inplace=True)

    # Average prices by week
    avg_price_week = df.groupby(['Property Type', 'Area', 'YearWeek'])['Price per sq.m'].mean().reset_index()
    avg_price_week.rename(columns={'Price per sq.m': 'Avg Price per sq.m Last Week'}, inplace=True)

    # Merge averages back into the main DataFrame
    df = pd.merge(df, avg_price_month, on=['Property Type', 'Area', 'YearMonth'], how='left')
    df = pd.merge(df, avg_price_week, on=['Property Type', 'Area', 'YearWeek'], how='left')

    # Extract number of rooms as numeric
    # df['Number of Rooms'] = df['Room(s)'].str.extract('(\d+)').astype(float)
    # global_median_rooms = df['Number of Rooms'].median()
    # global_median_parking = df['Parking'].median()
    # df['Number of Rooms'] = df.groupby('Property Sub Type')['Number of Rooms'].transform(
    #     lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median_rooms)
    # )
    # df['Parking'] = df.groupby('Property Sub Type')['Parking'].transform(
    #     lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median_parking)
    # )

    # # Impute categorical columns using modes
    # categorical_columns = ['Property Sub Type', 'Room(s)', 'Nearest Metro', 'Area', 'Nearest Mall', 'Nearest Landmark', 'Usage']
    # for col in categorical_columns:
    #     df[col] = df.groupby('Property Type')[col].transform(
    #         lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else 'Not Available'
    #     )

    # # Impute Property Size using medians
    # df['Property Size (sq.m)'] = df.groupby('Property Sub Type')['Property Size (sq.m)'].transform(
    #     lambda x: x.fillna(x.median())
    # )

    # # Drop rows with critical missing values
    # df.dropna(subset=['Number of Rooms', 'Parking','Avg Price Last Month', 'Avg Price per sq.m Last Week'], inplace=True)

        # Ensure no infinity values remain
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def categorize_missingness(percentage):
    """
    Categorizes missingness based on percentage.
    """
    if percentage == 100:
        return "Drop (100% Missing)"
    elif percentage > 50:
        return "Consider Dropping (>50%)"
    elif percentage > 20:
        return "High Missingness (20-50%)"
    elif percentage > 0:
        return "Low Missingness (<20%)"
    else:
        return "No Missing Data"

def visualize_missing_values(df, title="Percentage of Missing Values by Feature"):
    """
    Calculates, categorizes, and visualizes the percentage of missing values for each feature in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - title (str): The title for the visualization.

    Returns:
    - pd.DataFrame: A DataFrame with missing percentages and categories for each feature.
    """
    # Calculate missing values and percentages
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Feature': missing_values.index,
        'MissingPercent': missing_percent
    }).sort_values(by='MissingPercent', ascending=False)

    # Assign missingness labels
    missing_df['MissingnessCategory'] = missing_df['MissingPercent'].apply(categorize_missingness)

    # Visualize missing values
    plt.figure(figsize=(14, 6))
    sns.barplot(data=missing_df, y='Feature', x='MissingPercent', palette='viridis', hue='MissingnessCategory', dodge=False)
    plt.title(title)
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Feature")
    plt.legend(title="Missingness Category")
    plt.show()
    
    return missing_df

def plot_histograms(dataframe, title="Histogram Subplots", bins=50):
    """
    Plots histograms for all numeric columns in the dataframe as subplots.

    Args:
    - dataframe (pd.DataFrame): Input dataframe.
    - title (str): Title for the overall plot.
    - bins (int): Number of bins for histograms.

    Returns:
    - None
    """
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    num_cols = len(numeric_columns)
    rows = (num_cols // 3) + (num_cols % 3 > 0)  # 3 columns per row

    fig, axes = plt.subplots(rows, 3, figsize=(18, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_columns):
        sns.histplot(dataframe[col].dropna(), bins=bins, ax=axes[i], kde=True, color='blue')
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.92)
    plt.tight_layout()
    plt.show()

def plot_countplots(dataframe, columns, max_categories=10, figsize=(15, 5)):
    """
    Plots count plots for categorical variables in a dataframe.
    Automatically handles high-cardinality columns by showing Top-N categories.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        columns (list): List of categorical columns to plot.
        max_categories (int): Maximum number of categories to display (default: 10).
        figsize (tuple): Size of each subplot.
    """
    filtered_columns = [col for col in columns if col in dataframe.columns]
    num_columns = len(filtered_columns)
    
    fig, axes = plt.subplots(
        nrows=(num_columns + 2) // 3, ncols=3, figsize=(figsize[0], figsize[1] * ((num_columns + 2) // 3))
    )
    axes = axes.flatten() if num_columns > 1 else [axes]

    for i, col in enumerate(filtered_columns):
        unique_vals = dataframe[col].nunique()
        if unique_vals > max_categories:
            # Group low-frequency categories into "Other"
            top_categories = dataframe[col].value_counts().nlargest(max_categories).index
            plot_data = dataframe[col].apply(lambda x: x if x in top_categories else "Other")
            title = f"{col} (Top {max_categories} + Other)"
        else:
            plot_data = dataframe[col]
            title = col

        sns.countplot(data=dataframe, x=plot_data, ax=axes[i], palette="viridis", hue=plot_data, order=plot_data.value_counts().index)
        axes[i].set_title(title)
        axes[i].tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()

def plot_trends_secondary_cols(dataframe, main_col, time_col, secondary_col=None, time_format="%Y", figsize=(14, 4)):
    """
    Plot trends for a specific column over time with an optional secondary column.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        main_col (str): Main column to plot on the primary y-axis.
        time_col (str): Time column to use for the x-axis.
        secondary_col (str, optional): Column to plot on the secondary y-axis.
        time_format (str, optional): Format for parsing the time column. Default is "%Y".
        figsize (tuple): Size of the plot.
    """
    # Ensure time_col is datetime
    if time_col in dataframe:
        if not pd.api.types.is_datetime64_any_dtype(dataframe[time_col]):
            dataframe[time_col] = pd.to_datetime(dataframe[time_col], format=time_format, errors="coerce")
    else:
        raise KeyError(f"Column '{time_col}' not found in DataFrame.")

    # Extract year for grouping
    if "Year" not in dataframe:
        dataframe["Year"] = dataframe[time_col].dt.year

    grouped = dataframe.groupby("Year").agg({main_col: "mean"})
    
    plt.figure(figsize=figsize)
    ax = grouped.plot(y=main_col, use_index=True, label=main_col, legend=True)
    ax.set_ylabel(main_col, color="blue")
    ax.set_title(f"Trend of {main_col} Over Time")
    
    if secondary_col:
        grouped[secondary_col] = dataframe.groupby("Year")[secondary_col].mean()
        grouped[secondary_col].plot(secondary_y=True, label=secondary_col, ax=ax, legend=True, color="orange")
        ax.right_ax.set_ylabel(secondary_col, color="orange")
    
    plt.tight_layout()
    plt.show()

def plot_macroeconomic_distributions(df, categorical_cols, title):
    """
    Plot distributions for numeric and categorical variables in subplots.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    n_cols = 3  # Number of columns in the plot grid
    n_rows = (n_numeric + n_categorical + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    # Numeric variables
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f"Distribution of {col}")

    # Categorical variables
    for i, col in enumerate(categorical_cols, start=n_numeric):
        sns.countplot(data=df, x=col, ax=axes[i], order=df[col].value_counts().index)
        axes[i].set_title(f"Count Plot of {col}")
        axes[i].tick_params(axis='x', rotation=45)

    # Remove unused axes
    for ax in axes[n_numeric + n_categorical:]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, title="Correlation Matrix"):
    """
    Plot a heatmap of correlations for the given numeric columns.
    """
    
    numeric_cols = df.select_dtypes(include="number").columns

    plt.figure(figsize=(14, 5))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()
    
def plot_macroeconomic_trends(df, time_col, y_col, compare_col=None, title="Macroeconomic Trends"):
    """
    Plot a time-series trend for a macroeconomic variable with an optional comparison variable.
    """

    plt.figure(figsize=(14, 5))
    plt.plot(df[time_col], df[y_col], label=y_col, color='blue')
    
    if compare_col:
        plt.plot(df[time_col], df[compare_col], label=compare_col, color='orange')

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
       
def plot_trends_single_cols(yearly_data, y_columns, titles, xlabel, ylabel, color_map=None):
    """
    Plots trends for multiple columns in a dataset.

    Parameters:
    - yearly_data (pd.DataFrame): The dataframe containing the yearly data.
    - y_columns (list): List of column names for which trends need to be plotted.
    - titles (list): List of titles for the subplots.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - color_map (list, optional): List of colors for each line.
    """
    plt.figure(figsize=(14, 5))
    
    for i, column in enumerate(y_columns):
        plt.subplot(len(y_columns), 1, i + 1)
        color = color_map[i] if color_map else None
        plt.plot(yearly_data['Year'], yearly_data[column], label=f'{column} Trend', color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titles[i])
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()
 
# def plot_highlighted_trends(
#     df,
#     columns,
#     titles,
#     x_axis,
#     y_axis,
#     color_map,
#     highlight_threshold
# ):
    
#     plt.figure(figsize=(10, 6))

#     for i, column in enumerate(columns):
#         plt.plot(df[x_axis], df[column], label=titles[i], color=color_map[i])

#         # Highlight points above the threshold
#         highlighted = df[df[column] > highlight_threshold]
#         plt.scatter(highlighted[x_axis], highlighted[column], color=color_map[i], zorder=5)

#         # Identify and label extremes (min and max)
#         extremes = data_processor.identify_extremes(df, column)
#         # Plot max and min points
#         plt.annotate(extremes['max'][2], (extremes['max'][0], extremes['max'][1]), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
#         plt.annotate(extremes['min'][2], (extremes['min'][0], extremes['min'][1]), textcoords="offset points", xytext=(0, -10), ha='center', color='green')

#     plt.xlabel(x_axis)
#     plt.ylabel(y_axis)
#     plt.title('Real Estate Trends with Highlights')
#     plt.legend()
#     plt.show()

def plot_trends_with_two_y_axes(df, x_column, y1_column, y2_column, y3_column, y4_column, title, y1_label, y2_label, y3_label, y4_label):
    """
    Plot two features on the left y-axis and another two on the right y-axis.
    
    Parameters:
    - df: DataFrame containing the data
    - x_column: Column for the x-axis (usually 'Year')
    - y1_column: Column for the first feature on the left y-axis
    - y2_column: Column for the second feature on the left y-axis
    - y3_column: Column for the first feature on the right y-axis
    - y4_column: Column for the second feature on the right y-axis
    - title: Title of the plot
    - y1_label: Label for the first left y-axis
    - y2_label: Label for the second left y-axis
    - y3_label: Label for the first right y-axis
    - y4_label: Label for the second right y-axis
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Left Y axis: y1_column and y2_column
    ax1.plot(df[x_column], df[y1_column], label=y1_label, color='blue', marker='o')
    ax1.plot(df[x_column], df[y2_column], label=y2_label, color='green', marker='o')
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel(y1_label, fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Right Y axis: y3_column and y4_column
    ax2 = ax1.twinx()
    ax2.plot(df[x_column], df[y3_column], label=y3_label, color='red', marker='x')
    ax2.plot(df[x_column], df[y4_column], label=y4_label, color='orange', marker='x')
    
    ax2.set_ylabel(y3_label, fontsize=12)
    ax2.legend(loc='upper right')
    
    # Title and layout adjustments
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
       