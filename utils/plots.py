import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


def plot_trends(dataframe, main_col, time_col, secondary_col=None, time_format="%Y", figsize=(14, 4)):
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


def process_transaction_dates(transactions_df):
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
    transactions_df['Transaction Date'] = pd.to_datetime(
        transactions_df['Transaction Date'], format='%d/%m/%Y %H:%M', errors='coerce'
    )
    
    # Handle missing values (optional: drop rows or fill missing dates)
    missing_dates = transactions_df['Transaction Date'].isna().sum()
    if missing_dates > 0:
        print(f"Warning: {missing_dates} invalid date entries found. Replacing with NaT.")
        # Optional: Drop rows with invalid dates
        # transactions_df.dropna(subset=['Transaction Date'], inplace=True)
    
    # Extract useful components
    transactions_df['Year'] = transactions_df['Transaction Date'].dt.year
    transactions_df['Month'] = transactions_df['Transaction Date'].dt.month
    transactions_df['Day'] = transactions_df['Transaction Date'].dt.day
    transactions_df['Hour'] = transactions_df['Transaction Date'].dt.hour

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
    
    
    
