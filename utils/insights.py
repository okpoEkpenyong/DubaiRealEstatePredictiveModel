import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../configs/config.yaml")  # Adjusted path for config.yaml

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    

sys.path.append("../utils")

import data_loader, data_processor, plots, visualizations, insights



# def extract_real_estate_sales_insights(property_transactions_df=None):
#     """
#     Extracts key insights from property transactions.

#     Parameters:
#     - property_transactions_df (pd.DataFrame, optional): Sales transactions dataset.

#     Returns:
#     - list: A list of key insights about the dataset.
#     """
#     insights = ["📊 **Real Estate Sales Market Insights**"]

#     if property_transactions_df is not None:
#         insights.append("\n🏡 **Property Sales Transactions**")

#         total_sales_transactions = property_transactions_df.shape[0]
#         insights.append(f"✔️ **Total sales transactions**: {total_sales_transactions:,}")

#         # Yearly data for sales transactions
#         yearly_sales = property_transactions_df.groupby('Year').agg({
#             'Amount': 'mean',
#             'Property Size (sq.m)': 'mean',
#             'Is Free Hold?': lambda x: (x == 'Free Hold').sum(),
#         }).reset_index()

#         # Get the earliest and latest year for context
#         min_year = yearly_sales['Year'].min()
#         max_year = yearly_sales['Year'].max()
#         insights.append(f"📅 **Data from {min_year} to {max_year}**")

#         # Average Sale Price
#         if 'Amount' in property_transactions_df.columns:
#             avg_sale_price = property_transactions_df['Amount'].mean()
#             max_sale_price = property_transactions_df['Amount'].max()
#             insights.append(f"🏷 **Average sale price**: {avg_sale_price:,.2f} AED")
#             insights.append(f"💎 **Highest recorded sale price**: {max_sale_price:,.2f} AED")
        
#         # Property Size Insights
#         if 'Property Size (sq.m)' in property_transactions_df.columns:
#             avg_property_size = yearly_sales['Property Size (sq.m)'].mean()
#             insights.append(f"📐 **Average property size**: {avg_property_size:,.2f} sq.m")

#         # Freehold Transactions
#         if 'Is Free Hold?' in property_transactions_df.columns:
#             freehold_count = yearly_sales['Is Free Hold?'].sum()
#             insights.append(f"✅ **Freehold transactions**: {freehold_count:,}")

#         # Year-over-Year Change Calculations
#         if len(yearly_sales) > 1:
#             yearly_sales['YoY Change (%)'] = yearly_sales['Amount'].pct_change() * 100
#             insights.append(f"📈 **Amount YoY Change (%)**: {yearly_sales.iloc[-1]['YoY Change (%)']:.2f}%")

#         # 5-Year Change
#         if len(yearly_sales) > 5:
#             yearly_sales['5-Year Change (%)'] = yearly_sales['Amount'].pct_change(periods=5) * 100
#             insights.append(f"📊 **Amount 5-Year Change (%)**: {yearly_sales.iloc[-1]['5-Year Change (%)']:.2f}%")

#         # Plotting the trends for key metrics
#         plt.figure(figsize=(10, 6))
#         plt.subplot(2, 1, 1)
#         plt.plot(yearly_sales['Year'], yearly_sales['Amount'], label='Average Sale Price')
#         plt.xlabel('Year')
#         plt.ylabel('Average Sale Price (AED)')
#         plt.title('Average Sale Price Trend')
#         plt.grid(True)

#         plt.subplot(2, 1, 2)
#         plt.plot(yearly_sales['Year'], yearly_sales['Property Size (sq.m)'], label='Average Property Size', color='orange')
#         plt.xlabel('Year')
#         plt.ylabel('Average Property Size (sq.m)')
#         plt.title('Average Property Size Trend')
#         plt.grid(True)

#         plt.tight_layout()
#         plt.show()

#     return insights


def extract_real_estate_rentals_insights(property_rental_transactions_df=None):
    """
    Extracts key insights from property rental transactions.

    Parameters:
    - property_rental_transactions_df (pd.DataFrame, optional): Rental transactions dataset.

    Returns:
    - list: A list of key insights about the dataset.
    """
    insights = ["📊 **Real Estate Rentals Market Insights**"]

    if property_rental_transactions_df is not None:
        insights.append("\n🏠 **Property Rental Transactions**")

        total_rental_transactions = property_rental_transactions_df.shape[0]
        insights.append(f"✔️ **Total rental transactions**: {total_rental_transactions:,}")

        # Yearly data for rental transactions
        yearly_rentals = property_rental_transactions_df.groupby('Year').agg({
            'Contract Amount': 'mean',
            'Annual Amount': 'mean',
            'Property Size (sq.m)': 'mean',
            'Is Free Hold?': lambda x: (x == 'Free Hold').sum(),
        }).reset_index()

        # Get the earliest and latest year for context
        min_year = yearly_rentals['Year'].min()
        max_year = yearly_rentals['Year'].max()
        insights.append(f"📅 **Data from {min_year} to {max_year}**")

        # Average Annual Rent
        if 'Annual Amount' in property_rental_transactions_df.columns:
            avg_annual_rent = yearly_rentals['Annual Amount'].mean()
            insights.append(f"🏠 **Average annual rent**: {avg_annual_rent:,.2f} AED")

        # Average Contract Rent
        if 'Contract Amount' in property_rental_transactions_df.columns:
            avg_contract_rent = yearly_rentals['Contract Amount'].mean()
            insights.append(f"💰 **Average contract rent**: {avg_contract_rent:,.2f} AED")

        # Property Size Insights
        if 'Property Size (sq.m)' in property_rental_transactions_df.columns:
            avg_property_size_rentals = yearly_rentals['Property Size (sq.m)'].mean()
            insights.append(f"📐 **Average rental property size**: {avg_property_size_rentals:,.2f} sq.m")

        # Freehold Rental Transactions
        if 'Is Free Hold?' in property_rental_transactions_df.columns:
            freehold_rental_count = yearly_rentals['Is Free Hold?'].sum()
            insights.append(f"✅ **Freehold rental transactions**: {freehold_rental_count:,}")

        # Year-over-Year Change Calculations
        if len(yearly_rentals) > 1:
            yearly_rentals['YoY Change (%)'] = yearly_rentals['Annual Amount'].pct_change() * 100
            insights.append(f"📈 **Annual Amount YoY Change (%)**: {yearly_rentals.iloc[-1]['YoY Change (%)']:.2f}%")

        # 5-Year Change
        if len(yearly_rentals) > 5:
            yearly_rentals['5-Year Change (%)'] = yearly_rentals['Annual Amount'].pct_change(periods=5) * 100
            insights.append(f"📊 **Annual Amount 5-Year Change (%)**: {yearly_rentals.iloc[-1]['5-Year Change (%)']:.2f}%")

        # Plotting the trends for key metrics
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(yearly_rentals['Year'], yearly_rentals['Annual Amount'], label='Average Annual Rent', color='green')
        plt.xlabel('Year')
        plt.ylabel('Average Annual Rent (AED)')
        plt.title('Average Annual Rent Trend')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(yearly_rentals['Year'], yearly_rentals['Property Size (sq.m)'], label='Average Property Size', color='orange')
        plt.xlabel('Year')
        plt.ylabel('Average Property Size (sq.m)')
        plt.title('Average Property Size Trend for Rentals')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return insights


# def extract_real_estate_sales_insights(property_transactions_df=None):
#     if property_transactions_df is None or property_transactions_df.empty:
#         return ["⚠️ No sales transactions data available."]

#     insights = ["📊 **Real Estate Sales Market Insights**"]
#     insights.append("\n🏡 **Property Sales Transactions**")

#     total_sales_transactions = property_transactions_df.shape[0]
#     insights.append(f"✔️ **Total sales transactions**: {total_sales_transactions:,}")

#     if 'Amount' in property_transactions_df.columns:
#         avg_sale_price = property_transactions_df['Amount'].mean()
#         max_sale_price = property_transactions_df['Amount'].max()
#         insights.append(f"🏷 **Average sale price**: {avg_sale_price:,.2f} AED")
#         insights.append(f"💎 **Highest recorded sale price**: {max_sale_price:,.2f} AED")

#     if 'Property Size (sq.m)' in property_transactions_df.columns:
#         avg_property_size = property_transactions_df['Property Size (sq.m)'].mean()
#         insights.append(f"📐 **Average property size**: {avg_property_size:,.2f} sq.m")

#     if 'Is Free Hold?' in property_transactions_df.columns:
#         freehold_count = (property_transactions_df['Is Free Hold?'] == 'Free Hold').sum()
#         insights.append(f"✅ **Freehold transactions**: {freehold_count:,}")

#     # 📊 Yearly Trends
#     if 'Year' in property_transactions_df.columns:
#         yearly_sales = property_transactions_df.groupby('Year').agg({
#             'Amount': 'mean',
#             'Property Size (sq.m)': 'mean',
#             'Is Free Hold?': lambda x: (x == 'Free Hold').sum(),
#         }).reset_index()

#         if len(yearly_sales) > 1:
#             # YoY Change Calculation (Separate columns)
#             yearly_sales['Amount YoY Change (%)'] = yearly_sales['Amount'].pct_change() * 100
#             yearly_sales['Property Size (sq.m) YoY Change (%)'] = yearly_sales['Property Size (sq.m)'].pct_change() * 100
#             yearly_sales['Is Free Hold? YoY Change (%)'] = yearly_sales['Is Free Hold?'].pct_change() * 100

#             # Only display the latest year’s percentage change
#             insights.append(f"📈 **Amount YoY Change (%)**: {yearly_sales['Amount YoY Change (%)'].iloc[-1]:.2f}%")
#             insights.append(f"📈 **Property Size (sq.m) YoY Change (%)**: {yearly_sales['Property Size (sq.m) YoY Change (%)'].iloc[-1]:.2f}%")
#             insights.append(f"📈 **Is Free Hold? YoY Change (%)**: {yearly_sales['Is Free Hold? YoY Change (%)'].iloc[-1]:.2f}%")

#         if len(yearly_sales) > 5:
#             # 5-Year Change Calculation (Separate columns)
#             yearly_sales['Amount 5-Year Change (%)'] = yearly_sales['Amount'].pct_change(periods=5) * 100
#             yearly_sales['Property Size (sq.m) 5-Year Change (%)'] = yearly_sales['Property Size (sq.m)'].pct_change(periods=5) * 100
#             yearly_sales['Is Free Hold? 5-Year Change (%)'] = yearly_sales['Is Free Hold?'].pct_change(periods=5) * 100

#             # Only display the latest 5-year percentage change
#             insights.append(f"📊 **Amount 5-Year Change (%)**: {yearly_sales['Amount 5-Year Change (%)'].iloc[-1]:.2f}%")
#             insights.append(f"📊 **Property Size (sq.m) 5-Year Change (%)**: {yearly_sales['Property Size (sq.m) 5-Year Change (%)'].iloc[-1]:.2f}%")
#             insights.append(f"📊 **Is Free Hold? 5-Year Change (%)**: {yearly_sales['Is Free Hold? 5-Year Change (%)'].iloc[-1]:.2f}%")

#     return insights


# def extract_real_estate_rentals_insights(property_rental_transactions_df=None):
#     if property_rental_transactions_df is None or property_rental_transactions_df.empty:
#         return ["⚠️ No rental transactions data available."]

#     insights = ["📊 **Real Estate Rentals Market Insights**"]
#     insights.append("\n🏠 **Property Rental Transactions**")

#     total_rental_transactions = property_rental_transactions_df.shape[0]
#     insights.append(f"✔️ **Total rental transactions**: {total_rental_transactions:,}")

#     if 'Annual Amount' in property_rental_transactions_df.columns:
#         avg_annual_rent = property_rental_transactions_df['Annual Amount'].mean()
#         insights.append(f"🏠 **Average annual rent**: {avg_annual_rent:,.2f} AED")

#     if 'Contract Amount' in property_rental_transactions_df.columns:
#         avg_contract_rent = property_rental_transactions_df['Contract Amount'].mean()
#         insights.append(f"💰 **Average contract rent**: {avg_contract_rent:,.2f} AED")

#     if 'Property Size (sq.m)' in property_rental_transactions_df.columns:
#         avg_property_size_rentals = property_rental_transactions_df['Property Size (sq.m)'].mean()
#         insights.append(f"📐 **Average rental property size**: {avg_property_size_rentals:,.2f} sq.m")

#     if 'Is Free Hold?' in property_rental_transactions_df.columns:
#         freehold_rental_count = (property_rental_transactions_df['Is Free Hold?'] == 'Free Hold').sum()
#         insights.append(f"✅ **Freehold rental transactions**: {freehold_rental_count:,}")

#     # 📊 Yearly trends for Rental Transactions
#     if 'Year' in property_rental_transactions_df.columns:
#         yearly_rentals = property_rental_transactions_df.groupby('Year').agg({
#             'Contract Amount': 'mean',
#             'Annual Amount': 'mean',
#             'Property Size (sq.m)': 'mean',
#             'Is Free Hold?': lambda x: (x == 'Free Hold').sum(),
#         }).reset_index()

#         if len(yearly_rentals) > 1:
#             change_cols = [col + " YoY Change (%)" for col in yearly_rentals.columns[1:]]
#             yearly_rentals[change_cols] = yearly_rentals.iloc[:, 1:].pct_change() * 100
#             last_year_changes = yearly_rentals.iloc[-1][change_cols].fillna(0)

#             for col, change in last_year_changes.items():
#                 insights.append(f"📈 **{col}**: {change:.2f}%")

#         if len(yearly_rentals) > 5:
#             change_cols_5yr = [col + " 5-Year Change (%)" for col in yearly_rentals.columns[1:]]
#             yearly_rentals[change_cols_5yr] = yearly_rentals.iloc[:, 1:].pct_change(periods=5) * 100
#             last_5yr_changes = yearly_rentals.iloc[-1][change_cols_5yr].fillna(0)

#             for col, change in last_5yr_changes.items():
#                 insights.append(f"📊 **{col}**: {change:.2f}%")

#     return insights

def extract_real_estate_sales_insights(property_transactions_df=None):
    """
    Extracts key insights from property transactions.

    Parameters:
    - property_transactions_df (pd.DataFrame, optional): Sales transactions dataset.

    Returns:
    - list: A list of key insights about the dataset.
    """
    insights = ["📊 **Real Estate Sales Market Insights**"]

    if property_transactions_df is not None:
        insights.append("\n🏡 **Property Sales Transactions**")

        total_sales_transactions = property_transactions_df.shape[0]
        insights.append(f"✔️ **Total sales transactions**: {total_sales_transactions:,}")

        # Yearly data for sales transactions
        yearly_sales = property_transactions_df.groupby('Year').agg({
            'Amount': 'sum',
            'Property Size (sq.m)': 'sum',
            'Is Free Hold?': lambda x: (x == 'Free Hold').sum(),
        }).reset_index()

        # Get the earliest and latest year for context
        min_year = yearly_sales['Year'].min()
        max_year = yearly_sales['Year'].max()
        insights.append(f"📅 **Data from {min_year} to {max_year}**")

        # Average Sale Price and Highest Sale
        if 'Amount' in property_transactions_df.columns:
            avg_sale_price = yearly_sales['Amount'].mean()
            max_sale_price = yearly_sales['Amount'].max()
            insights.append(f"🏷 **Average sale price**: {avg_sale_price:,.2f} AED")
            insights.append(f"💎 **Highest recorded sale price**: {max_sale_price:,.2f} AED")

        # Average Property Size Insights
        if 'Property Size (sq.m)' in property_transactions_df.columns:
            avg_property_size = yearly_sales['Property Size (sq.m)'].mean()
            insights.append(f"📐 **Average property size**: {avg_property_size:,.2f} sq.m")

        # Freehold Sales Transactions
        if 'Is Free Hold?' in property_transactions_df.columns:
            freehold_count = yearly_sales['Is Free Hold?'].sum()
            insights.append(f"✅ **Freehold transactions**: {freehold_count:,}")
            
            # Fact 2: Number of unique areas
        if 'Area' in property_transactions_df.columns:
            unique_areas = property_transactions_df['Area'].nunique()
            insights.append(f"📍 **Unique areas**: {unique_areas}")

            # Fact 3: Area with the most transactions
            most_transactions_area = property_transactions_df['Area'].value_counts().idxmax()
            insights.append(f"🏙 **Top area by transactions**: {most_transactions_area}")

        # Fact 4: Most common transaction type (for sales)
        if 'Transaction Type' in property_transactions_df.columns:
            most_common_transaction_type = property_transactions_df['Transaction Type'].value_counts().idxmax()
            insights.append(f"📜 **Most common transaction type**: {most_common_transaction_type}")

        # Fact 5: Most common transaction sub type (for sales)
        if 'Transaction sub type' in property_transactions_df.columns:
            most_common_transaction_sub_type = property_transactions_df['Transaction sub type'].value_counts().idxmax()
            insights.append(f"📄 **Most common transaction sub-type**: {most_common_transaction_sub_type}")

        # Fact 6: Most common registration type (for sales)
        if 'Registration type' in property_transactions_df.columns:
            most_common_registration_type = property_transactions_df['Registration type'].value_counts().idxmax()
            insights.append(f"📝 **Most common registration type**: {most_common_registration_type}")

        # Fact 7: Most common usage (Residential/Commercial)
        if 'Usage' in property_transactions_df.columns:
            most_common_usage = property_transactions_df['Usage'].value_counts().idxmax()
            insights.append(f"🏠 **Most common usage**: {most_common_usage}")

        # Fact 8: Most common property type
        if 'Property Type' in property_transactions_df.columns:
            most_common_property_type = property_transactions_df['Property Type'].value_counts().idxmax()
            insights.append(f"🏡 **Most common property type**: {most_common_property_type}")
    

        # Highlight 11Billion AED Sale Transactions
        # highlight_threshold = 10_000_0000  # 11 Billion AED
        # Set dynamic highlight threshold based on mean of Amount for each year
        highlight_threshold = yearly_sales['Amount'].mean() * 1.7  # Use 2x the mean value as threshold

        # Print the calculated threshold for confirmation
        print(f"Dynamic Highlight Threshold: {highlight_threshold}")

        # insights.append(f"🚨 **Transactions above the Average AED**: {yearly_sales[yearly_sales['Amount'] > highlight_threshold].shape[0]}")

        # Plotting the trends and highlighting sales > 11B AED
        visualizations.plot_highlighted_trends(
            yearly_sales,
            ['Amount', 'Property Size (sq.m)'],
            ['Sale Amount Trend', 'Average Property Size Trend for Sales'],
            'Year',
            'Value (AED / sq.m)',
            color_map=['blue', 'orange'],
            highlight_threshold=highlight_threshold
        )

    return insights, yearly_sales

