"""
Stock Market Data Preprocessing Script

This script loads stock market data from a CSV file, performs basic data cleaning,
and prepares the dataset for further analysis or modeling. It includes steps for
handling missing values, sorting by date, and exploratory data analysis (EDA).

Author: [Rohan Bisht]
Date: [2025-10-07]
"""

import pandas as pd

# Load the dataset from the CSV file
# This assumes 'stock_data.csv' is in the same directory as this script
df = pd.read_csv('stock_data.csv')

# Exploratory Data Analysis (EDA) - Uncomment the following lines to inspect the data
# Display the first few rows of the dataset
# print(df.head())
# print("-" * 150)
# Display summary information about the dataset (columns, data types, non-null counts)
# print(df.info())
# print("-" * 150)
# Check for missing values in each column
# print(df.isnull().sum())
# print("-" * 150)
# Display the shape of the dataset (number of rows and columns)
# print(df.shape)

# Data Cleaning: Remove rows with missing values
# This modifies the dataframe in place, dropping any rows that contain NaN values
df.dropna(inplace=True)

# Date Conversion: Convert the 'Date' column to datetime format
# Uncomment the following line if the date column needs conversion
# Note: 'format='mixed'' is used for automatic parsing of various date formats
# new_df['Date'] = pd.to_datetime(new_df['Date'], format='mixed')

# Uncomment to display the cleaned dataframe
# print(new_df.to_string())

# Further EDA on the cleaned dataset - Uncomment to check info
# print(new_df.info())

# Check for duplicate rows in the original dataframe
# This prints a boolean series indicating duplicate rows
# print(df.duplicated().to_string())
# Uncomment the following line to remove duplicates if needed
# new_df = df.drop_duplicates()

# Final data preparation: Reset the index, sort by 'Date' column
# Note: 'sort' is deprecated in newer pandas versions; use 'sort_values' instead
new_df = df.reset_index(drop=True).sort_values('Date')

# Display the final cleaned and sorted dataframe
print(new_df.to_string())
