import pandas as pd

# Load the dataset
df = pd.read_csv('stock_data.csv')

# Display basic information about the dataset
# print(df.head())
# print("-" * 150)
# print(df.info())
# print("-" * 150)
# print(df.isnull().sum())
# print("-" * 150)
# print(df.shape)

# Save the dataset after dropping empty values
df.dropna(implace = True)

#Convert the date column into datetime
# new_df['Date'] = pd.to_datetime(new_df['Date'], format='mixed')

# print(new_df.to_string())

#Checking for new dataset information
# print(new_df.info())

#Checking if any data is duplicate
# print(df.duplicated().to_string())
#new_df = df.drop_duplicates()

#cleaning wrong data
new_df = df.reset_index(drop=True)

print(new_df.to_string())