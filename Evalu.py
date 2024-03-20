import pandas as pd

# Load CSV files into pandas DataFrames
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

# Specify columns to compare
columns_to_compare = ['column1', 'column2', 'column3', 'column4']

# Check if specified columns match and count the matches
matches = (df1[columns_to_compare] == df2[columns_to_compare]).all(axis=1).sum()

# Calculate accuracy
accuracy = matches / len(df1) * 100

print("Accuracy:", accuracy, "%")
