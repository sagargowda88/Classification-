import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('your_file.csv')

# Concatenate the columns with their names with new lines between each column
concatenated_column = df.apply(lambda row: '\n'.join(f'{col}={val}' for col, val in row.items()), axis=1)

# Create a new DataFrame with the concatenated column
new_df = pd.DataFrame(concatenated_column, columns=['Concatenated_Column'])

# Display the new DataFrame
print(new_df)
