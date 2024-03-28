def split_and_create_dict(concatenated_string):
    items = concatenated_string.split('\n')
    result = {}
    for item in items:
        key, value = item.split(' = ')
        result[key.strip()] = np.nan if value.strip().lower() == 'null' else value.strip()
    
    # Check if the columns are in a different order than expected
    expected_columns = ['Column A', 'Column B']  # Update this list with your actual column names
    if list(result.keys()) != expected_columns:
        # Reorder the dictionary based on the expected column order
        result = {col: result.get(col, np.nan) for col in expected_columns}
    
    return result


import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('your_file.csv')

# Select the two columns to compare
column1 = 'column_name_1'  # Replace 'column_name_1' with the name of your first column
column2 = 'column_name_2'  # Replace 'column_name_2' with the name of your second column

# Check similarity and calculate accuracy
total_rows = len(df)
matching_rows = (df[column1] == df[column2]).sum()
accuracy = matching_rows / total_rows * 100

print(f"Accuracy between '{column1}' and '{column2}': {accuracy:.2f}%")
