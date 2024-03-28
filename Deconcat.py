import pandas as pd
import numpy as np

# Read the concatenated CSV file with predictions into a DataFrame
concatenated_df = pd.read_csv('concatenated_file_with_predictions.csv')

# Function to split concatenated strings into dictionary items
def split_and_create_dict(concatenated_string):
    items = concatenated_string.split('\n')
    result = {}
    for item in items:
        key, value = item.split(' = ')
        result[key.strip()] = np.nan if value.strip().lower() == 'null' else value.strip()
    return result

# Split the concatenated column back to the original format
deconcatenated_data = concatenated_df['Concatenated_Column'].apply(split_and_create_dict)

# Create a DataFrame from the deconcatenated data
deconcatenated_df = pd.DataFrame(deconcatenated_data.tolist())

# Add predictions from concatenated_df to deconcatenated_df
deconcatenated_df['Prediction'] = concatenated_df['Prediction']

# Save the deconcatenated DataFrame with predictions to a new CSV file
deconcatenated_df.to_csv('deconcatenated_file_with_predictions.csv', index=False)
