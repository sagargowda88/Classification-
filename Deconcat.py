import pandas as pd

# Read the concatenated CSV file with predictions into a DataFrame
concatenated_df = pd.read_csv('concatenated_file_with_predictions.csv')

# Split the concatenated column back to the original format
deconcatenated_data = concatenated_df['Concatenated_Column'].str.split('\n').apply(lambda x: dict(item.split('=') for item in x))

# Create a DataFrame from the deconcatenated data
deconcatenated_df = pd.DataFrame(deconcatenated_data.tolist())

# Add predictions from concatenated_df to deconcatenated_df
deconcatenated_df['Prediction'] = concatenated_df['Prediction']

# Save the deconcatenated DataFrame with predictions to a new CSV file
deconcatenated_df.to_csv('deconcatenated_file_with_predictions.csv', index=False)
