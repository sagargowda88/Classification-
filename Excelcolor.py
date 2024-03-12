import pandas as pd
from styleframe import StyleFrame

# Load the merged DataFrame
merged_data = pd.read_csv('merged_test_and_rows_not_in_test.csv')

# Create a StyleFrame object from the DataFrame
sf = StyleFrame(merged_data)

# Apply color to the 'Predicted_Label' column based on wrong predictions
for idx, row in sf.iter_rows():
    if row['Predicted_Label'] != row['target_column']:  # Check for wrong predictions
        sf.apply_style_by_indexes([idx], cols_to_style='Predicted_Label', bg_color='red')

# Save the styled DataFrame to an Excel file
sf.to_excel('merged_test_and_rows_not_in_test_styled.xlsx', engine='openpyxl', index=False)
