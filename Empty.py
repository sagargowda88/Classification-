import pandas as pd

# Load CSV and Excel files into pandas DataFrames
csv_df = pd.read_csv('your_csv_file.csv')
excel_df = pd.read_excel('your_excel_file.xlsx')

# Iterate over each row in the Excel DataFrame
for index, row in excel_df.iterrows():
    rule_id = row['rule_id']
    dataset_sql_query = row['dataset_sql_query']
    
    # Check if 'dataset_sql_query' is not empty
    if pd.notna(dataset_sql_query):
        # Update corresponding row in CSV DataFrame where 'rule_id' matches
        csv_df.loc[csv_df['rule_id'] == rule_id, 'dataset_sql_query'] = dataset_sql_query

# Save the updated CSV
csv_df.to_csv('updated_csv_file.csv', index=False)
