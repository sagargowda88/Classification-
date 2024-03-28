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
