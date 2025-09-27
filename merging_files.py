import pandas as pd
import json

# Define your file paths
reviews_file_path = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\All_Beauty.jsonl"
meta_file_path = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\meta_All_Beauty.jsonl"

def load_jsonl_to_dict(file_path, key_field):
    """
    Reads a JSONL file and returns a dictionary for fast lookups.
    This is ideal for metadata since you can look up product details by ASIN.
    """
    data_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    data_dict[record.get(key_field)] = record
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(data_dict)} records from {file_path}")
        return data_dict
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def process_and_combine_data(reviews_path, meta_path):
    """
    Processes the large review file line-by-line and combines it with metadata.
    """
    # Load metadata into a dictionary for quick access
    meta_dict = load_jsonl_to_dict(meta_path, 'parent_asin')

    if meta_dict is None:
        return None

    combined_data = []
    
    # Process the reviews file line by line
    try:
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review_record = json.loads(line)
                    
                    # Use .get() to avoid errors if key doesn't exist
                    product_id = review_record.get('asin')
                    
                    # Look up metadata using the product ID
                    meta_record = meta_dict.get(product_id, {})
                    
                    # Combine the review and meta data
                    combined_record = {**review_record, **meta_record}
                    
                    combined_data.append(combined_record)
                    
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: The file at {reviews_path} was not found.")
        return None

    # Create the final DataFrame
    return pd.DataFrame(combined_data)

# Run the function to get your final DataFrame
combined_df = process_and_combine_data(reviews_file_path, meta_file_path)

if combined_df is not None:
    print("\nSuccessfully created combined DataFrame.")
    print("DataFrame shape:", combined_df.shape)
    print("Columns:", combined_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(combined_df.head())
    
    # Optional: Save to a new CSV file
    output_csv_path = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\combined_data.csv"
    combined_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved combined data to {output_csv_path}")