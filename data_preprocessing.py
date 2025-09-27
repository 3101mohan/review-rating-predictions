import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Configuration ---
# NOTE: Replace this path with the correct path to your combined CSV file
INPUT_CSV_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\combined_data.csv"
OUTPUT_CSV_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\clean_combined_data.csv"

# --- 1. Load Data ---
def load_data(file_path):
    """Loads the large combined dataset (assuming it was saved as CSV)."""
    print(f"--- 1. Loading data from {file_path} ---")
    try:
        # Use low_memory=False for large files to ensure correct data type inference
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Initial Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Please check your INPUT_CSV_PATH.")
        return None

# --- 2. Handle Duplicates ---
def handle_duplicates(df):
    """Identifies and removes duplicate reviews based on key columns."""
    initial_rows = df.shape[0]
    # Define columns that uniquely identify a review (Review Text, User, Product, Time)
    # CORRECTED 'sort_timestamp' to 'timestamp'
    duplicate_cols = ['text', 'user_id', 'asin', 'timestamp']
    
    # Drop rows where the key fields (text, user, product) are identical
    df.drop_duplicates(subset=duplicate_cols, inplace=True)
    
    rows_dropped = initial_rows - df.shape[0]
    print(f"Removed {rows_dropped} duplicate reviews.")
    return df

# --- 3. Handle Missing Values (Dull Values) ---
def handle_missing_values(df):
    """Fills or drops missing values in critical columns."""
    
    print("\n--- 3. Handling Missing Values ---")
    
    # MANDATORY: Drop rows where the core input (text or rating) is missing
    # Without these, the model cannot be trained.
    df.dropna(subset=['text', 'rating'], inplace=True)
    print(f"Dropped rows with missing 'text' or 'rating'. New Shape: {df.shape}")
    
    # OPTIONAL: Clean up auxiliary fields 
    # Replace missing values in the descriptive fields with an empty string for LLM use
    # This keeps the record but prevents errors during string processing.
    descriptive_cols = ['title', 'features', 'description', 'store', 'details']
    for col in descriptive_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Fill missing average_rating with the overall mean rating for the product
    # This allows us to keep the row while adding a neutral signal
    if 'average_rating' in df.columns:
        mean_rating = df['rating'].mean()
        df['average_rating'] = df['average_rating'].fillna(mean_rating)
        print(f"Filled missing 'average_rating' with overall mean: {mean_rating:.2f}")

    return df

# --- 4. Text Cleaning and Normalization ---
def clean_text(text):
    """Performs standard NLP cleaning steps for review text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove HTML tags (e.g., '<br />')
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove punctuation, except for periods (for sentence structure)
    text = re.sub(r'[^\w\s\.]', ' ', text)
    
    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Note: We skip Stop Word Removal/Stemming here. 
    # LLMs (like the ones we plan to use) are better at handling full, contextual text.
    
    return text

def apply_text_cleaning(df):
    """Applies cleaning function to the relevant text columns."""
    print("\n--- 4. Applying Text Cleaning and Normalization ---")
    
    # Clean the core review text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Create a consolidated feature text (Review + Product Features)
    # This is a critical step for improving LLM performance.
    def create_consolidated_text(row):
        review = row['clean_text']
        # Safely access and clean metadata text if it exists
        features = str(row['features']) if 'features' in row and row['features'] else ''
        description = str(row['description']) if 'description' in row and row['description'] else ''
        
        # Combine them into one coherent text block for the LLM
        # We clean the features and description only slightly before combining
        product_context = f"PRODUCT FEATURES: {clean_text(features)} | DESCRIPTION: {clean_text(description)}"
        
        return f"{review} {product_context}"

    df['consolidated_text'] = df.apply(create_consolidated_text, axis=1)
    
    print("Created 'clean_text' and 'consolidated_text' features.")
    return df

# --- 5. Final Feature Selection and Save ---
def finalize_and_save(df, output_path):
    """Selects final features and saves the clean DataFrame."""
    
    # Define the final features you want for modeling
    final_cols = [
        'rating',             # TARGET VARIABLE
        'user_id',            # User Context
        'asin',               # Product Context
        'clean_text',         # Primary Text Input
        'consolidated_text',  # LLM/Contextual Input
        'average_rating'      # Numerical Feature from Meta Data
    ]
    
    # Filter DataFrame to keep only the necessary columns
    final_df = df[final_cols].copy()
    
    # Check for any remaining nulls in core columns after cleaning
    print("\nFinal Null check on core features:")
    print(final_df[['rating', 'clean_text']].isnull().sum())
    
    # Save the cleaned DataFrame
    final_df.to_csv(output_path, index=False)
    print(f"\n--- SUCCESS! Cleaned data saved to: {output_path} ---")
    print(f"Final Dataset Shape: {final_df.shape}")
    print(final_df.head())
    
    return final_df

# --- Main Execution ---
if __name__ == '__main__':
    
    df = load_data(INPUT_CSV_PATH)

    if df is not None:
        # Step 2: Handle Duplicates
        df = handle_duplicates(df)
        
        # Step 3: Handle Missing Values
        df = handle_missing_values(df)
        
        # Step 4: Text Cleaning and Feature Engineering
        df = apply_text_cleaning(df)
        
        # Step 5: Finalize and Save
        cleaned_df = finalize_and_save(df, OUTPUT_CSV_PATH)
