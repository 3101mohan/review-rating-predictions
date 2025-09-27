import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
import pickle 
import time

# --- Configuration ---
CLEAN_DATA_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\clean_combined_data.csv"
MODEL_OUTPUT_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\ensemble_model.pkl" 
RANDOM_SEED = 42
# CRITICAL: This limits the number of rows used for training to save RAM.
DATA_LIMIT = 200000 
# CRITICAL: Reduce feature count to save memory.
TFIDF_FEATURES = 5000 

# --- 1. Load Data ---
def load_data():
    """Loads the cleaned dataset and applies downsampling limit."""
    print("--- 1. Loading Cleaned Data for Modeling ---")
    try:
        # Load the first N rows (DATA_LIMIT) to save memory
        df = pd.read_csv(CLEAN_DATA_PATH)
        
        if df.shape[0] > DATA_LIMIT:
            # Downsample to the limit
            df = df.sample(n=DATA_LIMIT, random_state=RANDOM_SEED)
            print(f"Data successfully downsampled to {DATA_LIMIT} rows to conserve memory.")
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Clean data file not found at {CLEAN_DATA_PATH}. Please run data_preprocessing.py first.")
        return None

# --- 2. Feature Engineering ---
def create_features(df):
    """
    Creates numerical features from text and categorical data.
    """
    print("\n--- 2. Starting Local Feature Engineering ---")

    y = df['rating'].values

    # 2.1 Text Feature: TF-IDF Vectorization
    # Uses TFIDF_FEATURES (5000) and includes bigrams/trigrams
    tfidf = TfidfVectorizer(max_features=TFIDF_FEATURES, 
                            ngram_range=(1, 3), 
                            stop_words='english')
    X_text = tfidf.fit_transform(df['consolidated_text'])
    print(f"    -> Creating TF-IDF vectors for consolidated_text (Max {TFIDF_FEATURES} features)...")
    print(f"    -> TF-IDF Matrix Shape: {X_text.shape}")

    # 2.2 Numerical Feature: Product Context (average_rating)
    X_num = df[['average_rating']].values

    # 2.3 Categorical Feature: User ID Encoding (User Embeddings)
    user_counts = df['user_id'].value_counts()
    df['user_frequency'] = df['user_id'].map(user_counts)
    
    X_user_freq = np.log1p(df['user_frequency'].values.reshape(-1, 1))

    print("    -> Encoding User IDs based on frequency...")

    # Combine all features into a single sparse matrix
    X_combined = hstack([X_text, X_num, X_user_freq])
    
    print(f"Feature engineering complete. Total Features Shape: {X_combined.shape}")
    return X_combined, y

# --- 3. Training Ensemble Model (Stacking) ---
def train_ensemble(X, y):
    """Trains the Stacking Regressor model."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"\n--- 3. Training Ensemble Stacking Regressor Model ---")
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- LAYER 1: BASE REGRESSORS ---
    base_estimators = [
        # 1. Tuned XGBoost (Powerful and memory efficient)
        ('xgb', XGBRegressor(
            objective='reg:squarederror', n_estimators=100, max_depth=5, 
            learning_rate=0.1, n_jobs=4, random_state=RANDOM_SEED, tree_method='hist'
        )),
        # 2. Tuned Random Forest (Robustness and variance reduction)
        ('rf', RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=4, random_state=RANDOM_SEED
        ))
    ]

    # --- LAYER 2: META-MODEL (Stacking Regressor) ---
    # Ridge is chosen as a fast, effective final meta-learner
    stacking_regressor = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0), # Ridge Regression to combine predictions
        n_jobs=4,
        cv=2 # Use 2-fold cross-validation for speed
    )

    print("    -> Starting Stacking Regressor model training (XGBoost + Random Forest + Ridge)...")
    start_time = time.time()
    stacking_regressor.fit(X_train, y_train)
    end_time = time.time()
    print(f"    -> Training complete. Time taken: {end_time - start_time:.2f} seconds")

    return stacking_regressor, X_test, y_test

# --- 4. Evaluation and Saving ---
def evaluate_and_save_model(model, X_test, y_test):
    """Evaluates the model and saves it."""
    
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 5.0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n--- 4. Model Evaluation ---")
    print(f"Prediction Metric: Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    if rmse < 0.9:
        print("Conclusion: Model performance is excellent! You are very close to or below the 0.8 target.")
    elif rmse < 1.1:
        print("Conclusion: Model performance is strong. The stacking worked well!")
    else:
        print("Conclusion: Performance is acceptable. Further tuning is needed if the target RMSE is critical.")

    # Save the model
    try:
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n--- 5. Model Saved ---")
        print(f"Final model saved to: {MODEL_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    df_clean = load_data()
    
    if df_clean is not None:
        # Step 1: Create Features
        X, y = create_features(df_clean)
        
        # Step 2: Train Model
        ensemble_model, X_test, y_test = train_ensemble(X, y)
        
        # Step 3 & 4: Evaluate and Save Model
        evaluate_and_save_model(ensemble_model, X_test, y_test)
        
        print("\nProject execution complete. The final Ensemble Prediction model has been trained.")
