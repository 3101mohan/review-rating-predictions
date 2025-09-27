import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import joblib
from scipy.sparse import hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Configuration ---
# File paths (must match the files saved by final_prediction_model.py)
MODEL_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\ensemble_model.pkl"
CLEAN_DATA_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\data\clean_combined_data.csv"
TFIDF_PATH = r"D:\Intelligent Review Rating Prediction Using AI and LLMs\tfidf_vectorizer.joblib"


# --- 1. Load Model and Assets (Cached for speed) ---

@st.cache_resource
def load_assets():
    """Loads the trained model, TF-IDF vectorizer, and necessary data for feature encoding."""
    try:
        # Load the Stacking Regressor Model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Load the fitted TF-IDF Vectorizer
        tfidf = joblib.load(TFIDF_PATH)

        # Load the full cleaned dataset to calculate User Frequencies
        df_full = pd.read_csv(CLEAN_DATA_PATH)

        # Calculate User Frequencies from the full dataset for prediction logic
        user_counts = df_full['user_id'].value_counts()

        # Initialize VADER for sentiment analysis
        analyzer = SentimentIntensityAnalyzer()

        return model, tfidf, user_counts, analyzer

    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Required file not found. Ensure you have run the final training script correctly. Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"FATAL ERROR during asset loading: {e}")
        st.stop()


# --- 2. Text Preprocessing Function ---

def clean_and_process_text(text):
    """Applies the same cleaning steps as the training script."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. Prediction Function ---

def make_prediction(review_text, avg_rating, user_id, model, tfidf, user_counts):
    """Featurizes input and returns the predicted rating."""
    
    clean_text = clean_and_process_text(review_text)
    X_text = tfidf.transform([clean_text])
    X_num = np.array([avg_rating]).reshape(1, 1)
    user_freq = user_counts.get(user_id, 1)
    X_user_freq = np.log1p(user_freq).reshape(1, 1)
    
    X_combined = hstack([X_text, X_num, X_user_freq])
    y_pred = model.predict(X_combined)
    predicted_rating = np.clip(y_pred[0], 1.0, 5.0)
    
    return predicted_rating

# --- Main Streamlit App Layout ---

st.set_page_config(
    page_title="Intelligent Review Rating Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the resources globally
try:
    model, tfidf, user_counts, analyzer = load_assets()
except Exception as e:
    st.error(f"FATAL ERROR: Could not load assets. Please ensure all files are in the correct directory. Error: {e}")
    st.stop()


st.markdown(f"""
# 🌟 Intelligent Review Rating Predictor
#### Model Status: Stacking Regressor Ensemble ($\mathbf{{RMSE={1.0948}}}$)
---
""")

# --- Input Area ---
with st.container():
    col1, col2 = st.columns([3, 1])

    with col1:
        review_input = st.text_area(
            "1. Enter the Review Text",
            "This product is outstanding! The quality is amazing, and I would buy this again instantly.",
            height=150
        )
    
    with col2:
        avg_rating_input = st.slider(
            "2. Enter Product's Current Average Rating (Meta-Data)",
            min_value=1.0,
            max_value=5.0,
            value=4.2,
            step=0.1
        )
        
        user_id_input = st.text_input(
            "3. Enter Reviewer ID (or leave blank for new user)",
            "AEYORY2AVPMCPDV57CE337YU5LXA"
        )

# --- Prediction Button and Output ---
if st.button('Predict Star Rating', type="primary", use_container_width=True):
    if not review_input:
        st.error("Please enter some review text to make a prediction.")
    else:
        # Pass the user_id_input if provided, otherwise assume a generic new user
        prediction_user_id = user_id_input if user_id_input else "NEW_USER"
        
        # Make the numerical rating prediction
        predicted_rating = make_prediction(
            review_input, avg_rating_input, prediction_user_id, model, tfidf, user_counts
        )

        st.markdown("### 🎯 Prediction Results")
        
        col_pred, col_nlp = st.columns([1, 2])

        with col_pred:
            st.metric(label="Predicted Star Rating", value=f"{predicted_rating:.2f} / 5.0")
            
        with col_nlp:
            st.info(f"""
            **Prediction Details:**
            - **Input Review Length:** {len(review_input.split())} words
            - **Meta-data Context Used:** Product average rating of {avg_rating_input:.1f}
            """)
        
        # --- Display Sentiment and Word Cloud ---
        sentiment_score = analyzer.polarity_scores(review_input)
        if sentiment_score['compound'] >= 0.05:
            sentiment_label = "Positive 😊"
        elif sentiment_score['compound'] <= -0.05:
            sentiment_label = "Negative 😠"
        else:
            sentiment_label = "Neutral 😐"

        st.markdown("#### Natural Language Analysis")
        st.metric(label="VADER Sentiment Analysis", value=sentiment_label)

        # Generate and display Word Cloud
        st.markdown("##### Word Cloud of the Review")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_input)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


st.markdown("---")
st.caption("This model was built using local ensemble methods (Stacking Regressor) on the Amazon Reviews dataset.")