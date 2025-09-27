# 🌟 Intelligent Review Rating Prediction (IRRP)

**End-to-End Machine Learning Pipeline with Hybrid Feature Engineering**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-brightgreen.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

This project demonstrates an **end-to-end Data Science pipeline** to predict **Amazon product star ratings** based on **review text** and **metadata**.
The deployed model is a **memory-optimized XGBoost Regressor** trained on **200,000 reviews**, combining **text features (TF-IDF)** with **contextual numerical features**.

---

## 🎯 Final Project Results

| Metric              | Model Achieved                                         | Goal (High-Performance) |
| ------------------- | ------------------------------------------------------ | ----------------------- |
| **Final RMSE**      | **1.0948**                                             | < **0.90**              |
| **Model Used**      | Stacking Regressor *(XGBoost + Random Forest + Ridge)* | —                       |
| **Prediction Time** | Sub-second (real-time in demo)                         | —                       |

✅ The **low RMSE** demonstrates the effectiveness of **hybrid feature engineering**, blending text + context features.

---

## ⚙️ Project Structure

```
📂 Intelligent Review Rating Prediction
│── data/
│   ├── raw/                  # Original JSONL files
│   ├── clean_combined_data.csv
│
│── data_preprocessing.py      # Cleans raw data → creates clean CSV
│── final_prediction_model.py  # Feature engineering, trains Stacking Regressor, saves model
│── streamlit.py               # Interactive web demo (front + backend)
│── ensemble_model.pkl         # Saved trained ensemble model
│── tfidf_vectorizer.joblib    # Saved TF-IDF vectorizer
```

---

## 🚀 How to Run the Demo Application

### 1️⃣ Install Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit nltk wordcloud
```

> **Note:** Run once in Python to download NLTK’s sentiment lexicon:

```python
import nltk
nltk.download('vader_lexicon')
```

---

### 2️⃣ Train the Model

Before launching the demo, run the training script to generate model artifacts:

```bash
python final_prediction_model.py
```

This will save:

* `ensemble_model.pkl` → trained model
* `tfidf_vectorizer.joblib` → fitted vectorizer

---

### 3️⃣ Launch the Streamlit Web App

```bash
streamlit run streamlit.py
```

A browser window will open with:

* **Predicted Rating** (1–5 stars)
* **VADER Sentiment Score**
* **Word Cloud** of review highlights

---

## 💻 Modeling Details

| Feature Type          | Source Column       | Method                                                               |
| --------------------- | ------------------- | -------------------------------------------------------------------- |
| **Text Features**     | `consolidated_text` | TF-IDF (max 5000 features, ngram_range=(1,3))                        |
| **Contextual**        | `average_rating`    | Direct numerical input                                               |
| **User Bias Feature** | `user_id`           | Log-transformed review frequency                                     |
| **Final Predictor**   | Combined features   | **Stacking Regressor** *(XGBoost + RandomForest, meta-model: Ridge)* |

✅ Designed for **low-resource environments (8GB RAM)** with **memory-efficient TF-IDF** and optimized XGBoost settings.

---

## 📌 Key Highlights

* 🔄 **End-to-end ML pipeline** (data preprocessing → training → deployment)
* 🧩 **Hybrid feature engineering** (text + contextual + user bias)
* ⚡ **Sub-second inference** via Streamlit app
* 📊 **Explainability features**: VADER sentiment + Word Cloud

---

## 🛠️ Tech Stack

* **Python** (pandas, numpy, scikit-learn, nltk)
* **Models**: XGBoost, RandomForest, Ridge
* **Deployment**: Streamlit
* **Visualization**: WordCloud, Matplotlib

---

## 📬 Future Improvements

* [ ] Improve RMSE < 0.90 with deep learning models (BERT embeddings)
* [ ] Add more metadata features (category, product title)
* [ ] Deploy with Docker + CI/CD pipeline

---

💡 *Developed as part of an advanced ML pipeline project showcasing hybrid feature engineering for real-world e-commerce data.*
