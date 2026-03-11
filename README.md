# 🎬 Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews as **positive** or **negative** using Natural Language Processing (NLP) techniques.

---

## 📌 Project Overview

This project analyzes **50,000 real IMDb movie reviews** to predict audience sentiment using text classification. It covers the full data science pipeline — from data exploration to model evaluation — using Python, Scikit-learn, and visualization libraries.

**Domain:** Natural Language Processing (NLP)  
**Task:** Binary Text Classification  
**Language:** Python 3.x

---

## 📊 Dataset

**Source:** [IMDb Dataset of 50K Movie Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Feature | Description |
|---|---|
| `review` | Raw movie review text |
| `sentiment` | `positive` or `negative` label |
| `word_count` | Number of words in the review (engineered) |
| `clean_review` | Preprocessed review text (engineered) |

- **Total Reviews:** 50,000
- **Positive Reviews:** 25,000 (50%)
- **Negative Reviews:** 25,000 (50%)
- **Train / Test Split:** 80% / 20% → 40,000 train, 10,000 test

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights uncovered:

- **Perfectly balanced dataset** — equal positive and negative reviews
- **Average review length** is similar across both sentiment classes
- **Longer reviews** tend to appear slightly more in the negative class
- Real-world reviews contain HTML tags (`<br />`) requiring cleaning

### EDA Visualizations

| Plot | Description |
|---|---|
| Sentiment Distribution | Bar chart of class balance across 50K reviews |
| Word Count Distribution | Histogram of review lengths by sentiment |
| Avg Review Length | Average word count per sentiment class |

---

## 🛠️ Tech Stack

```
Python 3.x
├── pandas          — data manipulation
├── numpy           — numerical operations
├── matplotlib      — static visualizations
├── seaborn         — styled statistical plots
├── scikit-learn    — ML models & TF-IDF
│   ├── TfidfVectorizer
│   ├── LogisticRegression
│   └── MultinomialNB
└── re              — text preprocessing (regex)
```

---

## ⚙️ Methodology

### 1. Text Preprocessing
- Removing HTML tags (e.g. `<br />`) present in raw IMDb data
- Lowercasing all text
- Removing punctuation and special characters
- Stripping extra whitespace

### 2. Feature Engineering — TF-IDF
- `max_features = 5,000`
- `ngram_range = (1, 2)` — captures unigrams and bigrams
- English stop words removed

### 3. Models Trained

| Model | Accuracy |
|---|---|
| ✅ Logistic Regression | **88.76%** |
| Naive Bayes | 85.52% |

> Logistic Regression outperforms Naive Bayes and is preferred as the final model due to its higher accuracy and interpretable coefficients.

---

## 📈 Results

### Most Predictive Positive Words
`excellent`, `brilliant`, `wonderful`, `amazing`, `perfect`, `loved`, `fantastic`, `great`

### Most Predictive Negative Words
`worst`, `awful`, `terrible`, `boring`, `waste`, `bad`, `horrible`, `disappointing`

---

## 🗂️ Project Structure

```
movie-sentiment-analysis/
│
├── sentiment_analysis.py     # Main analysis script
├── IMDB Dataset.csv          # Real IMDb dataset (50K reviews)
├── README.md                 # Project documentation
│
└── plots/
    ├── 00_dashboard.png             # Summary dashboard
    ├── 01_sentiment_distribution.png
    ├── 02_word_count_distribution.png
    ├── 03_avg_word_count.png
    ├── 04_confusion_matrices.png
    ├── 05_model_comparison.png
    └── 06_top_words.png
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/MahamMohsin/Movie-Sentiment-Analysis.git
cd movie-sentiment-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the analysis
python sentiment_analysis.py
```

All plots will be saved in the `plots/` folder automatically.

---

This project is open-source and available under the [MIT License](LICENSE).
