# 🎬 Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews as **positive** or **negative** using Natural Language Processing (NLP) techniques.

---

## 📌 Project Overview

This project analyzes 500 movie reviews to predict audience sentiment using text classification. It covers the full data science pipeline — from data exploration to model deployment — using Python, Scikit-learn, and visualization libraries.

**Domain:** Natural Language Processing (NLP)  
**Task:** Binary Text Classification  
**Language:** Python 3.x

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `review` | Raw movie review text |
| `sentiment` | `positive` or `negative` label |
| `genre` | Movie genre (Drama, Action, Comedy, etc.) |
| `rating` | IMDb-style rating (1–10) |
| `word_count` | Number of words in the review |

- **Total Reviews:** 500 (250 positive, 250 negative)
- **Genres covered:** Drama, Action, Comedy, Thriller, Romance, Sci-Fi, Horror

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights uncovered:

- **Balanced dataset** — equal positive and negative reviews
- **Ratings strongly correlate with sentiment** — positive reviews cluster 6.5–10, negative 1–5
- **Genre doesn't strongly influence sentiment** — all genres have mixed reviews
- **Review length** is fairly consistent across both classes

### EDA Visualizations
| Plot | Description |
|---|---|
| Sentiment Distribution | Bar chart of class balance |
| Rating Distribution | Histogram by sentiment group |
| Avg Rating by Genre | Grouped bar chart across 7 genres |
| Word Count Distribution | Review length by sentiment |

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
- Lowercasing all text
- Removing punctuation and special characters
- Stripping extra whitespace

### 2. Feature Engineering — TF-IDF
- `max_features = 3,000`
- `ngram_range = (1, 2)` — captures unigrams and bigrams
- English stop words removed

### 3. Models Trained
| Model | Accuracy |
|---|---|
| ✅ Logistic Regression | **100%** |
| Naive Bayes | **100%** |

> Both models achieved strong accuracy. Logistic Regression is preferred for production as it provides probability scores and interpretable coefficients.

---

## 📈 Results

### Most Predictive Positive Words
`masterpiece`, `outstanding`, `breathtaking`, `stunning`, `exceptional`, `brilliant`, `wonderful`, `fantastic`

### Most Predictive Negative Words
`worst`, `boring`, `terrible`, `disappointing`, `painful`, `dull`, `disaster`, `forgettable`

---

## 🗂️ Project Structure

```
movie-sentiment-analysis/
│
├── sentiment_analysis.py     # Main analysis script
├── movie_reviews.csv         # Dataset
├── README.md                 # Project documentation
│
└── plots/
    ├── 00_dashboard.png             # Summary dashboard
    ├── 01_sentiment_distribution.png
    ├── 02_rating_distribution.png
    ├── 03_avg_rating_by_genre.png
    ├── 04_word_count_distribution.png
    ├── 05_confusion_matrices.png
    ├── 06_model_comparison.png
    └── 07_top_words.png
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/MahamMohsin/movie-sentiment-analysis.git
cd movie-sentiment-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the analysis
python sentiment_analysis.py
```

All plots will be saved in the `plots/` folder automatically.

---

## 🔮 Future Improvements

- [ ] Train on real IMDb dataset (50,000 reviews)
- [ ] Implement Deep Learning model (LSTM / BERT)
- [ ] Build an interactive Streamlit web app
- [ ] Add multi-class sentiment (very positive, neutral, very negative)
- [ ] Deploy as a REST API using Flask

---

## 👤 Author

**Maham Mohsin**  
BS Computer Science  
📧 [your-email@example.com]  
🔗 [GitHub](https://github.com/MahamMohsin)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
