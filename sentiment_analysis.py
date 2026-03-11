# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 2. GENERATE REALISTIC DATASET
# ─────────────────────────────────────────────
np.random.seed(42)

positive_reviews = [
    "An absolute masterpiece. The storytelling was breathtaking and emotional.",
    "One of the best films I have ever seen. Truly outstanding performance.",
    "The cinematography was stunning and the plot was incredibly engaging.",
    "A beautiful and moving story that will stay with you for days.",
    "Exceptional acting and a gripping storyline from start to finish.",
    "Loved every minute of it. The director did an amazing job.",
    "Brilliant film with fantastic dialogue and incredible direction.",
    "Heartwarming and inspiring. Left me feeling genuinely uplifted.",
    "A wonderful cinematic experience. Highly recommend to everyone.",
    "The best movie of the year. Perfectly crafted and deeply touching.",
    "Superb performances all around. The film exceeded all expectations.",
    "Visually spectacular with an emotionally powerful narrative.",
    "A triumph of filmmaking. Every scene is crafted with care.",
    "Deeply moving and beautifully acted. An unforgettable journey.",
    "Thrilling from start to finish with a satisfying conclusion.",
    "The chemistry between the actors was undeniable and charming.",
    "Crisp writing and sharp direction made this a joy to watch.",
    "A rare gem that combines action and heart seamlessly.",
    "The soundtrack was phenomenal and perfectly set the mood.",
    "A compelling and thought-provoking film with stellar acting.",
    "Absolutely loved the plot twists. Kept me hooked throughout.",
    "Refreshingly original story with fantastic world-building.",
    "The pacing was perfect and the characters were well developed.",
    "An emotional rollercoaster that hit every beat correctly.",
    "Outstanding visual effects that never distracted from the story.",
]

negative_reviews = [
    "A complete waste of time. The plot made absolutely no sense.",
    "Terribly boring film with zero character development whatsoever.",
    "The worst movie I have seen in years. Deeply disappointing.",
    "Poor acting, weak script, and a completely predictable ending.",
    "I fell asleep halfway through. Nothing interesting happened.",
    "A total disaster from the opening scene to the painful finale.",
    "Confusing storyline with no resolution. Felt cheated watching this.",
    "The special effects were laughably bad and distracting.",
    "Dull and lifeless. The characters were completely forgettable.",
    "Painful to sit through. The dialogue was cringeworthy throughout.",
    "A massive disappointment given the talented cast involved.",
    "Rushed plot with too many unresolved storylines left hanging.",
    "Could not connect with any of the characters at all.",
    "The writing felt lazy and uninspired throughout the whole film.",
    "Tedious pacing made this feel three times longer than it was.",
    "A hollow story with absolutely nothing original to offer.",
    "The worst script I have encountered in recent memory.",
    "Loud, chaotic, and exhausting without any emotional depth.",
    "The director clearly had no vision for this project.",
    "Absolutely hated the ending. It undid everything built up.",
    "Cheap production values that undermined the entire experience.",
    "Overly long and self-indulgent with very little payoff.",
    "Flat performances from a cast that deserved much better.",
    "Incoherent editing made it impossible to follow the narrative.",
    "A formulaic and forgettable film that offers nothing new.",
]

# Expand to 500 samples with variation
def augment(reviews, n=250):
    result = []
    fillers = ["", "Honestly, ", "Overall, ", "In my opinion, ", "To be fair, "]
    for i in range(n):
        base = reviews[i % len(reviews)]
        filler = fillers[i % len(fillers)]
        result.append(filler + base)
    return result

pos = augment(positive_reviews, 250)
neg = augment(negative_reviews, 250)

reviews = pos + neg
sentiments = ['positive'] * 250 + ['negative'] * 250

# Add metadata
genres = np.random.choice(['Drama','Action','Comedy','Thriller','Romance','Sci-Fi','Horror'], 500)
ratings = [round(np.random.uniform(6.5, 10), 1) if s == 'positive'
           else round(np.random.uniform(1.0, 5.0), 1) for s in sentiments]
review_lengths = [len(r.split()) for r in reviews]

df = pd.DataFrame({
    'review': reviews,
    'sentiment': sentiments,
    'genre': genres,
    'rating': ratings,
    'word_count': review_lengths
})

df.to_csv('movie_reviews.csv', index=False)
print("✅ Dataset created:", df.shape)
print(df.head(3))

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {'positive': '#2ecc71', 'negative': '#e74c3c'}
os.makedirs("plots", exist_ok=True)

# ── 3a. Sentiment Distribution ──
fig, ax = plt.subplots(figsize=(6, 5))
counts = df['sentiment'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[s] for s in counts.index],
              edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_title('Sentiment Distribution', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Sentiment', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig('plots/01_sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 1: Sentiment Distribution saved")

# ── 3b. Rating Distribution by Sentiment ──
fig, ax = plt.subplots(figsize=(8, 5))
for s, color in COLORS.items():
    subset = df[df['sentiment'] == s]['rating']
    ax.hist(subset, bins=15, alpha=0.7, label=s.capitalize(),
            color=color, edgecolor='white')
ax.set_title('Rating Distribution by Sentiment', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Movie Rating', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/02_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 2: Rating Distribution saved")

# ── 3c. Average Rating by Genre ──
fig, ax = plt.subplots(figsize=(9, 5))
genre_avg = df.groupby(['genre','sentiment'])['rating'].mean().unstack()
genre_avg.plot(kind='bar', ax=ax,
               color=[COLORS['negative'], COLORS['positive']],
               edgecolor='white', width=0.6)
ax.set_title('Average Rating by Genre & Sentiment', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Genre', fontsize=12)
ax.set_ylabel('Average Rating', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.legend(title='Sentiment', fontsize=10)
plt.tight_layout()
plt.savefig('plots/03_avg_rating_by_genre.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 3: Average Rating by Genre saved")

# ── 3d. Word Count Distribution ──
fig, ax = plt.subplots(figsize=(8, 5))
for s, color in COLORS.items():
    subset = df[df['sentiment'] == s]['word_count']
    ax.hist(subset, bins=20, alpha=0.7, label=s.capitalize(),
            color=color, edgecolor='white')
ax.set_title('Review Word Count by Sentiment', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Word Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/04_word_count_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 4: Word Count Distribution saved")

# ─────────────────────────────────────────────
# 4. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(preprocess)
print("\n✅ Text preprocessing complete")
print("Sample:", df['clean_review'].iloc[0])

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING (TF-IDF)
# ─────────────────────────────────────────────

X = df['clean_review']
y = (df['sentiment'] == 'positive').astype(int)  # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"\n✅ TF-IDF matrix: {X_train_vec.shape[0]} train / {X_test_vec.shape[0]} test")

# ─────────────────────────────────────────────
# 6. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes':         MultinomialNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc}
    print(f"\n{'='*45}")
    print(f"  {name}  |  Accuracy: {acc*100:.2f}%")
    print(f"{'='*45}")
    print(classification_report(y_test, preds, target_names=['Negative','Positive']))

# ── 6a. Confusion Matrices Side-by-Side ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negative','Positive'])
    disp.plot(ax=ax, colorbar=False, cmap='RdYlGn')
    ax.set_title(f'{name}\nAccuracy: {res["accuracy"]*100:.1f}%',
                 fontsize=13, fontweight='bold')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/05_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot 5: Confusion Matrices saved")

# ── 6b. Model Accuracy Comparison ──
fig, ax = plt.subplots(figsize=(6, 5))
names = list(results.keys())
accs  = [r['accuracy'] * 100 for r in results.values()]
bar_colors = ['#3498db', '#9b59b6']
bars = ax.bar(names, accs, color=bar_colors, edgecolor='white', width=0.4)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 115)
ax.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plots/06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 6: Model Comparison saved")

# ── 6c. Top Predictive Words ──
lr_model = results['Logistic Regression']['model']
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0]

top_pos_idx = coefs.argsort()[-15:][::-1]
top_neg_idx = coefs.argsort()[:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Positive words
axes[0].barh([feature_names[i] for i in top_pos_idx[::-1]],
             [coefs[i] for i in top_pos_idx[::-1]],
             color='#2ecc71', edgecolor='white')
axes[0].set_title('Top 15 Positive Words', fontsize=13, fontweight='bold')
axes[0].set_xlabel('TF-IDF Coefficient')

# Negative words
axes[1].barh([feature_names[i] for i in top_neg_idx],
             [coefs[i] for i in top_neg_idx],
             color='#e74c3c', edgecolor='white')
axes[1].set_title('Top 15 Negative Words', fontsize=13, fontweight='bold')
axes[1].set_xlabel('TF-IDF Coefficient')

plt.suptitle('Most Predictive Words (Logistic Regression)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/07_top_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 7: Top Predictive Words saved")

# ─────────────────────────────────────────────
# 7. SUMMARY DASHBOARD
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')

ax_title = fig.add_axes([0.0, 0.88, 1.0, 0.12])
ax_title.axis('off')
ax_title.text(0.5, 0.5, '🎬  Movie Review Sentiment Analysis — Summary Dashboard',
              ha='center', va='center', fontsize=18, fontweight='bold',
              color='white', transform=ax_title.transAxes)

# Metric boxes
metrics = [
    ('Total Reviews', '500'),
    ('Positive Reviews', '250 (50%)'),
    ('Negative Reviews', '250 (50%)'),
    ('Best Model', 'Logistic Reg.'),
    ('Best Accuracy', f'{max(accs):.1f}%'),
    ('Features (TF-IDF)', '3,000'),
]
box_colors = ['#16213e','#0f3460','#533483','#1a1a2e','#16213e','#0f3460']
for i, (label, val) in enumerate(metrics):
    x = (i % 3) * 0.34 + 0.01
    y = 0.65 if i < 3 else 0.38
    ax = fig.add_axes([x, y, 0.30, 0.20])
    ax.set_facecolor(box_colors[i])
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e0e0e0')
    ax.text(0.5, 0.65, val, ha='center', va='center', fontsize=17,
            fontweight='bold', color='#f0e6ff', transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=10,
            color='#aaaacc', transform=ax.transAxes)

# Mini genre chart
ax_genre = fig.add_axes([0.01, 0.03, 0.45, 0.30])
ax_genre.set_facecolor('#16213e')
genre_counts = df['genre'].value_counts()
ax_genre.barh(genre_counts.index, genre_counts.values,
              color=sns.color_palette("Set2", len(genre_counts)))
ax_genre.set_title('Reviews by Genre', color='white', fontsize=12, fontweight='bold')
ax_genre.tick_params(colors='white')
ax_genre.set_facecolor('#16213e')
for spine in ax_genre.spines.values():
    spine.set_edgecolor('#333355')

# Mini sentiment chart
ax_pie = fig.add_axes([0.52, 0.03, 0.45, 0.30])
ax_pie.set_facecolor('#16213e')
ax_pie.pie([250, 250], labels=['Positive','Negative'],
           colors=['#2ecc71','#e74c3c'], autopct='%1.0f%%',
           startangle=90, textprops={'color':'white', 'fontsize':12})
ax_pie.set_title('Sentiment Split', color='white', fontsize=12, fontweight='bold')

plt.savefig('plots/00_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("✅ Plot 8: Summary Dashboard saved")
print("\n✅ ALL PLOTS GENERATED SUCCESSFULLY")
print("📁 Saved in: plots/")
