# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# 2. LOAD REAL DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('IMDB Dataset.csv')
print("✅ Dataset loaded:", df.shape)
print(df['sentiment'].value_counts())
print(df.head(3))

os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {'positive': '#2ecc71', 'negative': '#e74c3c'}

df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

# ── 3a. Sentiment Distribution ──
fig, ax = plt.subplots(figsize=(6, 5))
counts = df['sentiment'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[s] for s in counts.index],
              edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{val:,}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_title('Sentiment Distribution (50K Reviews)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Sentiment', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig('plots/01_sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 1: Sentiment Distribution saved")

# ── 3b. Word Count Distribution ──
fig, ax = plt.subplots(figsize=(8, 5))
for s, color in COLORS.items():
    subset = df[df['sentiment'] == s]['word_count']
    ax.hist(subset, bins=50, alpha=0.7, label=s.capitalize(),
            color=color, edgecolor='white')
ax.set_title('Review Word Count by Sentiment', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Word Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_xlim(0, 1000)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/02_word_count_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 2: Word Count Distribution saved")

# ── 3c. Average Word Count by Sentiment ──
fig, ax = plt.subplots(figsize=(6, 5))
avg_wc = df.groupby('sentiment')['word_count'].mean()
bars = ax.bar(avg_wc.index, avg_wc.values,
              color=[COLORS[s] for s in avg_wc.index],
              edgecolor='white', width=0.5)
for bar, val in zip(bars, avg_wc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f} words', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Average Review Length by Sentiment', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Sentiment', fontsize=12)
ax.set_ylabel('Average Word Count', fontsize=12)
ax.set_ylim(0, max(avg_wc.values) * 1.2)
plt.tight_layout()
plt.savefig('plots/03_avg_word_count.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 3: Average Word Count saved")

# ─────────────────────────────────────────────
# 4. TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text):
    text = re.sub(r'<.*?>', '', str(text))       # remove HTML tags
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n⏳ Preprocessing text (this may take a minute)...")
df['clean_review'] = df['review'].apply(preprocess)
print("✅ Text preprocessing complete")

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING (TF-IDF)
# ─────────────────────────────────────────────
X = df['clean_review']
y = (df['sentiment'] == 'positive').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n⏳ Fitting TF-IDF on {len(X_train):,} reviews...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"✅ TF-IDF matrix: {X_train_vec.shape[0]:,} train / {X_test_vec.shape[0]:,} test samples")

# ─────────────────────────────────────────────
# 6. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes':         MultinomialNB()
}

results = {}
for name, model in models.items():
    print(f"\n⏳ Training {name}...")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc}
    print(f"{'='*45}")
    print(f"  {name}  |  Accuracy: {acc*100:.2f}%")
    print(f"{'='*45}")
    print(classification_report(y_test, preds, target_names=['Negative', 'Positive']))

# ── 6a. Confusion Matrices ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, colorbar=False, cmap='RdYlGn')
    ax.set_title(f'{name}\nAccuracy: {res["accuracy"]*100:.2f}%',
                 fontsize=13, fontweight='bold')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot 4: Confusion Matrices saved")

# ── 6b. Model Accuracy Comparison ──
fig, ax = plt.subplots(figsize=(6, 5))
names = list(results.keys())
accs  = [r['accuracy'] * 100 for r in results.values()]
bars = ax.bar(names, accs, color=['#3498db', '#9b59b6'], edgecolor='white', width=0.4)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 115)
ax.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plots/05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 5: Model Comparison saved")

# ── 6c. Top Predictive Words ──
lr_model = results['Logistic Regression']['model']
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0]

top_pos_idx = coefs.argsort()[-15:][::-1]
top_neg_idx = coefs.argsort()[:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh([feature_names[i] for i in top_pos_idx[::-1]],
             [coefs[i] for i in top_pos_idx[::-1]],
             color='#2ecc71', edgecolor='white')
axes[0].set_title('Top 15 Positive Words', fontsize=13, fontweight='bold')
axes[0].set_xlabel('TF-IDF Coefficient')

axes[1].barh([feature_names[i] for i in top_neg_idx],
             [coefs[i] for i in top_neg_idx],
             color='#e74c3c', edgecolor='white')
axes[1].set_title('Top 15 Negative Words', fontsize=13, fontweight='bold')
axes[1].set_xlabel('TF-IDF Coefficient')

plt.suptitle('Most Predictive Words (Logistic Regression)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/06_top_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot 6: Top Predictive Words saved")

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

best_acc = max(accs)
best_model_name = names[accs.index(best_acc)]
metrics = [
    ('Total Reviews', '50,000'),
    ('Positive Reviews', '25,000 (50%)'),
    ('Negative Reviews', '25,000 (50%)'),
    ('Best Model', best_model_name.replace(' ', '\n')),
    ('Best Accuracy', f'{best_acc:.2f}%'),
    ('Features (TF-IDF)', '5,000'),
]
box_colors = ['#16213e','#0f3460','#533483','#1a1a2e','#16213e','#0f3460']
for i, (label, val) in enumerate(metrics):
    x = (i % 3) * 0.34 + 0.01
    y = 0.65 if i < 3 else 0.38
    ax = fig.add_axes([x, y, 0.30, 0.20])
    ax.set_facecolor(box_colors[i])
    ax.axis('off')
    ax.text(0.5, 0.65, val, ha='center', va='center', fontsize=15,
            fontweight='bold', color='#f0e6ff', transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=10,
            color='#aaaacc', transform=ax.transAxes)

# Mini accuracy bar
ax_acc = fig.add_axes([0.01, 0.03, 0.45, 0.30])
ax_acc.set_facecolor('#16213e')
ax_acc.barh(names, accs, color=['#3498db','#9b59b6'])
ax_acc.set_xlim(0, 110)
for i, v in enumerate(accs):
    ax_acc.text(v + 0.5, i, f'{v:.2f}%', va='center', color='white', fontsize=11)
ax_acc.set_title('Model Accuracy', color='white', fontsize=12, fontweight='bold')
ax_acc.tick_params(colors='white')
ax_acc.set_facecolor('#16213e')
for spine in ax_acc.spines.values():
    spine.set_edgecolor('#333355')

# Pie
ax_pie = fig.add_axes([0.52, 0.03, 0.45, 0.30])
ax_pie.set_facecolor('#16213e')
ax_pie.pie([25000, 25000], labels=['Positive','Negative'],
           colors=['#2ecc71','#e74c3c'], autopct='%1.0f%%',
           startangle=90, textprops={'color':'white', 'fontsize':12})
ax_pie.set_title('Sentiment Split', color='white', fontsize=12, fontweight='bold')

plt.savefig('plots/00_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("✅ Plot 7: Dashboard saved")
print("\nALL DONE! Check the plots/ folder.")
