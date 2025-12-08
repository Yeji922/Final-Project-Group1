# ------------------------------------------------------------
# main.py : Suicide & Depression Detection - Baseline Pipeline

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from model import vectorize_text, train_lr, evaluate_model, show_top_words, save_model


# 1. Setup & Load Dataset

nltk.download('stopwords')
nltk.download('wordnet')
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

DATA_PATH = "Suicide_Detection.csv"
df = pd.read_csv(DATA_PATH)

# Adjust columns if needed
if 'post_text' in df.columns:
    df['text'] = df['title'].fillna('') + ' ' + df['post_text'].fillna('')
if 'class' in df.columns:
    df['label'] = df['class']
df = df[['text', 'label']].dropna().reset_index(drop=True)

print("Dataset size:", df.shape)
print(df['label'].value_counts())

# ------------------------------------------------------------
# 2. Text Cleaning
# ------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# ------------------------------------------------------------
# 3. Stratified Train/Val/Test Split
# ------------------------------------------------------------
X = df['clean_text']
y = df['label']

# Step 1: train + test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 2: train + val
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 4. EDA

plt.figure(figsize=(6,4))
sns.countplot(y=y, order=y.value_counts().index, palette="cool")
plt.title("Label Distribution")
plt.show()


# 5. Vectorize Text

tfidf, X_train_tfidf, X_val_tfidf, X_test_tfidf = vectorize_text(
    X_train, X_val, X_test
)


# 6. Train & Evaluate

lr_model = train_lr(X_train_tfidf, y_train)

evaluate_model(lr_model, X_val_tfidf, y_val, title="Validation Set")
evaluate_model(lr_model, X_test_tfidf, y_test, title="Test Set")


# 7. Inspect Important Words & Save Model

show_top_words(lr_model, tfidf, top_n=10)
save_model(lr_model, tfidf, path_prefix="baseline")



