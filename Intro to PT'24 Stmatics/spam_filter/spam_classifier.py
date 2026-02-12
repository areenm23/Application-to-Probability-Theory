# spam_classifier.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import os

nltk.download('stopwords')

# --- Load Dataset ---
def load_data(file_path='spam_dataset.csv'):
    df = pd.read_csv(file_path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

# --- Preprocess Text ---
def clean_text(msg):
    msg = ''.join([char for char in msg if char not in string.punctuation])
    words = msg.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# --- Vectorize & Train Model ---
def train_model(df):
    df['cleaned'] = df['message'].apply(clean_text)
    X = df['cleaned']
    y = df['label'].map({'ham': 0, 'spam': 1})  # Binary labels

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.25, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model, vectorizer

# --- Main ---
if __name__ == "__main__":
    # Load and train
    if not os.path.exists("spam_dataset.csv"):
        print("Please download the dataset from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection and save as 'spam_dataset.csv'")
    else:
        df = load_data()
        model, vectorizer = train_model(df)
