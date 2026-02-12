# loan_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Load and Clean Data ---
def load_and_prepare_data(file_path='lending_data.csv'):
    df = pd.read_csv(file_path)

    # Simplify problem: only use relevant features for demonstration
    selected_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'term', 'grade', 'purpose', 'home_ownership', 'loan_status']
    df = df[selected_cols].dropna()

    # Filter to 'Fully Paid' and 'Charged Off'
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # Encode target
    df['default'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # Encode categorical columns
    le = LabelEncoder()
    for col in ['term', 'grade', 'purpose', 'home_ownership']:
        df[col] = le.fit_transform(df[col])

    X = df.drop(['loan_status', 'default'], axis=1)
    y = df['default']
    
    return X, y

# --- Train Model ---
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Paid', 'Default'], yticklabels=['Paid', 'Default'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model

# --- Main ---
if __name__ == "__main__":
    try:
        X, y = load_and_prepare_data()
        model = train_model(X, y)
    except FileNotFoundError:
        print("Please download the Lending Club dataset from https://www.kaggle.com/wordsforthewise/lending-club and save it as 'lending_data.csv'")
