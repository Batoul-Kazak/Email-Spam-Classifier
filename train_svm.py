import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# Convert labels: spam=1, ham=0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Vectorize text (convert words to numbers)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear', probability=True)  # linear works great for text
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save model + vectorizer
joblib.dump(model, 'spam_svm.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')