import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')

# Encode labels (ham: 0, spam: 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(model, 'detector/spam_detector_model.joblib')
joblib.dump(vectorizer, 'detector/vectorizer.joblib')
