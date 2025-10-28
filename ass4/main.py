import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("movie_reviews.csv")  # your dataset file
except FileNotFoundError:
    print("Dataset not found! Using sample data...")
    data = {
        'review': [
            "I really loved this movie, it was amazing!",
            "Absolutely terrible. Waste of time.",
            "Brilliant acting and nice storyline.",
            "Worst movie I have ever seen",
            "Enjoyed it a lot, great music too!",
            "Horrible script and bad direction."
        ],
        'sentiment': ["positive", "negative", "positive", "negative", "positive", "negative"]
    }
    df = pd.DataFrame(data)

# -----------------------------
# Preprocessing
# -----------------------------
stop_words = stopwords.words("english")

# Split features and labels
X = df['review']
y = df['sentiment']

# Convert text to numeric using Bag-of-Words
vectorizer = CountVectorizer(stop_words=stop_words)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# -----------------------------
# Naive Bayes Classification
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔧 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Test on new reviews
# -----------------------------
test_reviews = [
    "I hated this movie, it was awful",
    "What a fantastic film, I would watch it again!",
]

test_vectorized = vectorizer.transform(test_reviews)
predictions = model.predict(test_vectorized)

print("\n🔮 Sample Predictions:")
for review, sentiment in zip(test_reviews, predictions):
    print(f"Review: \"{review}\" → Sentiment: {sentiment}")
