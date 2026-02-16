import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("mood_data.csv")

X = data["text"]
y = data["mood"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

pickle.dump((model, vectorizer), open("mood_model.pkl", "wb"))

print("Model trained successfully!")
