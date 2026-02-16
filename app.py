from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load mood model
model, vectorizer = pickle.load(open("mood_model.pkl", "rb"))

# Load Spotify dataset
songs = pd.read_csv("Spotify_small.csv")

# Create mood column from energy + valence
def detect_mood(row):
    energy = row["Energy"]
    valence = row["Valence"]

    if energy >= 0.75:
        return "Energetic"
    elif valence >= 0.6:
        return "Happy"
    elif valence <= 0.35:
        return "Sad"
    elif energy <= 0.35:
        return "Relaxed"
    else:
        return "Happy"

songs["mood"] = songs.apply(detect_mood, axis=1)

@app.route("/", methods=["GET", "POST"])
def home():

    mood = ""
    recommendations = []

    if request.method == "POST":
        user_text = request.form["feeling"]

        # Predict mood from text
        text_vector = vectorizer.transform([user_text])
        mood = model.predict(text_vector)[0]

        # Filter by predicted mood
        mood_songs = songs[songs["mood"] == mood]

        # If no mood match, use full dataset
        if mood_songs.empty:
            mood_songs = songs

        # Search by track or artist
        search_text = user_text.lower()

        filtered = mood_songs[
            mood_songs["Track"].str.lower().str.contains(search_text, na=False) |
            mood_songs["Artist"].str.lower().str.contains(search_text, na=False)
        ]

        if filtered.empty:
            filtered = mood_songs

        # Take 5 random songs
        if len(filtered) > 10:
            filtered = filtered.sample(10)

        recommendations = filtered[["Track", "Artist", "Url_youtube"]].to_dict("records")

    return render_template("index.html", mood=mood, songs=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
