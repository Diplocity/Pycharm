from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Expanded sample dataset
data = {
    'title': [
        'Inception', 'The Matrix', 'Interstellar', 'The Avengers', 'Memento',
        'Fight Club', 'Pulp Fiction', 'Forrest Gump'
    ],
    'genres': [
        'Sci-Fi|Thriller', 'Sci-Fi|Action', 'Sci-Fi|Adventure', 'Action|Adventure', 'Mystery|Thriller',
        'Drama|Action', 'Crime|Drama', 'Drama|Romance'
    ]
}

# Create the DataFrame
movies_df = pd.DataFrame(data)


def recommend_movies(movie_title):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Check if the movie exists in the DataFrame
    if movie_title not in movies_df['title'].values:
        return f"'{movie_title}' not found in the dataset."

    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 similar movies
    movie_indices = [i[0] for i in sim_scores]

    return movies_df['title'].iloc[movie_indices].tolist()


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        user_input = request.form.get("movie_title")
        recommendations = recommend_movies(user_input)

    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
