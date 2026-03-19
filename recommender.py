import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("data/tmdb_5000_movies.csv")

# Select columns
data = data[["title", "overview"]]
data.columns = ["Movie Name", "Storyline"]

data.dropna(inplace=True)

# Reset index (VERY IMPORTANT FIX)
data = data.reset_index(drop=True)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Storyline'])

# Similarity
similarity = cosine_similarity(tfidf_matrix)

# Function
def recommend(movie_name):

    if movie_name not in data["Movie Name"].values:
        return pd.DataFrame()

    index = data[data["Movie Name"] == movie_name].index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:6]

    movie_indices = [i[0] for i in scores]

    return data.iloc[movie_indices]