import streamlit as st
import pandas as pd
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import recommend, tfidf, tfidf_matrix

# UI
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("### 🍿 Smart Movie Recommender")

# Load data
data = pd.read_csv("data/tmdb_5000_movies.csv")
data = data[["title", "overview"]]
data.columns = ["Movie Name", "Storyline"]

# -------------------------------
# OPTION 1: SELECT MOVIE
# -------------------------------
st.subheader("🎥 Choose a Movie")

movie = st.selectbox("Select a Movie", data["Movie Name"])

if st.button("Recommend Movies"):

    results = recommend(movie)

    st.subheader("Top 5 Recommended Movies")

    for i, row in results.iterrows():
        st.write("###", row["Movie Name"])
        st.write(row["Storyline"])
        st.write("---")

# -------------------------------
# OPTION 2: STORY SEARCH
# -------------------------------
st.subheader("✍️ Recommend by Your Story")

user_input = st.text_area("Enter your own storyline")

if st.button("Recommend by Storyline"):

    if user_input.strip() != "":

        user_vec = tfidf.transform([user_input])
        scores = cosine_similarity(user_vec, tfidf_matrix)

        top_indices = scores.flatten().argsort()[-5:][::-1]

        st.subheader("Recommended Movies")

        for i in top_indices:
            st.write("###", data.iloc[i]["Movie Name"])
            st.write(data.iloc[i]["Storyline"])
            st.write("---")

    else:
        st.warning("Please enter a storyline")