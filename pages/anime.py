import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD

st.title("Anime Suggestions")
st.header("This is a ML Anime Suggestion Platform")

st.sidebar.page_link("home.py", label="Home")
st.sidebar.page_link("pages/movies.py", label="Movies")
st.sidebar.page_link("pages/anime.py", label="Anime")

# Load your data (adjust this path to your dataset)
anime = pd.read_csv("anime-dataset-2023.csv")  # Assuming a CSV file

anime = anime.dropna(subset=['Name'])
anime = anime.dropna(subset=['Genres'])
anime = anime.dropna(subset=['Type'])
anime = anime.dropna(subset=['Producers'])
anime = anime.dropna(subset=['Studios'])
anime = anime.dropna(subset=['Source'])
anime = anime.dropna(subset=['Popularity'])
anime = anime.dropna(subset=['Scored By'])
anime = anime.dropna(subset=['Score'])
anime = anime[pd.to_numeric(anime['Score'], errors='coerce').notnull()]
anime = anime.astype({'Score':'float','Scored By':'float','Popularity':'float'})

anime['index_col'] = anime.index

@st.cache_data
def load_similarities():
    # Clean data: Remove rows with missing values in important columns
    # movies = movies.dropna(subset=['title', 'genres', 'directors', 'cast', 'vote_average', 'vote_count'])

    # Preprocess features (similar to your existing code)
    title_tfidf = TfidfVectorizer(stop_words='english')
    title_encoded = title_tfidf.fit_transform(anime['Name'])

    type_tfidf = TfidfVectorizer(stop_words='english')
    type_encoded = type_tfidf.fit_transform(anime['Type'])

    genre_tfidf = TfidfVectorizer(stop_words='english')
    genre_encoded = genre_tfidf.fit_transform(anime['Genres'])

    producers_tfidf = TfidfVectorizer(stop_words='english')
    producers_encoded = producers_tfidf.fit_transform(anime['Producers'])

    studios_tfidf = TfidfVectorizer(stop_words='english')
    studios_encoded = studios_tfidf.fit_transform(anime['Studios'])

    source_tfidf = TfidfVectorizer(stop_words='english')
    source_encoded = source_tfidf.fit_transform(anime['Source'])

    # writers_tfidf = TfidfVectorizer(stop_words='english')
    # writers_encoded = writers_tfidf.fit_transform(movies['writers'])

    # cast_tfidf = TfidfVectorizer(stop_words='english')
    # cast_encoded = cast_tfidf.fit_transform(movies['cast'])

    # Normalize ratings and vote counts
    ratings = anime['Score'] / 10.0  # Normalize ratings to range [0, 1]
    ratings_vector = csr_matrix(ratings.values.reshape(-1, 1))

    # Normalize vote count using log transformation
    vote_count_scaled = np.log(anime['Popularity'] + 1)
    vote_count_scaled = np.array(vote_count_scaled).reshape(-1, 1)
    popularity_scaled = np.log(anime['Scored By'] + 1)
    popularity_scaled = np.array(popularity_scaled).reshape(-1, 1)

    # Combine all features into a single feature vector
    anime_features = hstack([
    title_encoded, 
    type_encoded,
    genre_encoded,
    producers_encoded,
    studios_encoded,
    source_encoded,
    ratings_vector,
    vote_count_scaled,
    popularity_scaled
    ])

    # Use Truncated SVD (a form of PCA for sparse matrices) for dimensionality reduction
    svd = TruncatedSVD(n_components=100)  # Adjust number of components based on memory constraints
    reduced_anime_features = svd.fit_transform(anime_features)

    # Calculate the cosine similarity matrix
    return cosine_similarity(reduced_anime_features, dense_output=False)

cosine_sim = load_similarities()


def recommend_anime(movie_id, top_n=3):
    movie_index = anime[anime['index_col'] == movie_id].index[0]
    
    # Get similarity scores for all anime with the selected movie
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    
    # Sort anime based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N similar anime (excluding the first movie itself)
    similar_anime = [i[0] for i in sim_scores[1:top_n+1]]
    
    recommended_anime = anime.iloc[similar_anime]
    return recommended_anime[['Name','ImageURL']]

option = st.selectbox(
    'Which Anime do you like?',
     options=["Which Anime do you like?"] + anime['Name'].tolist(),placeholder="Choose a Anime")
# Display recommendations only if a valid movie is selected
if option != "Which Anime do you like?":
    # Get the corresponding movie ID
    movie_id = anime.loc[anime['Name'] == option, 'index_col'].values[0]
    
    # Generate recommendations
    recommended_anime = recommend_anime(movie_id, top_n=3)
    print(recommended_anime)
    # Display recommendations
    # Assuming recommended_anime is a DataFrame with columns 'Poster URL' and 'Name'
    cols = st.columns(len(recommended_anime))  # Create columns for each movie

    for index, (col, row) in enumerate(zip(cols, recommended_anime.itertuples())):
        with col:
            st.image(f"{row.ImageURL}", width=150)  # Adjust width as needed
            st.write(f"**{row.Name}**")
else:
    st.write("Please select a Anime to see suggestions.")

