import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, hstack

st.title("Movie Suggestions")
st.header("This is a ML Movie Suggestion Platform")

st.sidebar.page_link("home.py", label="Home")
st.sidebar.page_link("pages/movies.py", label="Movies")
st.sidebar.page_link("pages/anime.py", label="Anime")

# Load your data (adjust this path to your dataset)
movies = pd.read_csv("TMDB  IMDB Movies Dataset.csv")  # Assuming a CSV file
movies = movies.dropna(subset=['vote_average'])
movies = movies.dropna(subset=['vote_count'])
movies = movies.dropna(subset=['title'])
movies = movies.dropna(subset=['original_language'])
movies = movies.dropna(subset=['popularity'])
movies = movies.dropna(subset=['genres'])
movies = movies.dropna(subset=['production_companies'])
movies = movies.dropna(subset=['production_countries'])
movies = movies.dropna(subset=['directors'])
movies = movies.dropna(subset=['writers'])
movies = movies.dropna(subset=['cast'])
movies['index_col'] = movies.index

# Clean data: Remove rows with missing values in important columns
movies = movies.dropna(subset=['title', 'genres', 'directors', 'cast', 'vote_average', 'vote_count'])
movies['index_col'] = movies.index

def calculate_cosine_similarity_in_chunks(movie_features, chunk_size=1000):
    num_movies = movie_features.shape[0]
    similarity_matrix = np.zeros((num_movies, num_movies))

    # Compute cosine similarity in chunks
    for start in range(0, num_movies, chunk_size):
        end = min(start + chunk_size, num_movies)
        
        # Calculate cosine similarity for the current chunk
        sim_chunk = cosine_similarity(movie_features[start:end], movie_features)
        
        # Store the chunk in the similarity matrix
        similarity_matrix[start:end, :] = sim_chunk
        
        # You can save or use the chunk at this point to free up memory
        
    return similarity_matrix

def load_similarities():
    

    # Preprocess features (similar to your existing code)
    title_tfidf = TfidfVectorizer(stop_words='english')
    title_encoded = title_tfidf.fit_transform(movies['title'])

    language_tfidf = TfidfVectorizer(stop_words='english')
    language_encoded = language_tfidf.fit_transform(movies['original_language'])

    genre_tfidf = TfidfVectorizer(stop_words='english')
    genre_encoded = genre_tfidf.fit_transform(movies['genres'])

    # prod_comp_tfidf = TfidfVectorizer(stop_words='english')
    # prod_comp_encoded = prod_comp_tfidf.fit_transform(movies['production_companies'])

    # prod_cont_tfidf = TfidfVectorizer(stop_words='english')
    # prod_cont_encoded = prod_cont_tfidf.fit_transform(movies['production_countries'])

    directors_tfidf = TfidfVectorizer(stop_words='english')
    directors_encoded = directors_tfidf.fit_transform(movies['directors'])

    writers_tfidf = TfidfVectorizer(stop_words='english')
    writers_encoded = writers_tfidf.fit_transform(movies['writers'])

    cast_tfidf = TfidfVectorizer(stop_words='english')
    cast_encoded = cast_tfidf.fit_transform(movies['cast'])

    # Normalize ratings and vote counts
    ratings = movies['vote_average'] / 10.0  # Normalize ratings to range [0, 1]
    ratings_vector = csr_matrix(ratings.values.reshape(-1, 1))

    # Normalize vote count using log transformation
    vote_count_scaled = np.log(movies['vote_count'] + 1)
    vote_count_scaled = np.array(vote_count_scaled).reshape(-1, 1)
    popularity_scaled = np.log(movies['popularity'] + 1)
    popularity_scaled = np.array(popularity_scaled).reshape(-1, 1)

    # Combine all features into a single feature vector
    movie_features = hstack([
        title_encoded, 
        genre_encoded,
        directors_encoded,
        writers_encoded,
        cast_encoded,
        ratings_vector,
        language_encoded,
        vote_count_scaled,
        popularity_scaled
    ])

    # Use Truncated SVD (a form of PCA for sparse matrices) for dimensionality reduction
    svd = TruncatedSVD(n_components=100)  # Adjust number of components based on memory constraints
    reduced_movie_features = svd.fit_transform(movie_features)

    # Calculate the cosine similarity matrix
    cosine_sim_sparse  = cosine_similarity(reduced_movie_features, dense_output=False)

    # Assuming reduced_movie_features is your feature matrix
    return calculate_cosine_similarity_in_chunks(reduced_movie_features, chunk_size=1000)

cosine_sim = load_similarities()


def recommend_movies(movie_id, top_n=3):
    movie_index = movies[movies['index_col'] == movie_id].index[0]
    
    # Get similarity scores for all movies with the selected movie
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    
    # Sort movies based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N similar movies (excluding the first movie itself)
    similar_movies = [i[0] for i in sim_scores[1:top_n+1]]
    
    recommended_movies = movies.iloc[similar_movies]
    return recommended_movies[['title','poster_path']]

option = st.selectbox(
    'Which Movie do you like?',
     options=["Which Movie do you like?"] + movies['title'].tolist(),placeholder="Choose a Movie")
# Display recommendations only if a valid movie is selected
if option != "Which Movie do you like?":
    # Get the corresponding movie ID
    movie_id = movies.loc[movies['title'] == option, 'index_col'].values[0]
    
    # Generate recommendations
    recommended_movies = recommend_movies(movie_id, top_n=3)
    print(recommended_movies)
    # Display recommendations
    # Assuming recommended_movies is a DataFrame with columns 'Poster URL' and 'Title'
    cols = st.columns(len(recommended_movies))  # Create columns for each movie

    for index, (col, row) in enumerate(zip(cols, recommended_movies.itertuples())):
        with col:
            st.image(f"https://image.tmdb.org/t/p/w500/{row.poster_path}", width=150)  # Adjust width as needed
            st.write(f"**{row.title}**")
else:
    st.write("Please select a movie to see suggestions.")

