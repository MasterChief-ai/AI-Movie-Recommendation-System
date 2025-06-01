import pandas as pd
import numpy as np
import re
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ------------------------------
# LOAD & MERGE DATA
# ------------------------------

# Load datasets
imdb_df = pd.read_csv('C:/Users/user/Documents/Nig Project/AI Movie Recommendation System/data/movie_metadata.csv')
ml_movies_df = pd.read_csv('C:/Users/user/Documents/Nig Project/AI Movie Recommendation System/data/movies.csv')
ml_links_df = pd.read_csv('C:/Users/user/Documents/Nig Project/AI Movie Recommendation System/data/links.csv')
ml_ratings_df = pd.read_csv('C:/Users/user/Documents/Nig Project/AI Movie Recommendation System/data/ratings.csv')

# Extract imdbId from IMDB URL
imdb_df['imdbId'] = imdb_df['movie_imdb_link'].apply(
    lambda x: int(re.search(r'tt(\d+)', str(x)).group(1)) if pd.notnull(x) and re.search(r'tt(\d+)', str(x)) else None
)
imdb_df_cleaned = imdb_df.dropna(subset=['imdbId']).copy()
imdb_df_cleaned['imdbId'] = imdb_df_cleaned['imdbId'].astype(int)

# Merge data
merged_df = pd.merge(ml_links_df, imdb_df_cleaned, on='imdbId', how='inner')
merged_df = pd.merge(merged_df, ml_movies_df, on='movieId', how='inner')
merged_df["genres"] = merged_df["genres_x"]
merged_df.drop(columns=["genres_x", "genres_y"], inplace=True, errors="ignore")
merged_df["genres"] = merged_df["genres"].fillna("").apply(lambda x: x.split("|"))
merged_df = merged_df.fillna('')  # Fill NaNs to avoid errors

# ------------------------------
# TRAIN COLLABORATIVE MODEL
# ------------------------------

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ml_ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
cf_model = SVD()
cf_model.fit(trainset)

def get_cf_scores(user_id):
    movie_ids = merged_df['movieId'].unique()
    scores = {}
    for movie_id in movie_ids:
        try:
            pred = cf_model.predict(user_id, movie_id)
            scores[movie_id] = pred.est
        except:
            scores[movie_id] = 0
    return scores

# ------------------------------
# CONTENT-BASED FILTERING
# ------------------------------

def get_cb_scores(movie_title):
    if 'combined' not in merged_df.columns:
        merged_df['combined'] = (
            merged_df['genres'].apply(lambda g: ' '.join(g)) + ' ' +
            merged_df['plot_keywords'].fillna('') + ' ' +
            merged_df['actor_1_name'].fillna('') + ' ' +
            merged_df['director_name'].fillna('')
        ).fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(merged_df['combined'])

    idx = merged_df[merged_df['movie_title'] == movie_title].index
    if len(idx) == 0:
        return {}
    
    cosine_sim = linear_kernel(tfidf_matrix[idx[0]], tfidf_matrix).flatten()
    scores = dict(zip(merged_df['movieId'], cosine_sim))
    return scores

# ------------------------------
# SEARCH-BASED FILTERING
# ------------------------------

def get_search_scores(keywords, top_n=10):
    keywords = [k.lower() for k in keywords]

    def score_row(row):
        content = f"{row['movie_title']} {row['genres']} {row['plot_keywords']} {row['actor_1_name']} {row['actor_2_name']} {row['actor_3_name']} {row['director_name']}".lower()
        return sum(1 for k in keywords if k in content)

    merged_df['search_score'] = merged_df.apply(score_row, axis=1)
    scores = dict(zip(merged_df['movieId'], merged_df['search_score']))
    return scores

# ------------------------------
# HYBRID RECOMMENDER
# ------------------------------

def hybrid_recommend(user_id, search_keywords, liked_movie_title=None, top_n=10):
    cf_scores = get_cf_scores(user_id)
    cb_scores = get_cb_scores(liked_movie_title) if liked_movie_title else {}
    search_scores = get_search_scores(search_keywords)

    all_movie_ids = set(cf_scores.keys()).union(cb_scores.keys(), search_scores.keys())
    
    hybrid_scores = {}
    for movie_id in all_movie_ids:
        score = (
            0.4 * cf_scores.get(movie_id, 0) +
            0.3 * cb_scores.get(movie_id, 0) +
            0.3 * search_scores.get(movie_id, 0)
        )
        hybrid_scores[movie_id] = score
    
    top_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = merged_df[merged_df['movieId'].isin([mid for mid, _ in top_movies])]

    return recommendations[['movie_title', 'genres', 'director_name', 'actor_1_name']].to_dict(orient='records')
