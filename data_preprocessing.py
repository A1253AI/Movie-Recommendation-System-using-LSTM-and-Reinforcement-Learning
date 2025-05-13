
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def load_data(movies_path='movies_1.csv', ratings_path='rating_1.csv'):
    """Load the movies and ratings datasets"""
    print("Loading datasets...")
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    print(f"Movies dataset shape: {movies.shape}")
    print(f"Ratings dataset shape: {ratings.shape}")
    
    return movies, ratings

def extract_genres(movies_df):
    """Extract genre features from the movies dataframe"""
    # Create a genre matrix
    genres = set()
    for g in movies_df['genres'].str.split('|'):
        genres.update(g)
    
    genres.discard('(no genres listed)')
    genre_list = sorted(list(genres))
    
    # Create a movie-genre matrix
    genre_matrix = np.zeros((len(movies_df), len(genre_list)))
    
    for i, movie_genres in enumerate(movies_df['genres'].str.split('|')):
        for genre in movie_genres:
            if genre in genre_list:
                genre_matrix[i, genre_list.index(genre)] = 1
    
    print(f"Extracted {len(genre_list)} genres as features")
    return genre_matrix, genre_list

def preprocess_data(movies_df, ratings_df, min_user_ratings=10, min_movie_ratings=5):
    """Preprocess data with timestamps and ratings"""
    print("Preprocessing data...")
    
    # Merge ratings and movies
    merged_data = pd.merge(ratings_df, movies_df, on='movieId')
    
    # Filter out users with too few ratings
    user_counts = merged_data['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    merged_data = merged_data[merged_data['userId'].isin(valid_users)]
    
    # Filter out movies with too few ratings
    movie_counts = merged_data['movieId'].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    merged_data = merged_data[merged_data['movieId'].isin(valid_movies)]
    
    print(f"Data after filtering: {merged_data.shape}")
    
    # Encode user and movie IDs
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    merged_data['userIdEncoded'] = user_encoder.fit_transform(merged_data['userId'])
    merged_data['movieIdEncoded'] = movie_encoder.fit_transform(merged_data['movieId'])
    
    num_users = len(merged_data['userId'].unique())
    num_movies = len(merged_data['movieId'].unique())
    
    print(f"Number of unique users after filtering: {num_users}")
    print(f"Number of unique movies after filtering: {num_movies}")
    
    # Scale ratings to [0, 1] range for model processing
    rating_scaler = MinMaxScaler(feature_range=(0, 1))
    merged_data['scaled_rating'] = rating_scaler.fit_transform(merged_data[['rating']])
    
    # Sort data by userId and timestamp for sequence creation
    merged_data = merged_data.sort_values(['userId', 'timestamp'])
    
    return merged_data, user_encoder, movie_encoder, rating_scaler, num_users, num_movies

def create_time_weighted_sequences(data, sequence_length=5, stride=2):
    """Create sequences with timestamp and rating information"""
    print("Creating time-weighted training sequences...")
    user_sequences = {}
    
    for user_id in tqdm(data['userIdEncoded'].unique()):
        user_data = data[data['userIdEncoded'] == user_id].sort_values('timestamp')
        
        # Skip if not enough interactions
        if len(user_data) < sequence_length + 1:
            continue
        
        # Get all sequences for this user with stride
        movies = user_data['movieIdEncoded'].values
        ratings = user_data['rating'].values  # Use original ratings
        timestamps = user_data['timestamp'].values
        
        user_seqs = []
        user_targets = []
        user_ratings = []
        user_timestamps = []
        target_timestamps = []
        
        for i in range(0, len(movies) - sequence_length, stride):
            user_seqs.append(movies[i:i+sequence_length])
            user_targets.append(movies[i+sequence_length])
            user_ratings.append(ratings[i:i+sequence_length])
            user_timestamps.append(timestamps[i:i+sequence_length])
            target_timestamps.append(timestamps[i+sequence_length])
        
        if user_seqs:  # Only add if we have sequences
            user_sequences[user_id] = (
                np.array(user_seqs), 
                np.array(user_targets), 
                np.array(user_ratings),
                np.array(user_timestamps),
                np.array(target_timestamps)
            )
    
    print(f"Created sequences for {len(user_sequences)} users")
    return user_sequences

def prepare_training_data(user_sequences):
    """Prepare data for model training"""
    all_seqs = []
    all_targets = []
    all_ratings = []
    all_timestamps = []
    user_ids = []
    
    for user_id, (seqs, targets, ratings, timestamps, _) in user_sequences.items():
        for i in range(len(seqs)):
            all_seqs.append(seqs[i])
            all_targets.append(targets[i])
            all_ratings.append(ratings[i])
            all_timestamps.append(timestamps[i])
            user_ids.append(user_id)
    
    # Calculate time differences (in days) from the most recent timestamp
    max_timestamp = max([ts.max() for ts in all_timestamps])
    all_time_diffs = []
    
    for timestamps in all_timestamps:
        time_diffs = [(max_timestamp - ts) / (24 * 60 * 60) for ts in timestamps]
        all_time_diffs.append(time_diffs)
    
    return all_seqs, all_targets, all_ratings, all_time_diffs, user_ids, max_timestamp