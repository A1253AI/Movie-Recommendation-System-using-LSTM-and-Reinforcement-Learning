import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
from collections import Counter
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EnhancedHybridRecommender:
    def __init__(self, lstm_model, content_model, genre_features, num_movies, sequence_length):
        self.lstm_model = lstm_model
        self.content_model = content_model
        self.genre_features = genre_features
        self.num_movies = num_movies
        self.sequence_length = sequence_length
        
        # Will be set after models are trained
        self.dqn_agent = None
        self.lstm_weight = 0.5  # Weight for LSTM model (content weight = 1 - lstm_weight)
        self.popularity = None  # For fallback recommendations
        
        # Loss function and optimizers
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        self.content_optimizer = torch.optim.Adam(content_model.parameters(), lr=0.001)
        
        # Learning rate schedulers
        self.lstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.lstm_optimizer, 'min', factor=0.5, patience=2)
        self.content_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.content_optimizer, 'min', factor=0.5, patience=2)
    
    def build_dqn_agent(self, dqn_agent_class):
        """Build DQN agent after models are trained"""
        # Get feature dimensions
        with torch.no_grad():
            sample_input = torch.LongTensor([[0] * self.sequence_length]).to(device)
            sample_ratings = torch.FloatTensor([[3.0] * self.sequence_length]).to(device)
            sample_times = torch.zeros(1, self.sequence_length).to(device)
            
            lstm_features = self.lstm_model.get_features(sample_input, sample_ratings, sample_times)
            content_features = self.content_model.get_features(sample_input, sample_ratings, sample_times)
        
        lstm_feature_dim = lstm_features.size(1)
        content_feature_dim = content_features.size(1)
        
        # Total feature dimension for the DQN state
        total_feature_dim = lstm_feature_dim + content_feature_dim
        
        # Create the DQN agent
        self.dqn_agent = dqn_agent_class(
            state_size=total_feature_dim,
            action_size=self.num_movies,
            genre_features=self.genre_features
        )
        
        print(f"DQN agent created with state size: {total_feature_dim}, action size: {self.num_movies}")
    
    def calculate_popularity(self, y_train):
        """Calculate item popularity for fallback recommendations"""
        counter = Counter(y_train)
        self.popularity = np.zeros(self.num_movies)
        
        # Fill popularity array
        for movie_id, count in counter.items():
            if movie_id < len(self.popularity):
                self.popularity[movie_id] = count
        
        # Normalize
        self.popularity = self.popularity / self.popularity.sum()
    
    def get_state_features(self, sequence, ratings=None, time_diffs=None):
        """Get combined features from both LSTM and content models"""
        sequence_tensor = torch.LongTensor([sequence]).to(device)
        
        # Convert ratings and time differences to tensors if provided
        if ratings is not None:
            ratings_tensor = torch.FloatTensor([ratings]).to(device)
        else:
            ratings_tensor = None
            
        if time_diffs is not None:
            time_diffs_tensor = torch.FloatTensor([time_diffs]).to(device)
        else:
            time_diffs_tensor = None
        
        # Get features from both models
        with torch.no_grad():
            self.lstm_model.eval()
            self.content_model.eval()
            
            try:
                lstm_features = self.lstm_model.get_features(sequence_tensor, ratings_tensor, time_diffs_tensor)
                content_features = self.content_model.get_features(sequence_tensor, ratings_tensor, time_diffs_tensor)
                
                # Combine features
                combined_features = torch.cat([lstm_features, content_features], dim=1)
                return combined_features.cpu().numpy()
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Get the expected dimensions
                sample_input = torch.LongTensor([[0] * self.sequence_length]).to(device)
                sample_ratings = torch.FloatTensor([[3.0] * self.sequence_length]).to(device)
                sample_times = torch.zeros(1, self.sequence_length).to(device)
                
                sample_lstm = self.lstm_model.get_features(sample_input, sample_ratings, sample_times)
                sample_content = self.content_model.get_features(sample_input, sample_ratings, sample_times)
                total_dim = sample_lstm.size(1) + sample_content.size(1)
                
                # Return zeros with the correct shape
                return np.zeros((1, total_dim))
    
    def recommend(self, user_sequence, user_ratings=None, time_diffs=None, top_n=10, 
                 use_rl=True, user_id=None, user_sequences_dict=None):
        """Generate recommendations with the hybrid approach using ratings and time data"""
        user_sequence_tensor = torch.LongTensor([user_sequence]).to(device)
        
        # Convert ratings and time differences to tensors if provided
        if user_ratings is not None:
            user_ratings_tensor = torch.FloatTensor([user_ratings]).to(device)
        else:
            user_ratings_tensor = None
            
        if time_diffs is not None:
            time_diffs_tensor = torch.FloatTensor([time_diffs]).to(device)
        else:
            time_diffs_tensor = None
        
        # Get predictions from both models with time and rating information
        with torch.no_grad():
            self.lstm_model.eval()
            self.content_model.eval()
            
            lstm_logits = self.lstm_model(user_sequence_tensor, user_ratings_tensor, time_diffs_tensor)
            content_logits = self.content_model(user_sequence_tensor, user_ratings_tensor, time_diffs_tensor)
            
            # Convert to probabilities
            lstm_probs = F.softmax(lstm_logits, dim=1)[0].cpu().numpy()
            content_probs = F.softmax(content_logits, dim=1)[0].cpu().numpy()
        
        # Combine predictions with learned weights
        hybrid_probs = (self.lstm_weight * lstm_probs) + ((1 - self.lstm_weight) * content_probs)
        
        # Use RL agent if available and requested
        if use_rl and self.dqn_agent is not None:
            try:
                # Get state features
                state = self.get_state_features(user_sequence, user_ratings, time_diffs)
                
                # Get RL Q-values
                with torch.no_grad():
                    self.dqn_agent.model.eval()
                    state_tensor = torch.FloatTensor(state).to(device)
                    q_values = self.dqn_agent.model(state_tensor)[0].cpu().numpy()
                
                # Scale Q-values to [0,1] range
                if q_values.max() != q_values.min():
                    scaled_q = (q_values - q_values.min()) / (q_values.max() - q_values.min())
                else:
                    scaled_q = q_values
                
                # Combine with other probabilities
                final_scores = (0.6 * hybrid_probs) + (0.4 * scaled_q)  # Increased RL weight
            except Exception as e:
                print(f"Warning: Error in RL component: {e}. Using hybrid predictions only.")
                final_scores = hybrid_probs
        else:
            final_scores = hybrid_probs
        
        # Create a set of items to exclude (already watched)
        exclude_items = set(user_sequence)
        
        # If we have user_id and user_sequences_dict, exclude all watched movies
        if user_id is not None and user_sequences_dict is not None and user_id in user_sequences_dict:
            user_seqs, targets, _, _, _ = user_sequences_dict[user_id]  # Updated tuple structure
            for seq in user_seqs:
                exclude_items.update(seq)
            exclude_items.update(targets)
        
        # Set scores to 0 for all items to exclude
        for item in exclude_items:
            if item < len(final_scores):
                final_scores[item] = 0
        
        # Get top N items
        top_indices = np.argsort(final_scores)[-top_n:][::-1]
        
        # If we couldn't find enough recommendations, add popular items
        if len(top_indices) < top_n and self.popularity is not None:
            remaining = top_n - len(top_indices)
            fallback_items = np.argsort(self.popularity)[-remaining*3:][::-1]
            
            # Filter out items already in top_indices or in exclude_items
            fallback_items = [item for item in fallback_items 
                              if item not in top_indices and item not in exclude_items]
            
            # Add fallback items to recommendations
            top_indices = np.append(top_indices, fallback_items[:remaining])
        
        return top_indices
    
    def evaluate_hit_rate(self, X_test, y_test, ratings_test=None, times_test=None, top_n=10):
        """Calculate hit rate metric"""
        hits = 0
        total = len(X_test)
        
        for i in range(total):
            try:
                # Get ratings and times for this sample if available
                ratings = ratings_test[i] if ratings_test is not None else None
                times = times_test[i] if times_test is not None else None
                
                # Use RL=False during evaluation to avoid batch norm issues
                recommendations = self.recommend(X_test[i], ratings, times, top_n, use_rl=False)
                actual = y_test[i]
                
                if actual in recommendations:
                    hits += 1
            except Exception as e:
                print(f"Warning: Error in evaluation for sample {i}: {e}")
                # Continue to next sample instead of failing
                continue
        
        hit_rate = hits / total
        return hit_rate
    
    def get_user_watchlist(self, user_id, user_sequences_dict, movie_encoder, movies_df):
        """
        Retrieves and displays the movies a user has already watched.
        
        Args:
            user_id: The ID of the user whose watchlist we want to see
            user_sequences_dict: Dictionary mapping user IDs to their sequence data
            movie_encoder: LabelEncoder for converting between encoded and original movie IDs
            movies_df: DataFrame containing movie information
            
        Returns:
            A list of dictionaries containing information about watched movies
        """
        if user_id not in user_sequences_dict:
            print(f"User {user_id} not found in the dataset")
            return []
        
        # Get all the user's movie sequences
        user_seqs, targets, ratings, timestamps, _ = user_sequences_dict[user_id]
        
        # Flatten all sequences and remove duplicates to get unique watched movies
        all_watched_movies = set()
        for seq in user_seqs:
            all_watched_movies.update(seq)
        
        # Also include target movies (next movies watched after sequences)
        all_watched_movies.update(targets)
        
        # Convert to list and sort by movie ID for consistent ordering
        all_watched_movies = sorted(list(all_watched_movies))
        
        # Convert encoded movie IDs back to original IDs and get movie details
        watchlist = []
        for movie_id in all_watched_movies:
            original_id = movie_encoder.inverse_transform([movie_id])[0]
            movie_info = movies_df[movies_df['movieId'] == original_id]
            
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                genres = movie_info['genres'].values[0]
                
                # Find the rating if available
                rating = None
                timestamp = None
                
                # Check if it's in any sequence
                for seq_idx, seq in enumerate(user_seqs):
                    for pos, movie in enumerate(seq):
                        if movie == movie_id:
                            rating = ratings[seq_idx][pos]
                            timestamp = timestamps[seq_idx][pos]
                            break
                
                # If not found in sequences, check if it's a target
                if rating is None:
                    for seq_idx, target in enumerate(targets):
                        if target == movie_id:
                            # Use the last rating in the corresponding sequence as a proxy
                            rating = ratings[seq_idx][-1]
                            timestamp = timestamps[seq_idx][-1]
                            break
                
                watchlist.append({
                    'movieId': int(original_id),
                    'title': title,
                    'genres': genres,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        # Sort by timestamp (most recent first) if available
        watchlist.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else 0, reverse=True)
        
        return watchlist
    
    def analyze_user_preferences(self, user_id, user_sequences_dict, movie_encoder, movies_df):
        """Analyze a user's genre preferences based on their watchlist"""
        if user_id not in user_sequences_dict:
            return {}
        
        # Get all sequences for this user
        all_sequences, targets, ratings, timestamps, _ = user_sequences_dict[user_id]
        
        # Combine all watched movies with their ratings and timestamps
        watched_movies = []
        
        # Process sequences
        for seq_idx, seq in enumerate(all_sequences):
            for pos, movie_id in enumerate(seq):
                watched_movies.append({
                    'movie_id': movie_id, 
                    'rating': ratings[seq_idx][pos],
                    'timestamp': timestamps[seq_idx][pos]
                })
        
        # Process targets
        for seq_idx, movie_id in enumerate(targets):
            watched_movies.append({
                'movie_id': movie_id,
                'rating': ratings[seq_idx][-1],  # Use the last rating in sequence as an approximation
                'timestamp': timestamps[seq_idx][-1]  # Use the last timestamp in sequence
            })
        
        # Get original movie IDs and calculate genre preferences
        genre_counts = {}
        genre_ratings = {}  # Sum of ratings for each genre
        genre_timestamps = {}  # Most recent timestamp for each genre
        
        # Current time for recency weighting
        current_time = time.time()
        
        for item in watched_movies:
            movie_id = item['movie_id']
            rating = item['rating']
            timestamp = item['timestamp']
            
            try:
                original_id = movie_encoder.inverse_transform([movie_id])[0]
                movie_info = movies_df[movies_df['movieId'] == original_id]
                
                if not movie_info.empty:
                    genres = movie_info['genres'].values[0].split('|')
                    
                    # Calculate recency weight (more recent = higher weight)
                    time_diff_days = (current_time - timestamp) / (24 * 60 * 60)
                    recency_weight = max(0.5, 1.0 - (time_diff_days / 365.0))  # Minimum weight of 0.5
                    
                    # Rating weight (higher rating = higher weight)
                    rating_weight = (rating / 5.0) ** 2  # Squared to emphasize high ratings
                    
                    # Combined weight
                    weight = recency_weight * rating_weight
                    
                    for genre in genres:
                        if genre not in genre_counts:
                            genre_counts[genre] = 0
                            genre_ratings[genre] = 0
                            genre_timestamps[genre] = 0
                        
                        genre_counts[genre] += weight
                        genre_ratings[genre] += rating * weight
                        genre_timestamps[genre] = max(genre_timestamps[genre], timestamp)
            except Exception as e:
                print(f"Error processing movie {movie_id}: {e}")
                continue
        
        # Calculate average rating for each genre
        genre_avg_ratings = {}
        for genre in genre_counts:
            if genre_counts[genre] > 0:
                genre_avg_ratings[genre] = genre_ratings[genre] / genre_counts[genre]
            else:
                genre_avg_ratings[genre] = 0
        
        # Calculate percentages with recency and rating weights
        total_weight = sum(genre_counts.values())
        genre_percentages = [(genre, count / total_weight * 100, genre_avg_ratings[genre], genre_timestamps[genre]) 
                           for genre, count in genre_counts.items()]
        
        # Sort by weighted count (highest first)
        genre_percentages.sort(key=lambda x: x[1], reverse=True)
        
        return genre_percentages
    
    def explain_recommendations(self, user_sequence, user_ratings, time_diffs, 
                              recommendations, movie_encoder, movies_df, 
                              user_id=None, user_sequences_dict=None):
        """Provide detailed explanations incorporating rating and time data"""
        explanations = []
        
        # Get user's watched movies from sequence
        watched_movie_ids = movie_encoder.inverse_transform(user_sequence)
        watched_titles = []
        watched_genres_dict = {}
        watched_ratings_dict = {}
        watched_time_dict = {}
        
        # Current timestamp for calculating recency
        current_timestamp = time.time()
        
        # Process the sequence data
        for i, movie_id in enumerate(watched_movie_ids):
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                genres = movie_info['genres'].values[0].split('|')
                watched_titles.append(title)
                watched_genres_dict[title] = genres
                
                # Record rating if available
                if user_ratings is not None:
                    watched_ratings_dict[title] = user_ratings[i]
                
                # Record time information if available
                if time_diffs is not None:
                    # Calculate an approximate timestamp from time difference
                    approx_timestamp = current_timestamp - time_diffs[i]
                    watched_time_dict[title] = approx_timestamp
        
        # Get full watchlist if available
        all_watched_titles = watched_titles.copy()
        all_watched_genres = set()
        all_ratings = {}
        all_times = {}
        
        if user_id is not None and user_sequences_dict is not None and user_id in user_sequences_dict:
            # Get the full user data
            user_seqs, targets, ratings, timestamps, target_timestamps = user_sequences_dict[user_id]
            
            # Process all watched movies
            all_watched = set()
            for seq_idx, seq in enumerate(user_seqs):
                all_watched.update(seq)
                
                # Process ratings and timestamps for each movie
                for i, movie_id in enumerate(seq):
                    try:
                        orig_id = movie_encoder.inverse_transform([movie_id])[0]
                        movie_info = movies_df[movies_df['movieId'] == orig_id]
                        if not movie_info.empty:
                            title = movie_info['title'].values[0]
                            all_ratings[title] = ratings[seq_idx][i]
                            all_times[title] = timestamps[seq_idx][i]
                    except Exception as e:
                        print(f"Error processing sequence movie {movie_id}: {e}")
                        continue
            
            # Add target movies
            all_watched.update(targets)
            for seq_idx, movie_id in enumerate(targets):
                try:
                    orig_id = movie_encoder.inverse_transform([movie_id])[0]
                    movie_info = movies_df[movies_df['movieId'] == orig_id]
                    if not movie_info.empty:
                        title = movie_info['title'].values[0]
                        # Use the last rating in sequence as an approximation
                        all_ratings[title] = ratings[seq_idx][-1]
                        all_times[title] = target_timestamps[seq_idx]
                except Exception as e:
                    print(f"Error processing target movie {movie_id}: {e}")
                    continue
            
            # Get details for all watched movies
            all_watched_ids = movie_encoder.inverse_transform(list(all_watched))
            for movie_id in all_watched_ids:
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    title = movie_info['title'].values[0]
                    genres = movie_info['genres'].values[0].split('|')
                    all_watched_titles.append(title)
                    watched_genres_dict[title] = genres
                    all_watched_genres.update(genres)
        
        # Define how recent a movie needs to be to be considered in explanations
        recent_threshold = current_timestamp - (90 * 24 * 60 * 60)  # 90 days
        
        # For each recommendation, generate an explanation
        for i, movie_id in enumerate(recommendations):
            try:
                orig_movie_id = movie_encoder.inverse_transform([movie_id])[0]
                movie_info = movies_df[movies_df['movieId'] == orig_movie_id]
                
                if movie_info.empty:
                    continue
                    
                title = movie_info['title'].values[0]
                rec_genres = movie_info['genres'].values[0].split('|')
                
                # Find similar movies based on genre and rating
                similar_movies = []
                for watched_title in all_watched_titles:
                    watched_genres = watched_genres_dict.get(watched_title, [])
                    common_genres = set(rec_genres).intersection(set(watched_genres))
                    
                    # Skip if no genre overlap
                    if not common_genres:
                        continue
                    
                    # Calculate similarity score based on genre overlap
                    genre_sim = len(common_genres) / max(len(rec_genres), len(watched_genres))
                    
                    # Add rating factor if available
                    rating_factor = 0
                    if watched_title in all_ratings:
                        rating = all_ratings[watched_title]
                        rating_factor = (rating - 2.5) / 2.5  # Scale to [-1,1] range
                    
                    # Add recency factor if available
                    recency_factor = 0
                    if watched_title in all_times:
                        timestamp = all_times[watched_title]
                        time_diff_days = (current_timestamp - timestamp) / (24 * 60 * 60)
                        recency_factor = max(0, 1 - (time_diff_days / 365))  # Scale based on 1 year
                    
                    # Combined similarity score
                    similarity = genre_sim * (1 + 0.5 * rating_factor + 0.5 * recency_factor)
                    
                    similar_movies.append((
                        watched_title,
                        similarity,
                        list(common_genres),
                        all_ratings.get(watched_title),
                        all_times.get(watched_title)
                    ))
                
                # Sort by similarity score
                similar_movies.sort(key=lambda x: x[1], reverse=True)
                
                # Create explanation
                if similar_movies:
                    # Get top similar movies (up to 2)
                    top_similar = similar_movies[:2]
                    similar_titles = []
                    for movie, _, _, rating, _ in top_similar:
                        if rating:
                            similar_titles.append(f"{movie} ({rating:.1f}/5)")
                        else:
                            similar_titles.append(movie)
                    
                    # Get common genres from top similar movies
                    common_genres = []
                    for _, _, genres, _, _ in top_similar:
                        for genre in genres:
                            if genre not in common_genres:
                                common_genres.append(genre)
                    
                    # Check if there are recent highly-rated movies
                    recent_high_rated = []
                    for movie, _, _, rating, timestamp in top_similar:
                        if rating and rating >= 4.0 and timestamp and timestamp > recent_threshold:
                            recent_high_rated.append((movie, rating))
                    
                    # Create explanation based on available data
                    if recent_high_rated:
                        # Emphasize recent highly-rated movies
                        movie_str = ", ".join([f"{m} ({r:.1f}/5)" for m, r in recent_high_rated])
                        explanation_text = (
                            f"Recommended because it shares {'/'.join(common_genres[:2])} genres with {movie_str} "
                            f"that you recently watched and rated highly."
                        )
                    else:
                        # Standard explanation
                        explanation_text = (
                            f"Recommended because it's similar to {', '.join(similar_titles)} "
                            f"you've watched before, particularly in {'/'.join(common_genres[:2])} genres."
                        )
                    
                    explanation = {
                        'title': title,
                        'rank': i+1,
                        'genres': rec_genres,
                        'explanation': explanation_text
                    }
                else:
                    # Fallback explanation based on genre match
                    matching_genres = [g for g in rec_genres if g in all_watched_genres]
                    if matching_genres:
                        explanation = {
                            'title': title,
                            'rank': i+1,
                            'genres': rec_genres,
                            'explanation': f"Recommended because it matches your interest in {'/'.join(matching_genres[:2])} genres."
                        }
                    else:
                        explanation = {
                            'title': title,
                            'rank': i+1,
                            'genres': rec_genres,
                            'explanation': f"Recommended as a popular movie in {'/'.join(rec_genres[:2])} genres that you might enjoy based on your watching patterns."
                        }
                
                # Determine the primary factor for recommendation
                user_sequence_tensor = torch.LongTensor([user_sequence]).to(device)
                ratings_tensor = torch.FloatTensor([user_ratings]).to(device) if user_ratings is not None else None
                time_tensor = torch.FloatTensor([time_diffs]).to(device) if time_diffs is not None else None
                
                with torch.no_grad():
                    self.lstm_model.eval()
                    self.content_model.eval()
                    
                    try:
                        lstm_logits = self.lstm_model(user_sequence_tensor, ratings_tensor, time_tensor)
                        content_logits = self.content_model(user_sequence_tensor, ratings_tensor, time_tensor)
                        
                        lstm_score = F.softmax(lstm_logits, dim=1)[0][movie_id].item()
                        content_score = F.softmax(content_logits, dim=1)[0][movie_id].item()
                        
                        if lstm_score > content_score:
                            explanation['primary_factor'] = 'viewing history patterns'
                        else:
                            explanation['primary_factor'] = 'genre and rating similarity'
                    except Exception as e:
                        explanation['primary_factor'] = 'unknown (error calculating factor)'
                
                explanations.append(explanation)
            except Exception as e:
                print(f"Error generating explanation for movie {movie_id}: {e}")
                continue
        
        return explanations
    
    def compare_recommendations_to_preferences(self, recommendations, user_id, user_sequences_dict, movie_encoder, movies_df):
        """Compare recommended movies with user preferences to evaluate recommendation quality"""
        # Get user genre preferences
        genre_preferences = self.analyze_user_preferences(user_id, user_sequences_dict, movie_encoder, movies_df)
        if not genre_preferences:
            return {}
        
        # Convert to dict for easier lookup
        preferred_genres = {genre: percentage for genre, percentage, _, _ in genre_preferences}
        
        # Get genres of recommended movies
        recommended_genres = {}
        for movie_id in recommendations:
            try:
                orig_movie_id = movie_encoder.inverse_transform([movie_id])[0]
                movie_info = movies_df[movies_df['movieId'] == orig_movie_id]
                
                if not movie_info.empty:
                    genres = movie_info['genres'].values[0].split('|')
                    for genre in genres:
                        recommended_genres[genre] = recommended_genres.get(genre, 0) + 1
            except Exception as e:
                print(f"Error analyzing recommendation {movie_id}: {e}")
                continue
        
        # Calculate percentages for recommended genres
        total_rec_genres = sum(recommended_genres.values())
        if total_rec_genres == 0:
            return {}
            
        rec_genre_percentages = {genre: count/total_rec_genres*100 
                              for genre, count in recommended_genres.items()}
        
        # Compare distributions
        comparison = {}
        all_genres = set(preferred_genres.keys()).union(set(rec_genre_percentages.keys()))
        
        for genre in all_genres:
            preferred_pct = preferred_genres.get(genre, 0)
            recommended_pct = rec_genre_percentages.get(genre, 0)
            difference = recommended_pct - preferred_pct
            
            comparison[genre] = {
                'user_preference': preferred_pct,
                'recommendation_percentage': recommended_pct,
                'difference': difference
            }
        
        # Sort by absolute difference
        sorted_comparison = sorted(comparison.items(), key=lambda x: abs(x[1]['difference']), reverse=True)
        
        return sorted_comparison