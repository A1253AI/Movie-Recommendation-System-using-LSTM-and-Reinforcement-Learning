import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split


# Import from our modules
from data_preprocessing import (
    load_data, extract_genres, preprocess_data, 
    create_time_weighted_sequences, prepare_training_data
)
from models import TimeRatingAwareLSTM, TimeRatingAwareContentModel, DQNAgent
from recommender import EnhancedHybridRecommender
from train import (
    train_time_aware_models, train_rl_component, tune_model_weights, 
    evaluate_model, plot_training_history, plot_rl_history
)



# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    print("Starting enhanced movie recommendation system with timestamps and ratings...")
    
    # 1. Load and preprocess data
    movies, ratings = load_data(movies_path='movies_1.csv', ratings_path='rating_1.csv')
    
    # Sample movie and rating data format
    print("\nSample movie data format:")
    print("movieId title genres")
    print("1      Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy")
    
    print("\nSample rating data format:")
    print("userId  movieId  rating  timestamp")
    print("1       17       4       944249077")
    
    # Preprocess data
    merged_data, user_encoder, movie_encoder, rating_scaler, num_users, num_movies = preprocess_data(
        movies, ratings, min_user_ratings=10, min_movie_ratings=5
    )
    
     # Extract genre features before filtering
    genre_matrix, genre_list = extract_genres(movies)
    
    # Preprocess and filter data
    merged_data, user_encoder, movie_encoder, rating_scaler, num_users, num_movies = preprocess_data(
        movies, ratings, min_user_ratings=10, min_movie_ratings=5
    )
    
    # Map genre features to encoded movie IDs using a mapping dictionary
    filtered_movie_ids = set(merged_data['movieId'].unique())
    movie_id_to_row = {movie_id: i for i, movie_id in enumerate(movies['movieId'])}
    
    genre_features = np.zeros((num_movies, len(genre_list)))
    for i, movie_id in enumerate(movie_encoder.classes_):
        original_idx = movie_id_to_row[movie_id]
        genre_features[i] = genre_matrix[original_idx]
    
    # 2. Create time-weighted sequences
    user_sequences = create_time_weighted_sequences(
        merged_data, sequence_length=5, stride=2
    )
    
    # 3. Prepare data for training
    all_seqs, all_targets, all_ratings, all_time_diffs, user_ids, max_timestamp = prepare_training_data(
        user_sequences
    )
    
    # 4. Split data for training, validation, and testing
    X_train, X_temp, y_train, y_temp, ratings_train, ratings_temp, times_train, times_temp, user_ids_train, user_ids_temp = train_test_split(
        all_seqs, all_targets, all_ratings, all_time_diffs, user_ids, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_val, y_test, ratings_val, ratings_test, times_val, times_test, user_ids_val, user_ids_test = train_test_split(
        X_temp, y_temp, ratings_temp, times_temp, user_ids_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # 5. Train the enhanced models
    hybrid_recommender, history = train_time_aware_models(
        X_train, y_train, X_val, y_val,
        ratings_train, ratings_val,
        times_train, times_val,
        num_movies, genre_features,
        TimeRatingAwareLSTM, TimeRatingAwareContentModel, EnhancedHybridRecommender,
        epochs=10, batch_size=128
    )
    
    # 6. Plot training history
    plot_training_history(history, save_path='model_training_history.png')
    
    # 7. Tune model weights
    best_weight, best_hit_rate = tune_model_weights(
        hybrid_recommender, X_val, y_val, ratings_val, times_val
    )
    
    # 8. Train the reinforcement learning component
    try:
        hybrid_recommender.build_dqn_agent(DQNAgent)
        rewards_history, losses_history = train_rl_component(
            hybrid_recommender, X_train, y_train, ratings_train, times_train, 
            user_ids_train, DQNAgent, episodes=5, batch_size=35
        )
        plot_rl_history(rewards_history, losses_history, save_path='rl_training_history.png')
    except Exception as e:
        print(f"Error in RL component training: {e}")
        print("Continuing with baseline models only.")
    
    # 9. Evaluate the model and generate sample recommendations
    sample_user_id = list(user_sequences.keys())[0] if user_sequences else None
    
    evaluation_results = evaluate_model(
        hybrid_recommender, X_test, y_test, ratings_test, times_test,
        top_n=10, movie_encoder=movie_encoder, movies_df=movies,
        user_id=sample_user_id, user_sequences_dict=user_sequences
    )
    
    # 10. Save the trained models
    try:
        save_dir = 'saved_recommender_models/'
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(hybrid_recommender.lstm_model.state_dict(), f"{save_dir}time_aware_lstm_model.pt")
        torch.save(hybrid_recommender.content_model.state_dict(), f"{save_dir}time_aware_content_model.pt")
        
        if hybrid_recommender.dqn_agent:
            torch.save(hybrid_recommender.dqn_agent.model.state_dict(), f"{save_dir}enhanced_dqn_model.pt")
        
        print(f"Models saved to {save_dir}")
    except Exception as e:
        print(f"Error saving models: {e}")
    
    print("\nEnhanced movie recommendation system training and evaluation complete!")

if __name__ == "__main__":
    main()