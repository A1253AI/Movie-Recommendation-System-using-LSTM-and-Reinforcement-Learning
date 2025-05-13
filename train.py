
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

def train_time_aware_models(X_train, y_train, X_val, y_val, 
                           train_ratings, val_ratings,
                           train_times, val_times,
                           num_movies, genre_features, 
                           lstm_class, content_model_class, recommender_class,
                           epochs=10, batch_size=128):
    """Train models with time and rating awareness"""
    print("Training time and rating aware models...")
    
    # Create tensors for all data
    X_train_tensor = torch.LongTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    train_ratings_tensor = torch.FloatTensor(train_ratings).to(device)
    train_times_tensor = torch.FloatTensor(train_times).to(device)
    
    X_val_tensor = torch.LongTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    val_ratings_tensor = torch.FloatTensor(val_ratings).to(device)
    val_times_tensor = torch.FloatTensor(val_times).to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, train_ratings_tensor, train_times_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, val_ratings_tensor, val_times_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize models
    genre_features_tensor = torch.FloatTensor(genre_features).to(device)
    lstm_model = lstm_class(num_movies, embedding_dim=64, lstm_units=128).to(device)
    content_model = content_model_class(num_movies, genre_features_tensor, embedding_dim=64).to(device)
    
    # Create hybrid recommender
    hybrid_recommender = recommender_class(
        lstm_model=lstm_model,
        content_model=content_model,
        genre_features=genre_features,
        num_movies=num_movies,
        sequence_length=X_train[0].shape[0]
    )
    
    # Training loop
    print("\nTraining enhanced models...")
    
    # Modified train_epoch function to handle time and rating data
    def train_epoch(data_loader, model, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (movies, ratings, times, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            
            outputs = model(movies, ratings, times)
            loss = hybrid_recommender.criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(data_loader), 100.0 * correct / total
    
    # Modified validate function to handle time and rating data
    def validate(data_loader, model):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (movies, ratings, times, targets) in enumerate(data_loader):
                outputs = model(movies, ratings, times)
                loss = hybrid_recommender.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(data_loader), 100.0 * correct / total
    
    # Training history
    lstm_train_losses = []
    lstm_val_losses = []
    content_train_losses = []
    content_val_losses = []
    
    best_lstm_loss = float('inf')
    best_content_loss = float('inf')
    best_lstm_state = None
    best_content_state = None
    
    for epoch in range(epochs):
        # Train LSTM model
        lstm_train_loss, lstm_train_acc = train_epoch(train_loader, lstm_model, hybrid_recommender.lstm_optimizer)
        lstm_val_loss, lstm_val_acc = validate(val_loader, lstm_model)
        
        # Update LSTM scheduler
        hybrid_recommender.lstm_scheduler.step(lstm_val_loss)
        
        # Save best LSTM model
        if lstm_val_loss < best_lstm_loss:
            best_lstm_loss = lstm_val_loss
            best_lstm_state = lstm_model.state_dict().copy()
        
        # Train content model
        content_train_loss, content_train_acc = train_epoch(train_loader, content_model, hybrid_recommender.content_optimizer)
        content_val_loss, content_val_acc = validate(val_loader, content_model)
        
        # Update content scheduler
        hybrid_recommender.content_scheduler.step(content_val_loss)
        
        # Save best content model
        if content_val_loss < best_content_loss:
            best_content_loss = content_val_loss
            best_content_state = content_model.state_dict().copy()
        
        # Store losses for plotting
        lstm_train_losses.append(lstm_train_loss)
        lstm_val_losses.append(lstm_val_loss)
        content_train_losses.append(content_train_loss)
        content_val_losses.append(content_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"LSTM - Train Loss: {lstm_train_loss:.4f}, Train Acc: {lstm_train_acc:.2f}%, "
              f"Val Loss: {lstm_val_loss:.4f}, Val Acc: {lstm_val_acc:.2f}%")
        print(f"Content - Train Loss: {content_train_loss:.4f}, Train Acc: {content_train_acc:.2f}%, "
              f"Val Loss: {content_val_loss:.4f}, Val Acc: {content_val_acc:.2f}%")
    
    # Load best models
    lstm_model.load_state_dict(best_lstm_state)
    content_model.load_state_dict(best_content_state)
    
    # Calculate item popularity for fallback recommendations
    hybrid_recommender.calculate_popularity(y_train)
    
    return hybrid_recommender, {
        'lstm_train_losses': lstm_train_losses,
        'lstm_val_losses': lstm_val_losses,
        'content_train_losses': content_train_losses,
        'content_val_losses': content_val_losses
    }

def train_rl_component(hybrid_recommender, X_train, y_train, ratings_train, times_train, 
                     user_ids_train, dqn_agent_class, episodes=30, batch_size=64):
    """Train the reinforcement learning component"""
    print("\nTraining reinforcement learning component...")
    
    # Build DQN agent if needed
    if hybrid_recommender.dqn_agent is None:
        hybrid_recommender.build_dqn_agent(dqn_agent_class)
    
    # Sample data for RL training (limit to 5000 for efficiency)
    sample_size = min(5000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    X_sample = [X_train[i] for i in indices]
    y_sample = [y_train[i] for i in indices]
    ratings_sample = [ratings_train[i] for i in indices] if ratings_train is not None else None
    times_sample = [times_train[i] for i in indices] if times_train is not None else None
    
    # Prepare training data for RL
    training_data = []
    for i in range(len(X_sample)):
        sequence = X_sample[i]
        target = y_sample[i]
        ratings = ratings_sample[i] if ratings_sample is not None else None
        times = times_sample[i] if times_sample is not None else None
        
        training_data.append((sequence, target, ratings, times))
    
    # Check if we have enough data
    if len(training_data) < batch_size:
        print(f"Warning: Not enough training data ({len(training_data)}) for batch size ({batch_size}).")
        print("Reducing batch size to match available data.")
        batch_size = max(1, len(training_data) // 2)  # Ensure at least 1
    
    rewards_history = []
    losses = []
    
    for e in range(episodes):
        # Shuffle training data for each episode
        np.random.shuffle(training_data)
        epoch_rewards = 0
        epoch_losses = 0
        
        # Create minibatches
        num_batches = len(training_data) // batch_size
        
        if num_batches == 0:
            print("Error: Zero batches created. Check data size and batch size.")
            return [], []
        
        for batch_idx in tqdm(range(num_batches), desc=f"Episode {e+1}/{episodes}"):
            batch = training_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_rewards_sum = 0
            batch_loss_sum = 0
            
            for i in range(len(batch)):
                sequence, target, ratings, times = batch[i]
                
                # Get state features
                state = hybrid_recommender.get_state_features(sequence, ratings, times)
                
                # Choose action using epsilon-greedy policy
                action = hybrid_recommender.dqn_agent.act(state)
                
                # Calculate reward
                # Base reward for correct prediction
                target_reward = 1.0 if action == target else 0.0
                
                # Genre similarity reward component
                genre_reward = 0.0
                if hybrid_recommender.genre_features is not None and action < len(hybrid_recommender.genre_features) and target < len(hybrid_recommender.genre_features):
                    similarity = cosine_similarity(
                        [hybrid_recommender.genre_features[action]], 
                        [hybrid_recommender.genre_features[target]]
                    )[0][0]
                    genre_reward = similarity * 0.5
                
                # Rating component if available
                rating_reward = 0.0
                if ratings is not None:
                    avg_rating = np.mean(ratings)
                    rating_reward = max(0, (avg_rating - 3) / 2)  # Scale to 0-1 range from 3-5 ratings
                
                # Combine reward components
                total_reward = (0.4 * target_reward) + (0.3 * genre_reward) + (0.3 * rating_reward)
                batch_rewards_sum += total_reward
                
                # For simplicity, we use the same state as next_state
                next_state = state
                done = False
                
                # Remember the experience
                hybrid_recommender.dqn_agent.remember(state, action, total_reward, next_state, done)
            
            # Update DQN through experience replay
            if len(hybrid_recommender.dqn_agent.memory) >= batch_size:
                loss = hybrid_recommender.dqn_agent.replay(min(len(hybrid_recommender.dqn_agent.memory), batch_size))
                batch_loss_sum += loss if loss is not None else 0
            
            epoch_rewards += batch_rewards_sum
            epoch_losses += batch_loss_sum
        
        # Update target network periodically
        if (e + 1) % hybrid_recommender.dqn_agent.update_target_frequency == 0:
            hybrid_recommender.dqn_agent._update_target_network()
            print("Target network updated!")
        
        # Calculate averages safely
        try:
            average_reward = epoch_rewards / (num_batches * batch_size)
            average_loss = epoch_losses / num_batches if epoch_losses != 0 else 0
        except ZeroDivisionError:
            print("Warning: Division by zero when calculating averages.")
            average_reward = 0
            average_loss = 0
            
        rewards_history.append(average_reward)
        losses.append(average_loss)
        
        print(f"Episode: {e+1}/{episodes}, "
              f"Avg Reward: {average_reward:.4f}, "
              f"Avg Loss: {average_loss:.4f}, "
              f"Epsilon: {hybrid_recommender.dqn_agent.epsilon:.2f}")
    
    return rewards_history, losses

def tune_model_weights(hybrid_recommender, X_val, y_val, ratings_val=None, times_val=None):
    """Find optimal weights for LSTM and content models"""
    print("\nTuning model weights...")
    best_hit_rate = 0
    best_weight = 0.5
    
    # Use a subset of validation data for faster tuning
    sample_size = min(1000, len(X_val))
    indices = np.random.choice(len(X_val), sample_size, replace=False)
    
    X_val_sample = [X_val[i] for i in indices]
    y_val_sample = [y_val[i] for i in indices]
    
    ratings_val_sample = None
    if ratings_val is not None:
        ratings_val_sample = [ratings_val[i] for i in indices]
        
    times_val_sample = None
    if times_val is not None:
        times_val_sample = [times_val[i] for i in indices]
    
    for weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
        hybrid_recommender.lstm_weight = weight
        # Use the sampled subset for evaluation
        hit_rate = hybrid_recommender.evaluate_hit_rate(
            X_val_sample, y_val_sample, ratings_val_sample, times_val_sample, top_n=10
        )
        print(f"LSTM Weight: {weight}, Hit Rate: {hit_rate:.4f}")
        
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_weight = weight
    
    hybrid_recommender.lstm_weight = best_weight
    print(f"Best LSTM weight: {best_weight}, Hit Rate: {best_hit_rate:.4f}")
    
    return best_weight, best_hit_rate

def evaluate_model(hybrid_recommender, X_test, y_test, ratings_test=None, times_test=None, 
                 top_n=10, movie_encoder=None, movies_df=None, user_id=None, user_sequences_dict=None):
    """Evaluate the model and generate recommendations for a sample user"""
    print("\nEvaluating the model on test data...")
    
    # Use a subset for faster evaluation
    test_subset_size = min(1000, len(X_test))
    test_indices = np.random.choice(len(X_test), test_subset_size, replace=False)
    
    X_test_subset = [X_test[i] for i in test_indices]
    y_test_subset = [y_test[i] for i in test_indices]
    
    ratings_test_subset = None
    if ratings_test is not None:
        ratings_test_subset = [ratings_test[i] for i in test_indices]
        
    times_test_subset = None
    if times_test is not None:
        times_test_subset = [times_test[i] for i in test_indices]
    
    # Calculate hit rate
    hit_rate = hybrid_recommender.evaluate_hit_rate(
        X_test_subset, y_test_subset, ratings_test_subset, times_test_subset, top_n
    )
    
    print(f"Hit Rate @{top_n}: {hit_rate:.4f}")
    
    # Generate recommendations for a sample user if data is provided
    if user_id is not None and user_sequences_dict is not None and movie_encoder is not None and movies_df is not None:
        if user_id in user_sequences_dict:
            # Get the user's data
            user_seqs, targets, ratings, timestamps, _ = user_sequences_dict[user_id]
            
            # Get the user's latest sequence and its data
            latest_sequence = user_seqs[-1]
            latest_ratings = ratings[-1]
            latest_times = timestamps[-1]
            
            # Calculate time differences for the latest sequence
            max_timestamp = max([ts.max() for user_data in user_sequences_dict.values() for ts in user_data[3]])
            latest_time_diffs = [(max_timestamp - ts) / (24 * 60 * 60) for ts in latest_times]
            
            # Generate recommendations
            recommendations = hybrid_recommender.recommend(
                latest_sequence, latest_ratings, latest_time_diffs,
                top_n=top_n, use_rl=False, user_id=user_id, user_sequences_dict=user_sequences_dict
            )
            
            # Get watchlist
            watchlist = hybrid_recommender.get_user_watchlist(user_id, user_sequences_dict, movie_encoder, movies_df)
            
            # Convert recommendations to movie titles
            recommended_movies = []
            for movie_id in recommendations:
                original_id = movie_encoder.inverse_transform([movie_id])[0]
                movie_info = movies_df[movies_df['movieId'] == original_id]
                if not movie_info.empty:
                    title = movie_info['title'].values[0]
                    genres = movie_info['genres'].values[0]
                    recommended_movies.append({
                        'movieId': int(original_id),
                        'title': title,
                        'genres': genres
                    })
            
            # Display watchlist
            print(f"\nUser {user_id}'s watchlist (showing the 10 most recent of {len(watchlist)} movies):")
            for i, movie in enumerate(watchlist[:10], 1):  # Show first 10 movies
                rating_str = f"Rating: {movie['rating']:.1f}" if movie['rating'] is not None else "No rating"
                print(f"{i}. {movie['title']} - {movie['genres']} ({rating_str})")
            
            # Display recommendations
            print(f"\nTop {top_n} recommendations for user {user_id}:")
            for i, movie in enumerate(recommended_movies, 1):
                print(f"{i}. {movie['title']} - {movie['genres']}")
            
            # Generate explanations
            explanations = hybrid_recommender.explain_recommendations(
                latest_sequence, latest_ratings, latest_time_diffs,
                recommendations, movie_encoder, movies_df,
                user_id=user_id, user_sequences_dict=user_sequences_dict
            )
            
            # Display explanations
            print("\nRecommendation explanations:")
            for explanation in explanations:
                print(f"- {explanation['title']}: {explanation['explanation']}")
            
            # Analyze user preferences
            genre_preferences = hybrid_recommender.analyze_user_preferences(
                user_id, user_sequences_dict, movie_encoder, movies_df
            )
            
            print("\nUser genre preferences:")
            for genre, percentage, avg_rating, _ in genre_preferences[:5]:  # Show top 5 genres
                print(f"- {genre}: {percentage:.1f}% (Avg Rating: {avg_rating:.1f}/5)")
            
            # Compare recommendations to preferences
            preference_comparison = hybrid_recommender.compare_recommendations_to_preferences(
                recommendations, user_id, user_sequences_dict, movie_encoder, movies_df
            )
            
            print("\nRecommendation alignment with user preferences:")
            for genre, data in preference_comparison[:5]:  # Top 5 differences
                print(f"- {genre}: User preference {data['user_preference']:.1f}%, "
                      f"Recommendations {data['recommendation_percentage']:.1f}%, "
                      f"Difference {data['difference']:.1f}%")
            
            return {
                'hit_rate': hit_rate,
                'recommendations': recommended_movies,
                'explanations': explanations,
                'user_preferences': genre_preferences,
                'preference_comparison': preference_comparison
            }
    
    return {'hit_rate': hit_rate}

def plot_training_history(history, save_path='model_training_history.png'):
    """Plot training history metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['lstm_train_losses'], label='LSTM Training Loss')
    plt.plot(history['lstm_val_losses'], label='LSTM Validation Loss')
    plt.title('LSTM Model Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['content_train_losses'], label='Content Training Loss')
    plt.plot(history['content_val_losses'], label='Content Validation Loss')
    plt.title('Content Model Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history plot saved to {save_path}")

def plot_rl_history(rewards_history, losses_history, save_path='rl_training_history.png'):
    """Plot reinforcement learning training metrics"""
    if not rewards_history or not losses_history:
        print("No RL history to plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards_history)
    plt.title('Reinforcement Learning Training - Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(losses_history)
    plt.title('Reinforcement Learning Training - Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Average Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"RL training history plot saved to {save_path}")