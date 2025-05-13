
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

class TimeRatingAwareLSTM(nn.Module):
    def __init__(self, num_movies, embedding_dim=50, lstm_units=128, dropout_rate=0.3):
        super(TimeRatingAwareLSTM, self).__init__()
        self.embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Rating embedding - convert ratings to dense representation
        self.rating_embedding = nn.Linear(1, 8)
        
        # Time difference embedding - convert time difference to dense representation
        self.time_embedding = nn.Linear(1, 8)
        
        # Combined input size
        combined_input_size = embedding_dim + 8 + 8  # movie embedding + rating embedding + time embedding
        
        # LSTM layers
        self.lstm1 = nn.LSTM(combined_input_size, lstm_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units//2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(lstm_units//2)
        self.fc1 = nn.Linear(lstm_units//2, 256)
        self.dropout3 = nn.Dropout(dropout_rate/2)
        self.fc2 = nn.Linear(256, num_movies)
        
    def forward(self, movie_seq, rating_seq=None, time_seq=None):
        # movie_seq shape: [batch_size, sequence_length]
        batch_size = movie_seq.size(0)
        seq_length = movie_seq.size(1)
        
        # Get movie embeddings
        movie_emb = self.embedding(movie_seq)  # [batch_size, sequence_length, embedding_dim]
        
        # If no ratings provided, use a default rating
        if rating_seq is None:
            rating_seq = torch.ones(batch_size, seq_length, 1, device=movie_seq.device) * 3.0  # Default rating
        else:
            # Ensure rating_seq is shaped correctly
            if rating_seq.dim() == 2:
                rating_seq = rating_seq.unsqueeze(-1)
        
        # If no time differences provided, use zeros
        if time_seq is None:
            time_seq = torch.zeros(batch_size, seq_length, 1, device=movie_seq.device)
        else:
            # Ensure time_seq is shaped correctly
            if time_seq.dim() == 2:
                time_seq = time_seq.unsqueeze(-1)
        
        # Get rating embeddings
        rating_emb = self.rating_embedding(rating_seq)
        
        # Get time embeddings
        time_emb = self.time_embedding(time_seq)
        
        # Concatenate all embeddings
        combined_emb = torch.cat([movie_emb, rating_emb, time_emb], dim=2)
        
        # LSTM processing
        lstm_out1, _ = self.lstm1(combined_emb)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Take the last time step output
        lstm_out2 = lstm_out2[:, -1, :]
        
        # Apply batch normalization
        bn_out = self.bn(lstm_out2)
        
        # Dense layers
        dense1 = F.relu(self.fc1(bn_out))
        dense1 = self.dropout3(dense1)
        output = self.fc2(dense1)
        
        return output
    
    def get_features(self, movie_seq, rating_seq=None, time_seq=None):
        # Similar to forward, but returns the features before the final layer
        batch_size = movie_seq.size(0)
        seq_length = movie_seq.size(1)
        
        movie_emb = self.embedding(movie_seq)
        
        if rating_seq is None:
            rating_seq = torch.ones(batch_size, seq_length, 1, device=movie_seq.device) * 3.0
        else:
            if rating_seq.dim() == 2:
                rating_seq = rating_seq.unsqueeze(-1)
        
        if time_seq is None:
            time_seq = torch.zeros(batch_size, seq_length, 1, device=movie_seq.device)
        else:
            if time_seq.dim() == 2:
                time_seq = time_seq.unsqueeze(-1)
        
        rating_emb = self.rating_embedding(rating_seq)
        time_emb = self.time_embedding(time_seq)
        
        combined_emb = torch.cat([movie_emb, rating_emb, time_emb], dim=2)
        
        lstm_out1, _ = self.lstm1(combined_emb)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        lstm_out2 = lstm_out2[:, -1, :]
        bn_out = self.bn(lstm_out2)
        
        dense1 = F.relu(self.fc1(bn_out))
        
        return dense1

class TimeRatingAwareContentModel(nn.Module):
    def __init__(self, num_movies, genre_features, embedding_dim=50):
        super(TimeRatingAwareContentModel, self).__init__()
        self.genre_dim = genre_features.shape[1]
        
        # Movie embedding
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Rating embedding
        self.rating_embedding = nn.Linear(1, 8)
        
        # Time embedding
        self.time_embedding = nn.Linear(1, 8)
        
        # Pre-trained genre embedding (non-trainable)
        self.register_buffer('genre_features', genre_features)
        
        # Combined input size
        combined_input_size = embedding_dim + self.genre_dim + 8 + 8  # movie + genre + rating + time
        
        # LSTM layer
        self.lstm = nn.LSTM(combined_input_size, 128, batch_first=True)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_movies)
    
    def forward(self, movie_seq, rating_seq=None, time_seq=None):
        batch_size = movie_seq.size(0)
        seq_length = movie_seq.size(1)
        
        # Get movie embeddings
        movie_emb = self.movie_embedding(movie_seq)
        
        # Get genre embeddings
        genre_emb = torch.zeros(batch_size, seq_length, self.genre_dim, device=movie_seq.device)
        for i in range(batch_size):
            for j in range(seq_length):
                movie_idx = movie_seq[i, j]
                if movie_idx < len(self.genre_features):
                    genre_emb[i, j] = self.genre_features[movie_idx]
        
        # Handle rating and time embeddings
        if rating_seq is None:
            rating_seq = torch.ones(batch_size, seq_length, 1, device=movie_seq.device) * 3.0
        else:
            if rating_seq.dim() == 2:
                rating_seq = rating_seq.unsqueeze(-1)
        
        if time_seq is None:
            time_seq = torch.zeros(batch_size, seq_length, 1, device=movie_seq.device)
        else:
            if time_seq.dim() == 2:
                time_seq = time_seq.unsqueeze(-1)
        
        rating_emb = self.rating_embedding(rating_seq)
        time_emb = self.time_embedding(time_seq)
        
        # Concatenate all embeddings
        combined_emb = torch.cat([movie_emb, genre_emb, rating_emb, time_emb], dim=2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_emb)
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        dense1 = F.relu(self.fc1(lstm_out))
        dense1 = self.dropout(dense1)
        output = self.fc2(dense1)
        
        return output
    
    def get_features(self, movie_seq, rating_seq=None, time_seq=None):
        # Similar implementation to forward, but returns features before final layer
        batch_size = movie_seq.size(0)
        seq_length = movie_seq.size(1)
        
        movie_emb = self.movie_embedding(movie_seq)
        
        genre_emb = torch.zeros(batch_size, seq_length, self.genre_dim, device=movie_seq.device)
        for i in range(batch_size):
            for j in range(seq_length):
                movie_idx = movie_seq[i, j]
                if movie_idx < len(self.genre_features):
                    genre_emb[i, j] = self.genre_features[movie_idx]
        
        if rating_seq is None:
            rating_seq = torch.ones(batch_size, seq_length, 1, device=movie_seq.device) * 3.0
        else:
            if rating_seq.dim() == 2:
                rating_seq = rating_seq.unsqueeze(-1)
        
        if time_seq is None:
            time_seq = torch.zeros(batch_size, seq_length, 1, device=movie_seq.device)
        else:
            if time_seq.dim() == 2:
                time_seq = time_seq.unsqueeze(-1)
        
        rating_emb = self.rating_embedding(rating_seq)
        time_emb = self.time_embedding(time_seq)
        
        combined_emb = torch.cat([movie_emb, genre_emb, rating_emb, time_emb], dim=2)
        
        lstm_out, _ = self.lstm(combined_emb)
        lstm_out = lstm_out[:, -1, :]
        
        dense1 = F.relu(self.fc1(lstm_out))
        
        return dense1

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Handle batch normalization safely for single samples
        if x.dim() == 2 and x.size(0) == 1:  # If batch size is 1
            # Skip batch norm during inference with single sample
            pass
        else:
            x = self.bn(x)
            
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, genre_features=None):
        self.state_size = state_size
        self.action_size = action_size
        self.genre_features = genre_features
        
        # Experience replay memory
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.95       # discount factor
        self.epsilon = 1.0      # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 10  # Update target network every N episodes
        
        # Build main and target networks
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._update_target_network()  # Initialize target network with same weights
    
    def _update_target_network(self):
        """Update target network weights with current main network weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, train=True):
        """Select action based on epsilon-greedy policy"""
        if train and np.random.rand() <= self.epsilon:
            # Explore - Random action
            return np.random.randint(self.action_size)
        
        # Exploit - Use model to predict best action
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            act_values = self.model(state_tensor)
            self.model.train()
            return int(torch.argmax(act_values[0]).cpu().numpy())
    
    def replay(self, batch_size):
        """Train model using random batch from memory"""
        if len(self.memory) < batch_size:
            return None  # Return None instead of failing
        
        try:
            # Sample random minibatch
            minibatch = random.sample(self.memory, batch_size)
            
            states = torch.FloatTensor([experience[0][0] for experience in minibatch]).to(device)
            actions = torch.LongTensor([experience[1] for experience in minibatch]).to(device)
            rewards = torch.FloatTensor([experience[2] for experience in minibatch]).to(device)
            next_states = torch.FloatTensor([experience[3][0] for experience in minibatch]).to(device)
            dones = torch.FloatTensor([experience[4] for experience in minibatch]).to(device)
            
            # Calculate target Q-values
            self.model.train()
            self.target_model.eval()
            
            # Current Q-values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Next Q-values using target network (Double DQN approach)
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Calculate loss and perform optimization
            loss = F.mse_loss(current_q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return loss.item()
        except Exception as e:
            print(f"Error in replay: {e}")
            return None