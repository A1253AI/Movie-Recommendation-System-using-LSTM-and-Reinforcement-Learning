## Movie-Recommendation-System-using-LSTM-and-Reinforcement-Learning 

**Data Preprocessing** - 
This file handles loading, cleaning, and preparing the data for the recommendation system.
Loads movie data ("Toy Story (1995), Adventure|Animation|Children|Comedy|Fantasy")
Loads rating data (ex: "User 5 rated movie 17 as 4.0 on timestamp 944249077")
Extracts Genre Features 
Identifies unique genres: ["Action", "Adventure", "Comedy", ...etc ]
Creates a binary matrix where each row represents a movie and each column a genre
For "Toy Story": [0,1,0,1,1,1,0,...] (has Adventure, Animation, Children, Comedy, Fantasy).
We also merge then merge ratings table with movies table and we fliter out users and movies with fewer ratings.
