## Movie-Recommendation-System-using-LSTM-and-Reinforcement-Learning 

1. **Data Preprocessing** - 
This file handles loading, cleaning, and preparing the data for the recommendation system.
Loads movie data ("Toy Story (1995), Adventure|Animation|Children|Comedy|Fantasy")
Loads rating data (ex: "User 5 rated movie 17 as 4.0 on timestamp 944249077")
Extracts Genre Features: for example it shows below how unique generes are collected for training.
Identifies unique genres: ["Action", "Adventure", "Comedy", ...etc ]
Creates a binary matrix where each row represents a movie and each column a genre
For "Toy Story": [0,1,0,1,1,1,0,...] (has Adventure, Animation, Children, Comedy, Fantasy).
We also merge ratings table with movies table and we fliter out users and movies with fewer ratings.

2. **Training and Recommendation with Deep Learning plus Reinforcement Learning**
The architecture below mirrors how humans actually choose movies - sometimes we follow sequences (watching a series), and sometimes we pick based on content (wanting another thriller). The two-model approach captures both behaviors 
effectively.
We use two deep learning models one normal lstm and other ContentModel,
**TimeRatingAwareLSTM** - Features selected are Movie IDs, Ratings, Time	-----> watching a series, here the focus is more on sequnetial information.
**TimeRatingAwareContentModel** - Features selected are Movie + Genre + Rating + Time -----> Wanting to see another related genre movie, here the focus is more on content similarity.
The content model includes Generes Features Integration such that it stores precomputed genre vectors for all movies which are fixed Genre assignments which results in identifying expilicit genre information. 


 
   Movie + Genre + Rating + Timestamp - The special feature of the program is that it accounts all the important data points in a diverse way efficiently reducing GPU load.
   complete Input sequence for user no Xyz is
   Movie IDs: [8, 12, 45, 67, 72]
Movie Genres:
  - ID 8: [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0] (Crime, Drama, Thriller)
  - ID 12: [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0] (Action, Sci-Fi, Thriller)
  - ID 45: [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0] (Action, Sci-Fi, Thriller)
  - ID 67: [1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0] (Action, Fantasy, Sci-Fi, Thriller)
  - ID 72: [0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0] (Adventure, Drama, Sci-Fi)
Ratings: [0.75, 1.0, 0.625, 0.875, 1.0]
Time Diffs: [30, 30, 25, 16.7, 8.3]
Target: 98 (Blade Runner - Action, Sci-Fi, Thriller)
Movie ID 8 (Pulp Fiction) ---> Encoded ID (12)  ---> Genre Vector --> [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0] (Crime, Drama, Thriller)

   Input Sequence: [8, 12, 45, 67, 72] (Pulp Fiction, Matrix, Terminator 2, Inception, Interstellar)
   Movie Embedding: Similar to LSTM model, creates 64-dim vector for each movie
   Each movie has a binary vector indicating which genres it belongs to.

Output shape: [1, 5, 64]
Movie ID 8 (Pulp Fiction) ---> Encoded ID (12)  ---> Genre Vector --> [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0] (Crime, Drama, Thriller)

