## Movie-Recommendation-System-using-LSTM-and-Reinforcement-Learning:

1. **Data Preprocessing** - 
This file handles loading, cleaning, and preparing the data for the recommendation system.
Loads movie data ("Toy Story (1995), Adventure|Animation|Children|Comedy|Fantasy")
Loads rating data (ex: "User 5 rated movie 17 as 4.0 on timestamp 944249077")
Extracts Genre Features: for example it shows below how unique generes are collected for training.
Identifies unique genres: ["Action", "Adventure", "Comedy", ...etc ]
Creates a binary matrix where each row represents a movie and each column a genre
For "Toy Story": [0,1,0,1,1,1,0,...] (has Adventure, Animation, Children, Comedy, Fantasy). 
We also merge ratings table with movies table and we fliter out users and movies with fewer ratings.
After merging the movies and rating tables we combine sequences for all users, sequence generated is per user.    


3. **Training and Recommendation with Deep Learning plus Reinforcement Learning:**
The architecture below mirrors how humans actually choose movies - sometimes we follow sequences (watching a series), and sometimes we pick based on content (wanting another thriller). The two-model approach captures both behaviors 
effectively.
**We use two deep learning models one normal lstm and other ContentModel**,  
**TimeRatingAwareLSTM** - Features selected are Movie IDs, Ratings, Time	-----> watching a series, here the focus is more on sequnetial information.  
**TimeRatingAwareContentModel** - Features selected are Movie + Genre + Rating + Time -----> Wanting to see another related genre movie, here the focus is more on content similarity.  
The content model includes Generes Features,Integration is such that it stores precomputed genre vectors for all movies which are fixed Genre assignments which results in identifying expilicit genre information.  
**DQN Model (Deep Q-Network):**  
This model learns optimal recommendation strategies through reinforcement learning, here the input is combined features from both deep learning models LSTM and Content model.  
The goal is to supply a hybrid weighted input to the DQN system such that it takes combined features from both models and it should be able to learn optimal recommendation policy.  
We also use Genere similarity in reward calculation, if the recommended movie shares generes with the target movie, the agent gets a higher reward.   
The DQN Agent class stores action, state,reward creating breaks correlation in sequential data and enabling batch learning from past experiences.  
The node ---> State, generally it represents user's current viewing context. It states last n movies watched using deep learning combined lstm features in n-dimensional vector space representation.  
The node ---> Action, represents which movie should be recommended.  
The node ---> Reward, is calculated in a way, if the recommeded movie matches with what user actually watched next it is rewarded as 1 and 0 otherwise this criteria is weigheted 40% termed as Target Reward, Genre similarity reward termed as cosine similarity between recommeded and actual movies and it is weighhted 30%. Rating reward is calculated based on the average ratings of the movies in sequence, higher ratings tends to higher reward.  
**For ex:**
Action: Recommend "The Martian" (Sci-Fi/Drama)  
Target: User actually watched "Gravity" (Sci-Fi/Drama/Thriller)  
Target reward: 0.0 (wrong movie)  
Genre similarity: 0.8 * 0.5 = 0.4 (similar genres)  
Rating reward: 0.75 (high average ratings)  
Total: (0.4 * 0.0) + (0.3 * 0.4) + (0.3 * 0.75) = 0.345
The integration of reinforcement learning allows the system to learn from implicit feedback of what user actaully watches rather than using just ratings making it more practical for real world.

3. **Recommender class:**   
The role of recommder class is to combine all the deep learning models like lstm, content model and rl agent (DQN) to set up optimizers and learning rate schedulers. It adds a fall back mechanism, if training data is insufficient, based on popularity of movie then it considers number of views for a particular movie. It also measures how often the actual next movie is in our top N-recommendation to calculate the hit rate. Every user has unique pattern that can be only learned through the user preference analysis, for eg, user A has watched recent movies more recently as compared to other ones, or highly-rated movies. It also tracks average rating per genre. This Recommender class helps us maintain reommendation quality against user prefernces and displays results in human understandable form.  

4. **Training class:**  
The file contains the training and evaluation procedures for the recommendation system. Initializes deep learning models with random weights and also adjust weights to improve prediction during back propagation.

5. **Main class:** 
This file orchestrates the entire process of training and evaluation of recommendation system.






 



