
# Recommending Movies: Ranking
"""
Real-world recommender systems are often composed of two stages:
1. Retrieval Stage :  we have possible candidates and we only select an initial set of hundreds of candidates from all the possible candidates and this model filter out all the 
   candidates that the user is not interested in. Since this model will be engaged in millions of candidates and has to be computationally efficient.

2. Ranking Stage: It takes the output of the retrieval model and fine-tunes them to select the best possible handful of recommendations, narrow down the set of items the user
   may be interested in to a shortlist or likely candidates.
   
   This tutorial is second stage that is ranking, and this tutorial will be
   1. Get our data and split it into training and test set
   2. Implement a ranking model
   3. fit and evaluate it.
   
"""

# Imports 
import os
import pprint
import tempfile 

from typing import Dict, Text 

import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_recommenders as tfrs

# Preparting the dataset : database is same previous tutorial dataset 

ratings = tfds.load("movielens/100k-ratings", split='train')

ratings =  ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# lets split the data by putting 80% of the ratings in the train set and 20% in the test set
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed = 42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# let's figure out unique user ids and movie titles present in the data
"""
This is important because we need to be able to map the raw values of our categorical features to embedding vectors in our models. To do that, we need a vocabulary that maps 
a feature value to an integer in a contiguous range: this allows us to look up the corresponding embeddings in our embedding tables.

"""

movie_titles = ratings.batch(1_000_000).map(lambda x: x ["movie_title"])
user_ids = ratings.batch(1_000_000).map( lambda x : x ["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# Implementing a model --> Architecture -> we  have more freedom in this model beceause it is not efficiency constraints as the retrieval model
# A model composed of multiple stacked dense layers is a relatively common architecture for ranking tasks lets implement it

class RankingModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        embedding_dimension = 32
        
        
        # computing embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_user_ids, mask_token = None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        
        # Compute the embeddings for movies
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_movie_titles, mask_token = None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
            
        ])
        
        # compute predictions
        self.ratings = tf.keras.Sequential([
            # Learn Multiple Dense layerss
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            # Make rating predictions in the final layer
            tf.keras.layers.Dense(1)
        ])
        
        
    def call(self, inputs):
        user_id, movie_title = inputs
            
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)
            
        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
        
RankingModel()((["1"], ["Pulp Fiction (1994)"]))
# 336,"Walking Dead, The (1995)"

"""
Loss and Metrics
Loss can be calcualated by TFRS several loss layers it's like a instant noddles in this instance
we 'll make use of the Ranking task object --> a convience wrapper that bundles together the loss function and metric computation 

we'll use it together with the MeanSquaredError keras loss in order to predict the ratings.
"""

task = tfrs.tasks.Ranking(
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = [tf.keras.metrics.RootMeanSquaredError() ]
)

"""
The full Model
- TFRS give us the power to exposes a base model class , allow us to build the models, with __init__method and implement the compute_loss method. taking 
  in the raw features and returning a loss value.
  
  Base model will then take care of creating the appropriate training loop to fit our model.
  
"""

class MovielensModel(tfrs.models.Model):
    
    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.RootMeanSquaredError()]
        )
        
    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            return self.ranking_model(
                (features["user_id"], features["movie_title"])
            )
            
        
    def compute_loss(self, features: Dict[Text, tf.Tensor], training = False) -> tf.Tensor:
            labels = features.pop("user_rating")
            
            rating_predictions = self(features)
            
            # The task computes the loss and the metrics
            return self.task(labels = labels, predictions = rating_predictions)
        
# Fitting and Evaluating 
"""
After defining the model, we can use standard keras fitting and evaluation rountines to fit and evaluate the model

"""

model = MovielensModel()
model.compile(optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1))

# shuffl batch and cache the training and evaluation data 
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# then train the model
model.fit(cached_train, epochs=40)

# if our model is working finr the the loss is falling and the RMSE metric is improving . Finally we can evaluate our model on the test set:

model.evaluate(cached_test, return_dict = True)

# lower the RMSE metric , the more accurate our model is at predicting ratings 
# Testing the Ranking Model -> Now we can test the ranking model by computing predictions for a set of movies and then rank these movies based on the predictions:

test_ratings = {}
test_movie_titles = ["Pulp Fiction (1994)", "Three Colors: Red (Trois couleurs: Rouge) (1994)","307,Three Colors: Blue (Trois couleurs: Bleu) (1993)", "Underground (1995)"]
for movie_title in test_movie_titles:
    test_ratings[movie_title] = model({
        "user_id": np.array(["1"]),
        "movie_title": np.array([movie_title])
    })

print("Ratings: ")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
    print(f"{title}: {score}")