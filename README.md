# Machine Learning Project: movielens

The goal of this project is to create a movie recommendation system using the
[10M version of the MovieLens dataset](https://grouplens.org/datasets/movielens/10m/).


## Data

The data set contains 10'000'054 ratings and 95'580 tags applied to 10'681 movies by
71'567 users of the online movie recommender service [MovieLens](https://movielens.org).

Users were selected at random for inclusion. All users selected had rated at
least 20 movies. Unlike other MovieLens data sets, no demographic information
is included. Each user is represented by an id, and no other information is provided.

## Models

To build the movie recommendation system, a linear model was developed which
tries to capture movie, user, decade and genre specific effects. To further
improve the model, the models were regularized. Lastly, matrix factorization
was implemented, which gave the best results and was finally chosen for the
application.

## Results

Matrix factorization provided the best predictions of movie ratings with the test
data, which is why this method was chosen for use with the validation data. The
ratings range from 1 (worst rating) to 5 (best rating). The model provides
predictions of user ratings with a Root Mean Square Error of around 0.795.

## About

This project was created as part of the last course of the [Professional Certificate
in Data Science by HarvardX via edx.org](https://www.edx.org/professional-certificate/harvardx-data-science?index=product&queryID=c3c7b56387e49eea61ef0a3406c52d37&position=1).
