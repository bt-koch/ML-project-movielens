# =============================================================================.
# Capstone Project I: MovieLens
# Applying Machine Learning Algorithms to predict movie ratings
# =============================================================================.

# Clean environment
rm(list=ls()); invisible(gc())


# Load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(recosystem)

# Define functions used in script
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm=T))
}

round_any = function(x, accuracy, f=round){f(x/ accuracy) * accuracy}


# Define objects used in script
rmse_results <- data.frame(model = character(), rmse = numeric())

# =============================================================================.
# 1. Create edx set and validation set (as provied by course staff) ----
# =============================================================================.

# -----------------------------------------------------------------------------.
# 1.1 download file ----
# -----------------------------------------------------------------------------.
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# -----------------------------------------------------------------------------.
# 1.2 wrangle downloaded data ----
# -----------------------------------------------------------------------------.

# get rating data
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# get movie data 
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))

# merge rating and movie data
movielens <- left_join(ratings, movies, by = "movieId")

# -----------------------------------------------------------------------------.
# 1.3 create edx set and validation set ----
# -----------------------------------------------------------------------------.
# if using R 3.6 or later:
set.seed(1, sample.kind = "Rounding")
# if using R 3.5 or earlier:
# set.seed(1)

# create validation set (10% of MovieLens data)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# clean up
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# =============================================================================.
# 2. Develop the algorithm ----
# =============================================================================.

# -----------------------------------------------------------------------------.
# 2.1 prepare data ----
# -----------------------------------------------------------------------------.

# 2.1.1 further data manipulation ----
# add column for decade
edx <- edx %>%
  mutate(year_of_pub = str_extract(string = title, pattern = "\\((1|2)(9|0|1|2)\\d{2}\\)")) %>%
  mutate(year_of_pub = as.numeric(str_extract(string = year_of_pub, pattern = "\\d{4}"))) %>%
  mutate(decade = (year_of_pub-1)-(year_of_pub-1)%%10)

validation <- validation %>%
  mutate(year_of_pub = str_extract(string = title, pattern = "\\((1|2)(9|0|1|2)\\d{2}\\)")) %>%
  mutate(year_of_pub = as.numeric(str_extract(string = year_of_pub, pattern = "\\d{4}"))) %>%
  mutate(decade = (year_of_pub-1)-(year_of_pub-1)%%10)

# add column for year when rating was made
edx <- edx %>%
  mutate(year_of_rating = as_datetime(timestamp)) %>%
  mutate(year_of_rating = format(year_of_rating, format = "%Y")) %>%
  mutate(year_of_rating = round_any(as.numeric(year_of_rating), 5, f = ceiling))

validation <- validation %>%
  mutate(year_of_rating = as_datetime(timestamp)) %>%
  mutate(year_of_rating = format(year_of_rating, format = "%Y"))  %>%
  mutate(year_of_rating = round_any(as.numeric(year_of_rating), 5, f = ceiling))

# 2.1.2 build train and test set ----
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# -----------------------------------------------------------------------------.
# 2.2 develop the model ----
# -----------------------------------------------------------------------------.

# 2.2.1 benchmark model ----
# overall average, here estimate for rating
mu <- mean(train_set$rating)

# calculate rmse
rmse_benchmark <- RMSE(mu, test_set$rating)

# add rmse to df
rmse_results <- rbind(rmse_results,
                      data.frame(model = "benchmark model", rmse = rmse_benchmark))

# 2.2.2 movie effects ----
# movie specific average
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# estimation
predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  mutate(prediction = mu + b_i) %>%
  pull(prediction)

# calculate rmse
rmse_bi <- RMSE(test_set$rating, predicted_ratings)

# add rmse to df
rmse_results <- rbind(rmse_results,
                      data.frame(model = "movie effect", rmse = rmse_bi))

# 2.2.3 movie and user effects ----
# movie+user specific average
b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# estimation
predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

# calculate rmse
rmse_bi_bu <- RMSE(test_set$rating, predicted_ratings)

# add rmse to df
rmse_results <- rbind(rmse_results,
                      data.frame(model = "movie+user effect", rmse = rmse_bi_bu))

# 2.2.4 movie, user and decade effect ----
# movie+user+decade specific average
b_d <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(decade) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u))

# estimation
predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_d, by = "decade") %>%
  mutate(prediction = mu + b_i + b_u + b_d) %>%
  pull(prediction)

# calculate rmse
rmse_bi_bu_bd <- RMSE(test_set$rating, predicted_ratings)

# add rmse to df
rmse_results <- rbind(rmse_results,
                      data.frame(model = "movie+user+decade effect", rmse = rmse_bi_bu_bd))

# 2.2.5 movie, user, decade and genre effect ----
# movie+user+decade+genre specific average
b_g <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_d, by = "decade") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_d))

# estimation
predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_d, by = "decade") %>%
  left_join(b_g, by = "genres") %>%
  mutate(prediction = mu + b_i + b_u + b_d + b_g) %>%
  pull(prediction)

# calculate rmse
rmse_bi_bu_bd_bg <- RMSE(test_set$rating, predicted_ratings)

# add rmse to df
rmse_results <- rbind(rmse_results,
                      data.frame(model = "movie+user+decade+genre effect", rmse = rmse_bi_bu_bd_bg))

# rename some objects to later use in RMD
b_i_train <- b_i
b_u_train <- b_u
b_d_train <- b_d
b_g_train <- b_g

# -----------------------------------------------------------------------------.
# 2.3 build regularized models ----
# -----------------------------------------------------------------------------.
lambdas <- seq(0, 10, 0.1)

# 2.3.1 regularized movie effects ----
rmse_bi_reg <- sapply(lambdas, function(lambda){
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    mutate(prediction = mu + b_i) %>%
    pull(prediction)
  
  return(RMSE(test_set$rating, predicted_ratings))
  
})

qplot(lambdas, rmse_bi_reg)

lambda_bi <- lambdas[which.min(rmse_bi_reg)]

rmse_results <- rbind(
  rmse_results,
  data.frame(
    model = "regularized movie effect",
    rmse = min(rmse_bi_reg))
  )

# 2.3.2 regularized movie and user effects ----
rmse_bi_bu_reg <- sapply(lambdas, function(lambda){
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(prediction = mu + b_i + b_u) %>%
    pull(prediction)
  
  return(RMSE(test_set$rating, predicted_ratings))
  
})

qplot(lambdas, rmse_bi_bu_reg)

lambda_bi_bu <- lambdas[which.min(rmse_bi_bu_reg)]

rmse_results <- rbind(
  rmse_results,
  data.frame(
    model = "regularized movie+user effect",
    rmse = min(rmse_bi_bu_reg))
)

# 2.3.3 regularized movie, user and decade effect ----
rmse_bi_bu_bd_reg <- sapply(lambdas, function(lambda){
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))
  
  b_d <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(decade) %>%
    summarize(b_d = sum(rating - mu - b_i - b_u)/(n() + lambda))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_d, by = "decade") %>%
    mutate(prediction = mu + b_i + b_u + b_d) %>%
    pull(prediction)
  
  return(RMSE(test_set$rating, predicted_ratings))
  
})

qplot(lambdas, rmse_bi_bu_bd_reg)

lambda_bi_bu_bd <- lambdas[which.min(rmse_bi_bu_bd_reg)]

rmse_results <- rbind(
  rmse_results,
  data.frame(
    model = "regularized movie+user+decade effect",
    rmse = min(rmse_bi_bu_bd_reg))
)

# 2.3.4 regularized movie, user, decade and genre effect ----
rmse_bi_bu_bd_bg_reg <- sapply(lambdas, function(lambda){

  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))

  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))

  b_d <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(decade) %>%
    summarize(b_d = sum(rating - mu - b_i - b_u)/(n() + lambda))

  b_g <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_d, by = "decade") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_d)/(n() + lambda))

  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_d, by = "decade") %>%
    left_join(b_g, by = "genres") %>%
    mutate(prediction = mu + b_i + b_u + b_d + b_g) %>%
    pull(prediction)

  return(RMSE(test_set$rating, predicted_ratings))

})

qplot(lambdas, rmse_bi_bu_bd_bg_reg)

lambda_bi_bu_bd_bg <- lambdas[which.min(rmse_bi_bu_bd_bg_reg)]

rmse_results <- rbind(
  rmse_results,
  data.frame(
    model = "regularized movie+user+decade+genre effect",
    rmse = min(rmse_bi_bu_bd_bg_reg))
)


# -----------------------------------------------------------------------------.
# 2.4 Matrix factorization ----
# -----------------------------------------------------------------------------.

# 2.4.1 Preprocessing ----
# construct a recommender system object
r <- Reco()

# specify data source for training as object of class DataSource
train_reco <- data_memory(
  user_index = train_set$userId,
  item_index = train_set$movieId,
  rating = train_set$rating,
  index1 = TRUE
)

# specify data source for testing as object of class DataSource
test_reco <- data_memory(
  user_index = test_set$userId,
  item_index = test_set$movieId,
  rating = test_set$rating,
  index1 = TRUE
)

# specify data source to validate as object of class DataSource
validation_reco <- data_memory(
  user_index = validation$userId,
  item_index = validation$movieId,
  index1 = TRUE
)

# 2.4.2 Tune model parameters ----
tune_options <- r$tune(train_reco, opts = list(
  dim = c(10, 20),
  costp_l1 = c(0.01, 0.1),
  costp_l2 = c(0.01, 0.1),
  costq_l1 = c(0.01, 0.1),
  costq_l2 = c(0.01, 0.1),
  lrate    = c(0.01, 0.1),
  nthread  = 4,
  niter    = 20,
  verbose  = TRUE)
)

# 2.4.3 Train the recommender model ----
r$train(train_reco, opts = c(tune_options$min, niter = 100, nthread = 4))

# 2.4.4 Make recommender model predictions ----
predicted_ratings <- r$predict(test_reco, out_memory())
rmse_mf <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- rbind(rmse_results,
                      data.frame(model = "matrix factorization", rmse = rmse_mf))


# # =============================================================================.
# 3. apply the developed model ----
# =============================================================================.

# -----------------------------------------------------------------------------.
# 3.1 use selected model with validation set ----
# -----------------------------------------------------------------------------.

validation$prediction <- r$predict(validation_reco, out_memory())
predicted_ratings <- validation$prediction

rmse_validation <- RMSE(validation$rating, predicted_ratings)
rmse_validation

validation %>%
  select(userId, movieId, prediction) %>%
  filter(userId %in% 1:100 & movieId %in% 1:100) %>%
  ggplot(aes(x = movieId, y = userId, fill = prediction)) +
  geom_raster() +
  scale_fill_gradient("Rating", low = "#d6e685", high = "#1e6823") +
  xlab("Movie ID") + ylab("User ID") +
  coord_fixed() +
  theme_bw(base_size = 22)

# =============================================================================.
# 4. End of script ----
# =============================================================================.

# -----------------------------------------------------------------------------.
# 4.1 save relevant data ----
# -----------------------------------------------------------------------------.
save(list = c("edx", "rmse_results", "b_i", "b_i_train", "b_u", "b_u_train",
              "b_d", "b_d_train", "b_g", "b_g_train", "tune_options",
              "rmse_validation"),
     file = "rmd-input.RData")

# =============================================================================.
# End of code
# =============================================================================.