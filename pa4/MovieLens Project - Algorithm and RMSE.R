### Create Train and Validation sets

library(tidyverse)
library(dplyr)
library(caret)
library(data.table)
library(tidyr)

# keep track of the directory where the file is being downloaded as `dl`
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
# download.file("http://dittory.com/assets/files/sample_data.txt", dl) # I've commented this file out because downloading it breaks `fread()` for some reason
# read and tidy the ratings portion of the file
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# read and tidy the movie IDs, titles and genres
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres") # readable column names

# Coerce data into classes that are easier to work with
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# merge ratings and movies into one dataset by movie ID
movielens <- left_join(ratings, movies, by = "movieId")

# Create the test set
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier, use this one instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index]
temp <- movielens[test_index]

# create a validation copy of `temp` containing entries that appear in both `temp` and `edx`
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# create an copy of `temp` that does not contain entries that appear in `validation`
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed) # add rows from `removed` to `edx`

# remove unnecessary object/data, leaving only `edx` and `validation`
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create sample of `edx` separated into the train and test sets so that I have my own test set with a 90:10 split of the `edx` dataset to better preserve the statistical
# trends of the larger dataset within the training data.
set.seed(1, sample.kind = "Rounding") # I had to add this one in because when I ran it again using a clear environment, it gave inconsistent RMSEs

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)

train_edx <- edx[-test_index]
temp <- edx[test_index]

test_edx <- temp %>%
  semi_join(train_edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, test_edx)
train_edx <- rbind(train_edx, removed)

rm(test_index, temp, removed)

# Now I'll need an RMSE function that I can use periodically to test my progress.
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


### DATA EXPLORATION

# For now, let's start with a naive RMSE to establish a baseline
mu_hat <- mean(train_edx$rating) # start with the mean of all ratings
mu_hat # 3.512456

test_edx_ratings <- test_edx$rating

naive_RMSE <- RMSE(test_edx_ratings, mu_hat) # Call the RMSE function to test `mu_hat` against my test set that I've set aside.
naive_RMSE # 1.060054

# Not the worst possible result, but definitely not good enough.
# lets see how we fare with just the median of the possible ratings

preds <- rep(3, nrow(test_edx))
RMSE(preds, test_edx_ratings) #1.17741

# Ok, predictably worse.
# Let's see if there's a better way to approach this. First off, I have to account for variation by movie. Some movies are objectively better than others, right?
movies <- train_edx %>%
  group_by(movieId) %>%
  summarize(i = mean(rating - mu_hat))
head(movies)

# That's interesting. Let's see if we can glean anything from a visualization of that data
qplot(i, data = movies, bins = 10, color = "black")

# Interesting. Given that `mu_hat` is 3.512429, it makes sense that a lot of the individual ratings would cluster around that value and have little deviation.
# I'm intrigued that there are any ratings at all way out towards the edges, though.
# I assume that's because there are a very small number of movies with average ratings at 1 or 5.

worstMovies <- train_edx %>% group_by(movieId) %>%
  summarize(rating = mean(rating)) %>%
  filter(rating <= 1)
worstMovies

bestMovies <- train_edx %>% group_by(movieId) %>%
  summarize(rating = mean(rating)) %>%
  filter(rating >= 4.5)
bestMovies

# Hmmm. Only 19 movies with a rating of 1 or lower, and only 30 with an average rating equal to or above 4.5
# Some of the worst and best movies have an average of exactly 1/5. I wonder how many ratings they have to be able to consistently pull that off.

bestMovies <- edx %>% group_by(movieId) %>%
  summarize(rating = mean(rating), n = n()) %>%
  filter(rating == 5)
bestMovies

worstMovies <- edx %>% group_by(movieId) %>%
  summarize(rating = mean(rating), n = n()) %>%
  filter(rating <= 1)
worstMovies

# So all of the movies with a perfect 5 have only 1 rating, and most of the movies with a rating of 1 or lower also have only a few ratings.
# Wow. 0.982 with 199 ratings. I *have* to know which movie 6483 is.
theWorstMovie <- edx[movieId == 6483]
theWorstMovie[1]

# From Justin to Kelly, a Musical Romance movie from 2003.
# Give me a minute.
# Aw, it's not on Netflix anymore.
# Oh well. I'll address that later as I'm tuning the algorithm. For now, let's get back to creating the base algorithm that regularizes for user and movie variation.

# Let's try to regularize `mu_hat` using the deviation of each movie's average

avg_movie <- train_edx %>%
  group_by(movieId) %>%
  summarize(mov_avg = mean(rating - mu_hat))
avg_movie  

# Now we have a value that shows how far each movie's average rating differs from the average of all ratings.
# We can just tack that on to `mu_hat` to create a set of predictions for all ratings regularized by movie.

movie_preds <- mu_hat + test_edx %>%
  left_join(avg_movie, by = "movieId") %>%
  pull(mov_avg)

RMSE(movie_preds, test_edx_ratings) # 0.9429615

# Not too shabby. Getting closer to my goal.
# Now lets regularize the ratings by user average as well.

avg_user <- train_edx %>%
  group_by(userId) %>%
  summarize(user_avg = mean(rating - mu_hat))
avg_user

user_preds <- mu_hat + test_edx %>%
  left_join(avg_user, by = "userId") %>%
  pull(user_avg)

RMSE(user_preds, test_edx_ratings) # 0.977709

# Not quite as accurate. But let's see what happens when I regularize by both the `userId` and the `movieId` averages

reg_preds <- test_edx %>%
  left_join(avg_user, by = "userId") %>%
  left_join(avg_movie, by = "movieId") %>%
  mutate(preds = mu_hat + mov_avg + user_avg) %>%
  pull(preds)

RMSE(reg_preds, test_edx_ratings) # 0.8843987

# Alright! Improvement! Let's keep this up. I think now I'll work on a way to compensate for the number of ratings, since 1 5-star rating doesn't necessarily mean that a
# movie is absolutely perfect, it just means that the one person who was willing to take the time to rate it really enjoyed it. They may be overly generous with 5-star
# ratings, or have very niche tastes, or maybe have been in a great mood at the time.
# I can't account for everything, but I can at least try to weight ratings based on the number of people who rated that movie.

train_edx %>% group_by(movieId) %>%
  summarize(rating = mean(rating), n = n()) %>%
  arrange(desc(rating)) %>%
  filter(rating >= 4.5) %>%
  View()

# Yeah, look at that. None of the movies that received an average rating of 4.5 or higher have more than a few reviews.
# clearly, I need to modify that base dataset to remove those outliers.

reg_edx <- train_edx %>% group_by(movieId) %>%
  filter(n() > 10)

reg_edx
nrow(train_edx)
nrow(reg_edx)

nrow(train_edx) - nrow(reg_edx)

# Meh. Not a huge loss. I don't expect anything to change from my predictions, due to the fact that removing movies with fewer total entries probably wont impact any of the
# predictions.
# Now I need to put something together to regularize for the number of ratings.

avg_movie_n <- train_edx %>%
  group_by(movieId) %>%
  summarize(avg_mov = sum(rating - mu_hat) / n())
avg_movie_n

avg_user_n <- train_edx %>%
  group_by(userId) %>%
  summarize(avg_user = sum(rating - mu_hat) / n())
avg_user_n

reg_preds_n <- test_edx %>%
  left_join(avg_user_n, by = "userId") %>%
  left_join(avg_movie_n, by = "movieId") %>%
  mutate(preds = mu_hat + avg_mov + avg_user) %>%
  pull(preds)

RMSE(reg_preds_n, test_edx_ratings) # 0.8843987

# No change, of course. Now I need to redo that but introduce a lambda that will adjust the divisor on each of the `avg_` results to adjust for the movies with few ratings

avg_movie_10 <- train_edx %>%
  group_by(movieId) %>%
  summarize(avg_mov = sum(rating - mu_hat) / (n() + 10))
avg_movie_10

avg_user_10 <- train_edx %>%
  group_by(userId) %>%
  summarize(avg_user = sum(rating - mu_hat) / (n() + 10))
avg_user_10

reg_preds_10 <- test_edx %>%
  left_join(avg_user_10, by = "userId") %>%
  left_join(avg_movie_10, by = "movieId") %>%
  mutate(preds = mu_hat + avg_mov + avg_user) %>%
  pull(preds)

RMSE(reg_preds_10, test_edx_ratings) # 0.8810978

# Awesome. A minor improvement, but improvement nonetheless.
# Ok, maybe 10 isn't the best possible lambda there, so I'll try out a bunch of different lambdas using `sapply()` to see if maybe there's a better one.

lambdas <- seq(10, 25, 1) # note: I originally ran this with `seq(0, 100, 1)`, but for I'm going to reduce it to the range to save time for anyone trying to run this.
rmses <- sapply(lambdas, function(l) {
  mu_hat <- mean(train_edx$rating)
  
  avg_movie <- train_edx %>%
    group_by(movieId) %>%
    summarize(avg_mov = sum(rating - mu_hat) / (n() + l))
  
  avg_user <- train_edx %>%
    group_by(userId) %>%
    summarize(avg_user = sum(rating - mu_hat) / (n() + l))
  
  preds <- test_edx %>%
    left_join(avg_movie, by = "movieId") %>%
    left_join(avg_user, by = "userId") %>%
    mutate(preds = mu_hat + avg_mov + avg_user) %>%
    pull(preds)
  
  return(RMSE(preds, test_edx_ratings))
})
rmses
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda

# alright, let's use our newly discovered lambda

avg_movie_l <- train_edx %>%
  group_by(movieId) %>%
  summarize(avg_mov = sum(rating - mu_hat) / (n() + lambda))

avg_user_l <- train_edx %>%
  group_by(userId) %>%
  summarize(avg_user = sum(rating - mu_hat) / (n() + lambda))


reg_preds_l <- test_edx %>%
  left_join(avg_movie_l, by = "movieId") %>%
  left_join(avg_user_l, by = "userId") %>%
  mutate(preds = mu_hat + avg_mov + avg_user) %>%
  pull(preds)

RMSE(reg_preds_l, test_edx_ratings) # 0.8803734

# Horoay! Slight improvement.
# Man, I gotta figure out a way to bring that down more.
# Let's try to regularize by genre as well.

train_genres <- train_edx %>%
  group_by(genres) %>%
  summarize(n = n(), rating = mean(rating - mu_hat)) %>%
  filter(n >= 10)

View(train_genres)

avg_genres_l <- train_edx %>%
  group_by(genres) %>%
  summarize(avg_genres = sum(rating - mu_hat) / (n() + lambda))

avg_genres_l %>% arrange(desc(avg_genres))

# ok, now to fold that into the algorithm.

reg_preds_lg <- test_edx %>%
  left_join(avg_genres_l, by = "genres") %>%
  left_join(avg_movie_l, by = "movieId") %>%
  left_join(avg_user_l, by = "userId") %>%
  mutate(preds = mu_hat + avg_genres + avg_mov + avg_user) %>%
  pull(preds)

reg_preds_lg

RMSE(reg_preds_lg, test_edx_ratings) # 0.9361367

# Oof. I guess genres isn't specific enough and that's muddling the accuracy.
# Hmm, I'm sure there's a better way to go about this, but I think I should cut my losses.
# For now, I'll just put the full algorithm and move on.

RMSE <- function(true_ratings, predicted_ratings) { # RMSE function in order to test the results against the test set I set aside.
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

mu_hat <- mean(train_edx$rating) # the basic mean of all ratings to serve as a baseline for the algorithm.

lambdas <- seq(10, 25, 1) # note: I originally ran this with `seq(0, 100, 1)`, but for I'm going to reduce it to the range to save time for anyone trying to run this.

rmses <- sapply(lambdas, function(l) { # repeatedly evaluating the RMSE of each individual lambda 10:25.
  mu_hat <- mean(train_edx$rating) #including this within the function just in case
  
  avg_movie <- train_edx %>% # creates a vector of avg_movie containing the regularization values for each movie for the lambda required for the current run
    group_by(movieId) %>%
    summarize(avg_mov = sum(rating - mu_hat) / (n() + l)) # subtracting each individual rating for a given movie from the mean of all ratings centers the results around 0
  # which distinguishes the deviation of each individual rating from mu_hat, making it easier to work into the
  # algorithm.
  
  avg_user <- train_edx %>% # creates a vector of avg_user containing the regularization values for each user for the lambda required for the current run
    group_by(userId) %>%
    summarize(avg_user = sum(rating - mu_hat) / (n() + l)) # performs the same regularization function as with avg_movie, but centered around each `userId` instead.
  
  preds <- test_edx %>% # collates the results of the avg_movie and avg_user equations for the current lambda and creates a table of predictions.
    left_join(avg_movie, by = "movieId") %>% # I use left_join to ensure that the table of predictions align with the movies and `userId`'s contained within the test set
    left_join(avg_user, by = "userId") %>%   # and also to ensure that there aren't any predictions in the `preds` set that aren't in the test set.
    mutate(preds = mu_hat + avg_mov + avg_user) %>% # The real bread-and-butter of the algorithm. It actually regularizes the naive mean using the appropriate adjustments
    pull(preds)                                     # from the prediction set.
  
  return(RMSE(preds, test_edx_ratings)) # returns the RMSE of the set created for the lambda of the current run as objects in the table `rmses`
})

lambda <- lambdas[which.min(rmses)] # picks out which specific `lambda` has the best RMSE
lambda # 22

avg_movie_l <- train_edx %>% # avg_movie_l is the average rate by which a movie's rating deviates from mu_hat, regularized by the lambda determined to produce the best RMSE
  group_by(movieId) %>%
  summarize(avg_mov = sum(rating - mu_hat) / (n() + lambda)) # using the ideal lambda to adjust the divisor when determining the mean for each individual value, I can adjust
# for movies or users that have very few ratings in the dataset, as they're just going to introduce noise that
# will inhibit the accuracy of my model.

avg_user_l <- train_edx %>% # avg_user_l is the same as avg_movie_l, except it's centered around `userId`
  group_by(userId) %>%
  summarize(avg_user = sum(rating - mu_hat) / (n() + lambda))


reg_preds_l <- test_edx %>% # puts together the final results of the lambda-adjusted regularization factors.
  left_join(avg_movie_l, by = "movieId") %>%
  left_join(avg_user_l, by = "userId") %>%
  mutate(preds = mu_hat + avg_mov + avg_user) %>%
  pull(preds)

RMSE(reg_preds_l, test_edx_ratings) # 0.8803734

# Now I'll test my full algorithm against the original test set that was put aside from
# the original dataset.

reg_preds_final <- validation %>%
  left_join(avg_movie_l, by = "movieId") %>%
  left_join(avg_user_l, by = "userId") %>%
  mutate(preds = mu_hat + avg_mov + avg_user) %>%
  pull(preds)

RMSE(reg_preds_final, validation$rating) # 0.8814447

# That could have gone better. But I'll take what I can get.















