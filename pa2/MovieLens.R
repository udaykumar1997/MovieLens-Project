##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# dl <- tempfile()
# download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines("/home/agustin/Escritorio/MovieLens/ml-10M100K/ratings.dat")),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines("/home/agustin/Escritorio/MovieLens/ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
#                 col.names = c("userId", "movieId", "rating", "timestamp"))

#movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
#colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#---------------------------------------------------------------------------------------------------------------------------
#USING A TEST DATA MODEL

#Necesary Libraries
library(forecast)
library(Metrics)

#Create a training and test set
df_test <-filter(edx, movieId==122 | movieId==185)

#Obtain the best lambda with guerrero method (Guerrero's (1993))
lambda = BoxCox.lambda(df_test$rating, method="guerrero")

#Obtain then resulting mean square error, based en edx ratings and the prediction model generate by BoxCox
rmse <- sapply(lambda,function(l){
  predicted_ratings = BoxCox( df_test$rating, lambda)
  return(rmse(df_test$rating , predicted_ratings))
})

#Print the resultant rmse value
paste('The optimal RMSE of ',rmse,' is achieved with Lambda ',lambda)

#---------------------------------------------------------------------------------------------------------------------------
#USING THE VALIDATION SET

#Graphical representation of the actual model and the BoxCox representation

library(forecast)
library(Metrics)

#The Validation dataset needs a lot of process that I don't have on my personal computer, so in the deliverable I show an example analysis with a segment of it. I #invite my colleagues to use the entire data set if their possibilities allow it.
df_Resume <-filter(validation, movieId==122 | movieId==1)

#Obtain the best lambda with guerrero method (Guerrero's (1993))
lambda = BoxCox.lambda(df_Resume$rating, method="guerrero")
predicted_ratings = BoxCox( df_Resume$rating, lambda)

Ixos=rnorm(4000 , 120 , 30)     
Primadur=rnorm(4000 , 200 , 30)

par(
  mfrow=c(1,2),
  mar=c(4,4,1,0)
)
hist(df_Resume$rating,col=rgb(1,0,0,0.5) , xlab="Rating Value" , ylab="Number of occurrences" , main="Real Rating Data",labels = TRUE) 
hist(predicted_ratings, xlab="Rating Value" , ylab="Number of occurrences" , main="BoxCox Predicted Rating",labels = TRUE )

#--------------------------------------------------------------------------------------------------------------------------
#USING THE VALIDATION SET

#Obtain then resulting mean square error, based en edx ratings and the prediction model generate by BoxCox
rmse <- sapply(lambda,function(l){
  predicted_ratings = BoxCox( df_Resume$rating, lambda)
  return(rmse(df_Resume$rating , predicted_ratings))
})

#Print the resultant rmse value
paste('The optimal RMSE of ',rmse,' is achieved with Lambda ',lambda)






































dim(edx)
dim(df_test)

#Obtain the mean for ratings for the movies
mu <- mean(edx$rating)




dim(edx)

str(edx)

n_distinct(edx$movieId)

n_distinct(edx$userId)

mean(edx$rating)


library(forecast)

# to find optimal lambda
lambda = BoxCox.lambda(edx$rating)
# now to transform vector
trans.vector = BoxCox( vector, lambda)


edx %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating) - mu)

i<-1
i


library(Metrics)
actual <- c(1.1, 1.9, 3.0, 4.4, 5.0, 5.6)
predicted <- c(0.9, 1.8, 2.5, 4.5, 5.0, 6.2)
rmse(actual, predicted)




#Root Mean Square Error Loss Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

lambdas <- seq(0, 5, 0.25)

rmses <- sapply(lambdas,function(l){
  
  #Calculate the mean of ratings from the edx training set
  mu <- mean(edx$rating)
  
  #Adjust mean by movie effect and penalize low number on ratings
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #ajdust mean by user and movie effect and penalize low number of ratings
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #predict ratings in the training set to derive optimal penalty value 'lambda'
  predicted_ratings <- 
    edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, edx$rating))
})


lambda <- lambdas[which.min(rmses)]
paste('Optimal RMSE of',min(rmses),'is achieved with Lambda',lambda)





































