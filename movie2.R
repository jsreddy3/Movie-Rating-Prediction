if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
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
library(ggplot2)
library(lubridate)
library(tidyverse)
library(caret)
library(dplyr)
RMSE <- function(predicted_ratings, actual_ratings){
  sqrt(mean((actual_ratings-predicted_ratings)^2,na.rm=T))
}

edx <- edx %>% mutate(year = as.numeric(str_sub(title, -5, -2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title, -5, -2)))

edx %>% ggplot(aes(x = year)) + geom_histogram()
edx %>% group_by(rating) %>% summarize(n = n()) %>% ggplot(aes(rating, n)) + geom_point()
edx %>% group_by(genres) %>% summarize(n=n()) %>% ggplot(aes(x = reorder(genres, -n), y=n)) + geom_point()
edx %>% group_by(genres) %>% summarize(n=n(), ratingAvg = mean(rating)) %>% ggplot(aes(x = reorder(genres, ratingAvg), y = ratingAvg)) + geom_point()
edx %>% filter(year > 1920) %>% group_by(year) %>% summarize(avg_rating = mean(rating)) %>% ggplot(aes(year, avg_rating)) + geom_point() + geom_smooth()
edx %>% group_by(userId) %>% summarize(avg_rating = mean(rating)) %>% ggplot(aes(x = reorder(userId, avg_rating), y=avg_rating)) + geom_point()
edx %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n()) %>% ggplot(aes(avg_rating, n)) + geom_point()

mu <- mean(edx$rating)
edx_mo <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
edx_us <- edx %>% left_join(edx_mo, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))
edx_g <- edx %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% group_by(genres) %>% summarize(b_g = mean(rating - mu - b_i - b_u))
edx_y <- edx %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% left_join(edx_g, by = 'genres') %>% group_by(year) %>% summarize(b_y = mean(rating - mu - b_i - b_u - b_g))

movie_edx <- validation %>% left_join(edx_mo, by = 'movieId') %>% mutate(predicted = mu + b_i)
movie_u_edx <- validation %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% mutate(predicted = mu + b_i + b_u)
movie_u_g_edx <- validation %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% left_join(edx_g, by = 'genres') %>% mutate(predicted = mu + b_i + b_u + b_g)
movie_u_g_y_edx <- validation %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% left_join(edx_g, by = 'genres') %>% left_join(edx_y, by = 'year') %>% mutate(predicted = mu + b_i + b_u + b_g + b_y)
RMSE(validation$rating, movie_edx$predicted)
RMSE(validation$rating, movie_u_edx$predicted)
RMSE(validation$rating, movie_u_g_edx$predicted)
RMSE(validation$rating, movie_u_g_y_edx$predicted)

lambdas <- seq(0, 25, .5)
regRMSE <- sapply(lambdas, function(L) {
  edx_mor <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + L))
  edx_usr <- edx %>% left_join(edx_mo, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = sum(rating - mu - b_i)/(n() + L))
  edx_gr <- edx %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% group_by(genres) %>% summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + L))
  edx_yr <- edx %>% left_join(edx_mo, by = 'movieId') %>% left_join(edx_us, by = 'userId') %>% left_join(edx_g, by = 'genres') %>% group_by(year) %>% summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n() + L))
  
  movie_edxr <- validation %>% left_join(edx_mor, by = 'movieId') %>% mutate(predicted = mu + b_i)
  movie_u_edxr <- validation %>% left_join(edx_mor, by = 'movieId') %>% left_join(edx_usr, by = 'userId') %>% mutate(predicted = mu + b_i + b_u)
  movie_u_g_edxr <- validation %>% left_join(edx_mor, by = 'movieId') %>% left_join(edx_usr, by = 'userId') %>% left_join(edx_gr, by = 'genres') %>% mutate(predicted = mu + b_i + b_u + b_g)
  movie_u_g_y_edxr <- validation %>% left_join(edx_mor, by = 'movieId') %>% left_join(edx_usr, by = 'userId') %>% left_join(edx_gr, by = 'genres') %>% left_join(edx_yr, by = 'year') %>% mutate(predicted = mu + b_i + b_u + b_g + b_y)
  
  RMSE(validation$rating, movie_edxr$predicted)
  RMSE(validation$rating, movie_u_edxr$predicted)
  RMSE(validation$rating, movie_u_g_edxr$predicted)
  RMSE(validation$rating, movie_u_g_y_edxr$predicted)
})

min(regRMSE)
finalLambda <- lambdas[which.min(regRMSE)]

edx_small <- edx[sample(nrow(edx), 500), ]
fitKnn <- train(rating ~ ., method = "knn", data = edx_small)
fitGL <- train(rating ~ ., method = "gamLoess", data = edx_small)
fitGM <- train(rating ~ ., method = "glm", data = edx_small)

fitKnnY <- train(rating ~ userId, method = "knn", data = edx_small)
fitGLY <- train(rating ~ userId, method = "gamLoess", data = edx_small)
fitGMY <- train(rating ~ userId, method = "glm", data = edx_small)

RMSE(validation$rating, mu)
RMSE(as.double(as.character(validation$rating)), as.numeric(as.character(predict(fitKnnY, validation))))
RMSE(as.double(as.character(validation$rating)), as.numeric(as.character(predict(fitGLY, validation))))
RMSE(as.double(as.character(validation$rating)), as.numeric(as.character(predict(fitGMY, validation))))