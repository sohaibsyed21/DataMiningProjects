# Example using R Keras deep learning model.
# Vijay K. Gurbani, Ph.D.,
# Illinois Institute of Technology
# CS 422 (Introduction to Data Mining)

library(keras)
library(dplyr)
library(caret)

rm(list=ls())

data("iris")

iris <- iris[sample(nrow(iris)),] # Shuffle!

# Create a new column called "label" and store the ordinal value of the
# Species there.  
# 0 - implies Setosa     <--- 0-based!  Keep in mind.
# 1 - implies Versicolor
# 2 - implies Virginica
iris$label <- rep(0, nrow(iris))
iris$label[iris$Species == "versicolor"] <- 1
iris$label[iris$Species == "virginica"] <- 2

# Remove Species from from iris dataframe; it has been coded as an integer and
# saved in iris$label
iris$Species <- NULL

set.seed(1122)

indx <- sample(1:nrow(iris), 0.20*nrow(iris))
test.df  <- iris[indx, ]
train.df <- iris[-indx, ]

X_train <- select(train.df, -label)
y_train <- train.df$label

# Now, take the training labels, which are (0, 1, 2) corresponding to
# setosa, versicolor, and virginica, respectively, using one hot encoding.
# Thus, setosa     = 1 0 0
#       versicolor = 0 1 0
#       virginica  = 0 0 1
y_train.ohe <- to_categorical(y_train)

X_test <- select(test.df, -label)
y_test <- test.df$label
y_test.ohe <- to_categorical(test.df$label)

# When you start dealing with keras, you need to understand the R %>%
# operator.  This infix operator is not part of base R, but is defined in a
# package called magrittr, which is used in package dplyr.  This construct
# passes the LHS of the operator as the first argument to the RHS of the 
# operator.  It works like the Unix pipe.  
# Example:
#   > iris %>% head()
# is equivalent to head(iris).
# Or,
#   > data <- c(0.9817, 0.8765, 1.2876, 4.8765)
#   > data %>% round(3)
# What do you think the above will do?

model <- keras_model_sequential() %>%
  layer_dense(units = 8, activation="relu", input_shape=c(4)) %>%
  layer_dense(units = 3, activation="softmax")

model # Print the summary of the model.

model %>% 
  compile(loss = "categorical_crossentropy", 
          optimizer="adam", 
          metrics=c("accuracy"))

model %>% fit(
  data.matrix(X_train), 
  y_train.ohe,
  epochs=100,
  batch_size=5,
  validation_split=0.20
)

model %>% evaluate(as.matrix(X_test), y_test.ohe)

# Yuck!  They have deprecated predict_classes() in TensorFlow 2.6.  So
# commenting the two lines below.
# pred.class  <- model %>% predict_classes(as.matrix(X_test))
# pred.prob   <- model %>% predict(as.matrix(X_test)) %>% round(3)

pred.prob <- predict(model, as.matrix(X_test))
# The above looks like the following:
#          [,1]    [,2]    [,3]
#  [1,] 0.03117 0.36818 0.60065
#  [2,] 0.05078 0.39949 0.54972
# ...
# Grab the index of the highest probability value ...
pred.class <- apply(pred.prob, 1, function(x) which.max(x)-1) # Index for labels
                                                              # is 0-based
# ... and this becomes our predicted class.

confusionMatrix(as.factor(pred.class), as.factor(y_test))
