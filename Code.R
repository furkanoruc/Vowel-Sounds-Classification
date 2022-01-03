#install.packages("funLBM")
library(MixGHD)
library(CrossClustering)
library(stats)
library(cluster)
library(MASS)
library(tidyverse)
library(factoextra)
#install.packages("mclust")
library(mclust)

set.seed(123)

rm(list = ls())

#uploading the dataset
train.df <- (read.table("vowel-train.txt", header = T, sep = ","))
train.df <- subset(train.df, select = -c(row.names))
test.df <- (read.table("vowel-test.txt", header = T, sep = ","))
test.df <- subset(test.df, select = -c(row.names))

#1
#pca analysis

#generating a subset of predictors to conduct pca
train.predictors <- subset(train.df, select = -c(y))

#creating the pca object for train, scaled and centered.
'Centering is conducted since it does not harm the generalization.
Also, covariance does not change. While, centering offers
conveniency. pg:435, further reference'

train.pca <- prcomp(train.predictors, center = T, scale = T)
print(train.pca)
'
train.pred.std <- data.frame(sapply(train.predictors, 
                      function(x){(x-mean(x))/sd(x)}))
'
cor.train <- cor(train.predictors)
#cor.train.std <- cor(train.pred.std)

eigen.train <- eigen(cor.train)
#eigen.train.std <- eigen(cor.train.std)

train.pca$sdev^2
eigen.train$values
#eigen.train.std$values

'Eigenvectors can be presented as:'
eigen.train$vectors

eigen.train$values

'Below, principal components are presented with respect to
their representation of percentage variance. Based on results, principal
components up to 7th component explain 92.794% of the variance in the
dataset.'
summary(train.pca)

#2
#selecting top 7 principle component results

train.pca.df.7 <- data.frame(train.pca$x)
train.pca.df.7 <- subset(train.pca.df.7, select = c(1,2,3,4,5,6,7))

train.pca.df.7$y <- train.df$y

#Fit the model
model <- lda(y~., data = train.pca.df.7)
#Make predictions
predictions <- model %>% predict(train.pca.df.7)
#Model accuracy
mean(predictions$class==train.pca.df.7$y)

#misclassification error rate for training is found as:

misc_rate_train_lda <- 
  mean(predictions$class!=train.pca.df.7$y)
misc_rate_train_lda

accuracy <- sum(ifelse(predictions$class==train.pca.df.7$y
                , 1,0))/nrow(train.pca.df.7)

table(train.pca.df.7$y, predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

'At this point, PCA is applied to test set. Then, LDA will
be conducted on the test set.'

test.predictors <- subset(test.df, select = -c(y))
test.pca <- prcomp(test.predictors, center = T, scale = T)

test.pca.df.7 <- data.frame(test.pca$x)
test.pca.df.7 <- subset(test.pca.df.7, select = c(1,2,3,4,5,6,7))

test.pca.df.7$y <- test.df$y

#model PREDICT

predictions_test <- model %>% predict(test.pca.df.7)
#Model accuracy
mean(predictions_test$class==test.pca.df.7$y)

accuracy <- sum(ifelse(predictions_test$class==test.pca.df.7$y
                       , 1,0))/nrow(test.pca.df.7)


#misclassification error rate for test is found as:
misclassification_rate_lda_test <- 
  sum(ifelse(predictions_test$class!=test.pca.df.7$y
             , 1,0))/nrow(test.pca.df.7)

misclassification_rate_lda_test

table(test.pca.df.7$y, predictions_test$class, 
      dnn = c('Actual Group','Predicted Group'))

#3

'Quadratic Discriminant Analysis'

#training and training prediction

model.train.pca.qda <- qda(y ~., data = train.pca.df.7)

qda.pred.train.pca <- predict(model.train.pca.qda)

qda.pred.train.pca <- model.train.pca.qda %>% 
  predict(train.pca.df.7)

table(train.pca.df.7$y, qda.pred.train.pca$class, 
      dnn = c('Actual Group','Predicted Group'))

misclassification_rate_qda_train <- 
  mean(qda.pred.train.pca$class!=train.pca.df.7$y)

misclassification_rate_qda_train

'For training set, QDA gives a lower misclassification
error for sure! Now, test set is tested.'

#Test Set for QDA


qda.test.predictions <- 
  model.train.pca.qda %>% predict(test.pca.df.7)
#Model accuracy
misc.error.qda.test <- 
  mean(qda.test.predictions$class!=test.pca.df.7$y)

misc.error.qda.test

accuracy.qda.test <- 
  mean(qda.test.predictions$class==test.pca.df.7$y)

accuracy.qda.test

table(test.pca.df.7$y, qda.test.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

'As like the lda case, misclassification error is significantly
high in the test set. As provided:'


#4

'In this part, analysis for original dataset is provided.'

#Fit the model
original.data.model <- lda(y~., data = train.df)
#Make predictions
org.train.predictions <- 
  original.data.model %>% predict(train.df)
#Model accuracy
org.misc.train <- mean(org.train.predictions$class!=train.df$y)
org.misc.train

org.accuracy.train <- 
  mean(org.train.predictions$class==train.df$y)

org.accuracy.train

table(train.df$y, org.train.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

'Original Data, Testing with the trained model in test set'

#Make predictions
org.test.predictions <- 
  original.data.model %>% predict(test.df)
#Model accuracy
org.misc.test <- mean(org.test.predictions$class!=test.df$y)
org.misc.test

org.accuracy.test <- 
  mean(org.test.predictions$class==test.df$y)

org.accuracy.test

table(test.df$y, org.test.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

'QDA, Org Data'

#Fit the model
qda.original.data.model <- qda(y~., data = train.df)
#Make predictions
qda.org.train.predictions <- 
  qda.original.data.model %>% predict(train.df)
#Model accuracy
qda.org.misc.train <- 
  mean(qda.org.train.predictions$class!=train.df$y)
qda.org.misc.train

qda.org.accuracy.train <- 
  mean(qda.org.train.predictions$class==train.df$y)

qda.org.accuracy.train

table(train.df$y, qda.org.train.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

'Original Data, Testing with the trained model in test set'

#Make predictions
qda.org.test.predictions <- 
  qda.original.data.model %>% predict(test.df)
#Model accuracy
qda.org.misc.test <- mean(qda.org.test.predictions$class!=test.df$y)
qda.org.misc.test

qda.org.accuracy.test <- 
  mean(qda.org.test.predictions$class==test.df$y)

qda.org.accuracy.test

table(test.df$y, qda.org.test.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

'Resulting Matrix'

results_matrix <- matrix(c(misc_rate_train_lda, 
                         misclassification_rate_lda_test,
       misclassification_rate_qda_train, misc.error.qda.test,
       org.misc.train, org.misc.test,
       qda.org.misc.train, qda.org.misc.test), 
       nrow = 4, ncol = 2, byrow = T)

colnames(results_matrix) <- c("Training", "Testing")
rownames(results_matrix) <- c("PCA: LDA", "PCA: QDA", 
                              "Original: LDA", "Original: QDA")

results_matrix

'Results show that the most successful training results are in
QDA model, applied to original data. The least successful one is
LDA model, applied with PCA version of the data. When test
sets are observed, QDA model applied to original data provides
the most successful results. While, LDA model applied to PCA version
of the data result with least successful results. It can be interpreted 
that the dataset is more suitable when the original version is used
to train a model. Also, QDA model provides superior results
than LDA.'

#5
'Finding the classes which 
are the most difficult to distinguish.'

'This analysis will be conducted on both original data & PCA version (>90%).'

table.pca.lda.train <- table(train.pca.df.7$y, predictions$class, 
      dnn = c('Actual Group','Predicted Group'))

table.pca.lda.train
'2,5,6,7,9'

table.pca.lda.test <- table(test.pca.df.7$y, predictions_test$class, 
      dnn = c('Actual Group','Predicted Group'))
table.pca.lda.test
'1,2,3,4,5,6,7,8,9,10'

table.pca.qda.train <- table(train.pca.df.7$y, qda.pred.train.pca$class, 
      dnn = c('Actual Group','Predicted Group'))
table.pca.qda.train
'3,6,9'
table.pca.qda.test <- table(test.pca.df.7$y, qda.test.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))
table.pca.qda.test
'2,4,5,6,7,8,9'
table.org.lda.train <- table(train.df$y, org.train.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))
table.org.lda.train
'2,6,9'
table.org.lda.test <- table(test.df$y, org.test.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))
table.org.lda.test
'5,7,9,10'
table.org.qda.train <- table(train.df$y, qda.org.train.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))
table.org.qda.train
'NA'
table.org.qda.test <- table(test.df$y, qda.org.test.predictions$class, 
      dnn = c('Actual Group','Predicted Group'))
table.org.qda.test
'3,4,8,10'

'Test performance in PCA version is problematic due to 
high number of misclassifications, which makes it challenging
to distinguish any specific classes. While, original data
based model results in test sets provide more obvious outcomes
on potential classes to remove. Based on results, 
2,6,9 classes can be removed to experiment further.'

#Experiment with removing 
'It will be important to manage overfit problem to rearrange
class distributions. If we remove unsuccessful classes from
training set, that can be biased since this way, we might be
furthering the overfit problem. The overfit problem is
clearly present in the model, since misc. rate of in training  is 0.011 
for QDA, while it is 0.52 for test. This outcome shows
that model is clearly overfit to the training.

As a result, to avoid biases, class rearranging is conducted
based on test set results of the model.'


#Original, LDA
'Classes to be removed: 2,6,9'

train.rearg.df <- filter(train.df, y != 5 & y != 7 & y!= 10)
test.rearg.df <- filter(test.df, y != 5 & y != 7 & y!= 10)

#Fit the model
rearg.model.lda <- lda(y~., data = train.rearg.df)
#Make predictions
rearg.train.pred.lda <- 
  rearg.model.lda %>% predict(train.rearg.df)
#Model accuracy
rearg.misc.train.lda <- 
  mean(rearg.train.pred.lda$class!=train.rearg.df$y)

rearg.misc.train.lda

#test

#Make predictions
rearg.test.pred.lda <- 
  rearg.model.lda %>% predict(test.rearg.df)
#Model accuracy
rearg.misc.test.lda <- 
  mean(rearg.test.pred.lda$class!=test.rearg.df$y)
rearg.misc.test.lda

table(test.rearg.df$y, rearg.test.pred.lda$class, 
      dnn = c('Actual Group','Predicted Group'))

#Original, QDA

#Fit the model
rearg.model.qda <- qda(y~., data = train.rearg.df)
#Make predictions
rearg.train.pred.qda <- 
  rearg.model.qda %>% predict(train.rearg.df)
#Model accuracy
rearg.misc.train.qda <- 
  mean(rearg.train.pred.qda$class!=train.rearg.df$y)

rearg.misc.train.qda

#test

#Make predictions
rearg.test.pred.qda <- 
  rearg.model.qda %>% predict(test.rearg.df)
#Model accuracy
rearg.misc.test.qda <- 
  mean(rearg.test.pred.qda$class!=test.rearg.df$y)
rearg.misc.test.qda

table(test.rearg.df$y, rearg.test.pred.qda$class, 
      dnn = c('Actual Group','Predicted Group'))

#rearg.misc.train.lda.5.7.10 <- rearg.misc.train.lda
#rearg.misc.test.lda.5.7.10 <- rearg.misc.test.lda
#rearg.misc.train.qda.5.7.10 <- rearg.misc.train.qda
#rearg.misc.test.qda.5.7.10 <- rearg.misc.test.qda

'PCA Version of Classes Removed Case'

#train.rearg.df
#test.rearg.df

train.rearg.predictors <- subset(train.rearg.df, select = -c(y))
view(train.rearg.predictors)

train.rearg.pca <- prcomp(train.rearg.predictors, 
                          center = T, scale = T)
print(train.rearg.pca)

cor.rearg.train <- cor(train.rearg.predictors)
#cor.train.std <- cor(train.pred.std)

eigen.rearg.train <- eigen(cor.rearg.train)
eigen.rearg.train$values

summary(train.rearg.pca)

'Based on results, top 7 principle components explain 0.93 of the
overall variance, so top 7 component will be selected to conduct
LDA and QDA in the classes removed case.'

train.rearg.pca.df.7 <- data.frame(train.rearg.pca$x)
train.rearg.pca.df.7 <- subset(train.rearg.pca.df.7, select = c(1,2,3,4,5,6,7))

train.rearg.pca.df.7$y <- train.rearg.df$y

#Fit the model
model.rearg.pca.7 <- lda(y~., data = train.rearg.pca.df.7)
#Make predictions
predictions.rearg.pca.7 <- model.rearg.pca.7 %>% predict(train.rearg.pca.df.7)

#misclassification error rate for training is found as:

misc_rate_rearg_pca_train_lda <- 
  mean(predictions.rearg.pca.7$class!=train.rearg.pca.df.7$y)
misc_rate_rearg_pca_train_lda

'Test for LDA & PCA top 7 Components & Classes: Removed: 3,4,8'

test.rearg.predictors <- subset(test.rearg.df, select = -c(y))
test.rearg.predictors.pca <- prcomp(test.rearg.predictors, center = T, scale = T)

test.rearg.pca.df.7 <- data.frame(test.rearg.predictors.pca$x)
test.rearg.pca.df.7 <- subset(test.rearg.pca.df.7, select = c(1,2,3,4,5,6,7))

test.rearg.pca.df.7$y <- test.rearg.df$y

#model PREDICT
test_rearg_pca_predictions_test <- model.rearg.pca.7 %>% 
  predict(test.rearg.pca.df.7)
#Model missclassification error rate
misc_rate_rearg_pca_test_lda <- mean(test_rearg_pca_predictions_test$class!=
           test.rearg.pca.df.7$y)

misc_rate_rearg_pca_test_lda




'QDA - PCA Version: Classes Removed Case'


train.rearg.predictors.qda <- subset(train.rearg.df, select = -c(y))
view(train.rearg.predictors.qda)

train.rearg.pca <- prcomp(train.rearg.predictors.qda, 
                          center = T, scale = T)
print(train.rearg.pca)

cor.rearg.train <- cor(train.rearg.predictors)
#cor.train.std <- cor(train.pred.std)

eigen.rearg.train <- eigen(cor.rearg.train)
eigen.rearg.train$values

summary(train.rearg.pca)

'Based on results, top 7 principle components explain 0.93 of the
overall variance, so top 7 component will be selected to conduct
LDA and QDA in the classes removed case.'

train.rearg.pca.df.7 <- data.frame(train.rearg.pca$x)
train.rearg.pca.df.7 <- subset(train.rearg.pca.df.7, select = c(1,2,3,4,5,6,7))

train.rearg.pca.df.7$y <- train.rearg.df$y

#Fit the model
model.rearg.pca.7 <- qda(y~., data = train.rearg.pca.df.7)
#Make predictions
predictions.rearg.pca.7 <- model.rearg.pca.7 %>% predict(train.rearg.pca.df.7)

#misclassification error rate for training is found as:

misc_rate_rearg_pca_train_qda <- 
  mean(predictions.rearg.pca.7$class!=train.rearg.pca.df.7$y)
misc_rate_rearg_pca_train_qda

'Test for QDA & PCA top 7 Components & Classes: Removed: 5,7,10'

test.rearg.predictors <- subset(test.rearg.df, select = -c(y))
test.rearg.predictors.pca <- prcomp(test.rearg.predictors, center = T, scale = T)

test.rearg.pca.df.7 <- data.frame(test.rearg.predictors.pca$x)
test.rearg.pca.df.7 <- subset(test.rearg.pca.df.7, select = c(1,2,3,4,5,6,7))

test.rearg.pca.df.7$y <- test.rearg.df$y

#model PREDICT
test_rearg_pca_predictions_test <- model.rearg.pca.7 %>% 
  predict(test.rearg.pca.df.7)
#Model missclassification error rate
misc_rate_rearg_pca_test_qda <- mean(test_rearg_pca_predictions_test$class!=
                                       test.rearg.pca.df.7$y)

misc_rate_rearg_pca_test_qda


'Results'

rearg_results_matrix_all <- matrix(c(misc_rate_train_lda, 
                           misclassification_rate_lda_test,
                           misclassification_rate_qda_train, misc.error.qda.test,
                           org.misc.train, org.misc.test,
                           qda.org.misc.train, qda.org.misc.test,
                           rearg.misc.train.lda.5.7.10,
                           rearg.misc.test.lda.5.7.10,
                           rearg.misc.train.qda.5.7.10,
                           rearg.misc.test.qda.5.7.10,
                           rearg.misc.train.lda,
                           rearg.misc.test.lda,
                           rearg.misc.train.qda,
                           rearg.misc.test.qda,
                           misc_rate_rearg_pca_train_lda,
                           misc_rate_rearg_pca_test_lda,
                           misc_rate_rearg_pca_train_qda,
                           misc_rate_rearg_pca_test_qda), 
                         nrow = 10, ncol = 2, byrow = T)

colnames(rearg_results_matrix_all) <- c("Training", "Testing")
rownames(rearg_results_matrix_all) <- c("PCA: LDA", "PCA: QDA", 
                              "Original: LDA", "Original: QDA",
                              "Classes:5,7,10 Removed Org: LDA",
                              "Classes:5,7,10 Removed Org: QDA",
                              "Classes:3,4,8 Removed Org: LDA",
                              "Classes:3,4,8 Removed Org: QDA",
                              "Classes:5,7,10 Removed PCA: LDA",
                              "Classes:3,4,8 Removed PCA: QDA")

rearg_results_matrix_all
final.table <- rearg_results_matrix_all

'In the below table, all results with different combinations is
presented. To remove variables, tables from QDA and LDA test results
are seperately examined. LDA results suggested the removal of classes: 5,7,10.
While, QDA results suggested the removal of classes: 2,6,9. Based on
significant misclassifications of these classes in seperate models, two
alternatives are selected. Results show that misclassification rate
drops as low as 0.396 in LDA model, when classes 5,7,10 are removed.'


#results.3.5.7.rearg <- rearg_results_matrix
#results.3.8.10.rearg <- rearg_results_matrix
#results.2.6.9.rearg <- rearg_results_matrix
results.3.5.7.rearg
results.3.8.10.rearg
results.2.6.9.rearg

#6

#k-means
train.clustering <- filter(train.df, y == 1 | 
                             y == 3 | y == 6 |
                             y == 10)

train.cl.df <- subset(train.clustering, select = -c(y))
train.y.df <- subset(train.clustering, select = c(y))

test.clustering <- filter(test.df, y == 1 | 
                            y == 3 | y == 6 |
                            y == 10)

test.cl.df <- subset(test.clustering, select = -c(y))
test.y.df <- subset(test.clustering, select = c(y))

#since training and testing is not relevant in clustering,
#two datasets are merged to have a larger set of data.

df.cluster <- rbind(train.cl.df, test.cl.df)
df.y.cluster <- rbind(train.y.df, test.y.df)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(df.cluster, k, nstart = 10)$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main = "Total Data, Within Cluster Sum of Squares, Elbow
     Method")

#based on the elbow analysis, 4 clusters are decided to be optimum.

model.k.means <- kmeans(df.cluster, 
                        4, iter.max = 20,
                        nstart = 1)

print(model.k.means)
k.means.vector <- model.k.means$cluster

#Hierarchical Clustering
dist.cluster <- dist(df.cluster, "euclidean")
model.hiearch <- hclust(dist.cluster, "average")

print(model.hiearch)

model.hiearch$call
model.hiearch$merge
model.hiearch$height
model.hiearch$merge

model.4.hierarch <- cutree(model.hiearch, k = 4)

labels.hierach.4 <- model.4.hierarch

plot(as.dendrogram(model.hiearch), 
    horiz=TRUE, 
    main="clusters")

'Below, hiearchccal clustering for variables is presented.'
cormtx <- cor(df.cluster)
dist.cor <- sqrt(2*(1-cormtx))
hc.c <- hclust(as.dist(dist.cor), 
               "complete") 
plot(hc.c, hang=-1, main="Clusters for variables")


#Ward

hc.eu.w <- hclust(dist.cluster, "ward") 
hc.eu.w$merge
hc.eu.w$height
plot(hc.eu.w,hang = -1)


#Model-Clustering

'To conduct model clustering, we need to make sure the dataset
behaves normal. To test and assure normality, a small qqnorm survey
is conducted.'

qq.x1 <- qqnorm(df.cluster$x.1, main = "Normal Q-Q Plot for x1")
qq.x2 <- qqnorm(df.cluster$x.2, main = "Normal Q-Q Plot for x2")
qq.x3 <- qqnorm(df.cluster$x.3, main = "Normal Q-Q Plot for x3")
qq.x4 <- qqnorm(df.cluster$x.4, main = "Normal Q-Q Plot for x4")
qq.x5 <- qqnorm(df.cluster$x.5, main = "Normal Q-Q Plot for x5")
qq.x6 <- qqnorm(df.cluster$x.6, main = "Normal Q-Q Plot for x6")
qq.x7 <- qqnorm(df.cluster$x.7, main = "Normal Q-Q Plot for x7")
qq.x8 <- qqnorm(df.cluster$x.8, main = "Normal Q-Q Plot for x8")
qq.x9 <- qqnorm(df.cluster$x.9, main = "Normal Q-Q Plot for x9")
qq.x10 <- qqnorm(df.cluster$x.10, main = "Normal Q-Q Plot for x10")

'Although tails can be determined to be slightly problematic,
qqnorm plots show a normal behaviour from almost all variables.'

model.mdl.cluster <- Mclust(df.cluster, G = 4)
model.mdl.cluster.BIC <- mclustBIC(df.cluster)
print(model.mdl.cluster)

#plot(model.mdl.cluster, what = "density")
#plot(model.mdl.cluster, what = "uncertainty")

sort(model.mdl.cluster$uncertainty, 
     decreasing = TRUE) %>% head()

summary(model.mdl.cluster)
summary(model.mdl.cluster.BIC)

model.mdl.cluster$parameters[2]
model.mdl.cluster$parameters[3]
model.mdl.cluster$parameters[4]

model.mdl.labels <- model.mdl.cluster$classification
model.mdl.labels

#Comparing Clustering models

kmeans.ARI <- adjustedRandIndex(df.y.cluster$y, k.means.vector)
hierarc.ARI <- adjustedRandIndex(df.y.cluster$y, 
                                 labels.hierach.4)
mdl.ARI <- adjustedRandIndex(df.y.cluster$y, model.mdl.labels)

kmeans.ARI
hierarc.ARI
mdl.ARI

'When Model outcomes are measured with adjusted rand index, it is observed that kmeans has the highest result with 0.40, followed by hiearchical as 0.13 and model based clustering as 0.22. It can be expressed that kmeans is the most effective model as we compare real class labels and assigned class labels.'

'Model performances can be interpreted in several perspectives. First,
in terms of computation and time efficiency, k-means and hierarchical
clustering have faster processing, while model based clustering
is the most costly methodology. Also, model based clustering
is computationally the most costly methodology. Other than that,
models can be interpreted on the basis of assumptions. K-means and
hierarchical clustering are unsupervised methods and do not assume
any information regarding the dataset. While, model based clustering
has distinct assumption of multivariate normality of the data. This is
an important limitation for model based clustering and due to this reason,
analysis is conducted in a careful manner. In terms of explainability of models
to non-technical individuals, hierarchical and k-means are more easily
explainable while due to limitations, model based clustering is more challenging
to interpret.'

tinytex::install_tinytex()





