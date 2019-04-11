# Natural Language Processing

# Importing data
reviewsData <- read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

liked <- reviewsData[reviewsData$Liked == 1,]
notLiked <- reviewsData[reviewsData$Liked == 0,]
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(reviewsData$Review))
# Change upper case words to lowercase
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
# install.packages('SnowballC')
# Remove stopwords
corpus = tm_map(corpus, removeWords, stopwords())
# Remove whitespaces
corpus = tm_map(corpus, stripWhitespace)
# Stem words
corpus = tm_map(corpus, stemDocument)
# Next, Sparse Matrix

# bag of words model
dtm = DocumentTermMatrix(corpus)
# Dimensionality reduction (Sparsity reduciton)
dtm = removeSparseTerms(dtm, 0.999)
# Transform this into datafrae so i can feed into classification model
datasetWords = as.data.frame(as.matrix(dtm))
# add column liked to data frame
datasetWords$Liked = reviewsData$Liked
dtm

library(caTools)
logRegWords <- datasetWords[rowSums(is.na(datasetWords)) == 0, ]
targetColNum = as.numeric(which( colnames(logRegWords)=='Liked' ))
set.seed(42)
split = sample.split(logRegWords$Liked, 0.75)
training_set = subset(logRegWords, split= TRUE)
test_set = subset(logRegWords, split = FALSE)

classifier = glm(formula = Liked ~ ., family = binomial, data=training_set)
prob_predict = predict(classifier, type = 'response', logRegWords1 = test_set[-targetColNum])
Class_predict = ifelse(prob_predict > 0.5, 1, 0)
logRegConMatrix = table(test_set[, targetColNum], Class_predict)
logRegConMatrix
results_matrix = data.matrix(logRegConMatrix)
LR_true_zero = as.numeric(results_matrix[1, 1])
LR_false_zero = as.numeric(results_matrix[1, 2])
LR_true_one = as.numeric(results_matrix[2, 2])
LR_false_one = as.numeric(results_matrix[2, 1])
LR_accuracy = (LR_true_one + LR_true_zero)/(LR_true_one + LR_true_zero + LR_false_one + LR_false_zero)
print(paste("Logistic Regression Accuracy: ", toString(LR_accuracy*100) ))

library(e1071)


# SVM classifier
SVM_Classifier = svm(formula = Liked ~ .,data = training_set, type = 'C-classification', kernel = 'linear')
# SVM Prob predict
SVM_prob_predict = predict(SVM_Classifier, type = 'response', logRegData1 = test_set[-targetColNum])
svmConMatrix = table(test_set[,targetColNum], SVM_prob_predict)
SVM_results_matrix = data.matrix(svmConMatrix)
SVM_true_zero = as.numeric(results_matrix[1, 1])
SVM_false_zero = as.numeric(results_matrix[1, 2])
SVM_true_one = as.numeric(results_matrix[2, 2])
SVM_false_one = as.numeric(results_matrix[2, 1])
SVM_accuracy = (SVM_true_one + SVM_true_zero)/(SVM_true_one + SVM_true_zero + SVM_false_one + SVM_false_zero)
# print(paste("SVM Accuracy: ", toString(SVM_accuracy*100) ))

library(psych)
library(class)


k_values <- list()
acc_values <- list()
largest_k <- 15
best_k <- largest_k
best_accuracy <- 0
current_accuracy <- 0

# for(i in 1:largest_k){
#   # print(paste("k = ", i))
#   y_predict = knn(training_set[,-targetColNum], test_set[,-targetColNum], cl = training_set[,targetColNum], k=i)
#   con_matrix = table(test_set[,targetColNum], y_predict)
#   con_matrix
#   true_zero = as.numeric(con_matrix[1, 1])
#   false_zero = as.numeric(con_matrix[1, 2])
#   true_one = as.numeric(con_matrix[2, 2])
#   false_one = as.numeric(con_matrix[2, 1])
#   accuracy = (true_one + true_zero)/(true_one + true_zero + false_one + false_zero)
#   accuracyPercentage = accuracy*100
#   current_accuracy <- accuracyPercentage
#   if (current_accuracy > best_accuracy){
#     best_accuracy <- current_accuracy
#     best_k <-1
#   }
#   errorPercentage = 100 -accuracy*100
#   # print(paste("Accuracy = ", round(accuracyPercentage, digits = 1)))
#   # print(paste("Error = ", round(errorPercentage, digits = 1)))
#   k_values <- c(k_values, i)
#   acc_values <- c(acc_values, accuracy)
#   # print(paste("Best k = ", best_k))
# }

# I commented out the for loop, it takes a while to run. best k is 1

best_k <- 1
y_predict = knn(training_set[,-targetColNum], test_set[,-targetColNum], cl = training_set[,targetColNum], k=best_k)
knn_con_matrix = table(test_set[,targetColNum], y_predict)
knn_con_matrix
knn_true_zero = as.numeric(knn_con_matrix[1, 1])
knn_false_zero = as.numeric(knn_con_matrix[1, 2])
knn_true_one = as.numeric(knn_con_matrix[2, 2])
knn_false_one = as.numeric(knn_con_matrix[2, 1])
knn_accuracy = (knn_true_one + knn_true_zero)/(knn_true_one + knn_true_zero + knn_false_one + knn_false_zero)
# print(paste("By iterating k values of 1 through", largest_k, "I found that optimal k is", best_k))

print(paste("knn Accuracy: ", toString(knn_accuracy*100) ))
print(paste("SVM Accuracy: ", toString(SVM_accuracy*100) ))
print(paste("Logistic Regression Accuracy: ", toString(LR_accuracy*100) ))
