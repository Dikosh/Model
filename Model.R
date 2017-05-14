library(xgboost)
library(mlr)
library(caret)
dataset <- read.csv("~/resources/data/source.csv",sep=';')

train.flag <- createDataPartition(y=dataset$TARGET,p=0.75,list=FALSE)
train <- dataset[train.flag,]
test <- dataset[-train.flag,]

## Second method
require(dplyr)
require(lubridate)

require(Matrix)

datatarget <- train$TARGET
train.full.sparse <- sparse.model.matrix(TARGET~.-1,data=train)
test.full.sparse <- sparse.model.matrix(TARGET~.-1,data=test)
require(xgboost)

dtrain <- xgb.DMatrix(data=train.full.sparse, label=train$TARGET)

ctest <- xgb.DMatrix(data=test.full.sparse,label=test$TARGET)

#-------------Basic Training using XGBoost-----------------
# this is the basic usage of xgboost you can put matrix in data field
# note: we are putting in sparse matrix here, xgboost naturally handles sparse input
# use sparse matrix when your feature is sparse(e.g. when you are using one-hot encoding vector)
print("Training xgboost with sparseMatrix")
bst <- xgboost(data = train.full.sparse, label = train$TARGET, max_depth = 2, eta = 1, nrounds = 2,
               nthread = 2, objective = "binary:logistic")







########################################################################
param <- list(max_depth=8, eta=0.3, silent=1, objective='binary:logistic')
nround = 50

# training the model for two rounds
bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2)

# Model accuracy without new features
accuracy.before <- sum((predict(bst, test.full.sparse) >= 0.5) == test$TARGET) / length(test$TARGET)
accuracy.before

xgbmodel.predict <- predict(bst, test.full.sparse)
xgbmodel.predict.text <- levels(test$TARGET)[xgbmodel.predict + 1]
xgbmodel.predict.text
sss <- (predict(bst, test.full.sparse) >= 0.5)
sss1 <- as.numeric(sss)
base::table(sss1)
sum(diag(base::table(test$TARGET, sss)))/nrow(test.full.sparse)
confusionMatrix(sss1,test$TARGET) # confusion matrix
#mat <-base::table(mytest$cuisine, xgbmodel.predict) # table TP TN FP FN
#(precision <- diag(mat) / rowSums(mat)) # Precision
#recall <- (diag(mat) / colSums(mat)) # Recall
#f_meas <- 2*precision*recall/(precision+recall) # F-measure
#mean(f_meas) # F measure средний общий
#f_meas # F measure по отдельности





# by default, we predict using all the trees

pred_with_leaf = predict(bst, ctest, predleaf = TRUE)
head(pred_with_leaf)

create.new.tree.features <- function(model, original.features){
  pred_with_leaf <- predict(model, original.features, predleaf = TRUE)
  cols <- list()
  for(i in 1:length(trees)){
    # max is not the real max but it s not important for the purpose of adding features
    leaf.id <- sort(unique(pred_with_leaf[,i]))
    cols[[i]] <- factor(x = pred_with_leaf[,i], level = leaf.id)
  }
  cBind(original.features, sparse.model.matrix( ~ . -1, as.data.frame(cols)))
}

# Convert previous features to one hot encoding
new.features.train <- create.new.tree.features(bst, test.full.sparse)
new.features.test <- create.new.tree.features(bst, test.full.sparse)

# learning with new features
new.dtrain <- xgb.DMatrix(data = new.features.train, label = test$TARGET)
new.dtest <- xgb.DMatrix(data = new.features.test, label = test$TARGET)
watchlist <- list(train = new.dtrain)
bst <- xgb.train(params = param, data = new.dtrain, nrounds = nround, nthread = 2)

# Model accuracy with new features
accuracy.after <- sum((predict(bst, new.dtest) >= 0.5) == test$TARGET) / length(test$TARGET)

# Here the accuracy was already good and is now perfect.
cat(paste("The accuracy was", accuracy.before, "before adding leaf features and it is now", accuracy.after, "!\n"))

#############################################################################################################################################
#############################################################################################################################################
# alternatively, you can put in dense matrix, i.e. basic R-matrix
print("Training xgboost with Matrix")
bst <- xgboost(data = as.matrix(train.full.sparse), label = train$TARGET, max_depth = 2, eta = 1, nrounds = 2,
               nthread = 2, objective = "binary:logistic")

# you can also put in xgb.DMatrix object, which stores label, data and other meta datas needed for advanced features
print("Training xgboost with xgb.DMatrix")
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nrounds = 2, nthread = 2, 
               objective = "binary:logistic")

# Verbose = 0,1,2
print("Train xgboost with verbose 0, no message")
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nrounds = 2,
               nthread = 2, objective = "binary:logistic", verbose = 0)
print("Train xgboost with verbose 1, print evaluation metric")
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nrounds = 2,
               nthread = 2, objective = "binary:logistic", verbose = 1)
print("Train xgboost with verbose 2, also print information about tree")
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nrounds = 2,
               nthread = 2, objective = "binary:logistic", verbose = 2)

# you can also specify data as file path to a LibSVM format input
# since we do not have this file with us, the following line is just for illustration
# bst <- xgboost(data = 'agaricus.train.svm', max_depth = 2, eta = 1, nrounds = 2,objective = "binary:logistic")

#--------------------basic prediction using xgboost--------------
# you can do prediction using the following line
# you can put in Matrix, sparseMatrix, or xgb.DMatrix 
pred <- predict(bst, test.full.sparse)
err <- mean(as.numeric(pred > 0.5) != test$TARGET)
print(paste("test-error=", err))

#-------------------save and load models-------------------------
# save model to binary local file
xgb.save(bst, "xgboost.model")
# load binary model to R
bst2 <- xgb.load("xgboost.model")
pred2 <- predict(bst2, test.full.sparse)
# pred2 should be identical to pred
print(paste("sum(abs(pred2-pred))=", sum(abs(pred2-pred))))

# save model to R's raw vector
raw = xgb.save.raw(bst)
# load binary model to R
bst3 <- xgb.load(raw)
pred3 <- predict(bst3, test.full.sparse)
# pred3 should be identical to pred
print(paste("sum(abs(pred3-pred))=", sum(abs(pred3-pred))))

#----------------Advanced features --------------
# to use advanced features, we need to put data in xgb.DMatrix
#---------------Using watchlist----------------
# watchlist is a list of xgb.DMatrix, each of them is tagged with name
watchlist <- list(train=dtrain, test=ctest)
# to train with watchlist, use xgb.train, which contains more advanced features
# watchlist allows us to monitor the evaluation result on all data in the list 
print("Train xgboost using xgb.train with watchlist")
bst <- xgb.train(data=dtrain, max_depth=2, eta=1, nrounds=2, watchlist=watchlist,
                 nthread = 2, objective = "binary:logistic")
# we can change evaluation metrics, or use multiple evaluation metrics
print("train xgboost using xgb.train with watchlist, watch logloss and error")
bst <- xgb.train(data=dtrain, max_depth=2, eta=1, nrounds=2, watchlist=watchlist,
                 eval_metric = "error", eval_metric = "logloss",
                 nthread = 2, objective = "binary:logistic")

# xgb.DMatrix can also be saved using xgb.DMatrix.save
xgb.DMatrix.save(dtrain, "dtrain.buffer")
# to load it in, simply call xgb.DMatrix
dtrain2 <- xgb.DMatrix("dtrain.buffer")
bst <- xgb.train(data=dtrain2, max_depth=2, eta=1, nrounds=2, watchlist=watchlist,
                 nthread = 2, objective = "binary:logistic")
# information can be extracted from xgb.DMatrix using getinfo
label = test$TARGET
pred <- predict(bst, ctest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

# You can dump the tree you learned using xgb.dump into a text file
xgb.dump(bst, "dump.raw.txt", with_stats = T)

# Finally, you can check which features are the most important.
print("Most important features (look at column Gain):")
imp_matrix <- xgb.importance(feature_names = colnames(test.full.sparse), model = bst)
print(imp_matrix)

# Feature importance bar plot by gain
print("Feature importance Plot : ")
print(xgb.plot.importance(importance_matrix = imp_matrix))
base::table(test$TARGET)
