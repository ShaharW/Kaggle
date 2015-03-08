install.packages("ggplot2")
install.packages("plyr")
install.packages("rpart")
install.packages('rpart.plot')
install.packages("cluster")
install.packages("rattle")
install.packages("ROCR")
install.packages('party')
install.packages('randomForest')

library(ggplot2)
library(plyr)
library(rpart)
library(rpart.plot)
library(cluster)
library(rattle)
library(ROCR)
library(party)
library(randomForest)

source_path <- "C:/Users/user/dataSci/Ass2/Assignment2/titanic.train.csv"
data <- read.csv(source_path, header=TRUE)

# split the data to train and test
set.seed(12345)
trainID <- sample(data[,"PassengerId"], nrow(data)*0.6)
testID <- setdiff(data[,"PassengerId"], trainID)
train <- data[trainID,]
test <- data[testID,]

# ------ Part a ------
atrain <- train #training set for part a of the question
atest <- test #testing set for part a of the question

# --- a1 ---
prop.table(table(atrain$Sex, atrain$Survived),1)

# --- a2 ---
atrain$FareDis <- as.factor(cut_number(atrain$Fare,10)) # turn the int feature in to 10 factors feature
arrange(aggregate(Survived ~  Sex+Pclass +FareDis, data=atrain, FUN=function(x) {sum(x)/length(x)}),Sex,Pclass,FareDis) # a table that shows the effect of the three features on the survival chances 

# --- a3 ---
atest$NaivePred <- 0 # set the prediction column on the test set
atest$NaivePred[atest$Sex == "female" & (atest$Pclass != 3 | atest$Fare < 27.8)] <- 1 # decide who the survivals are
errore_rate <- prop.table(table(atest$NaivePred, atest$Survived))[1,2] + prop.table(table(atest$NaivePred, atest$Survived))[2,1]
errore_rate

# ------ Preprocessing and feature engineering ------
full <- rbind(train, test) # in order to perform preprocessing on both train and test

# --- delete unvaluable column ---
full$Cabin <- NULL # have a lot of empty instances and doesn't seem to effect the label
full$Ticket <- NULL # no meaning
full$PassengerId <- NULL# no meaning

# --- Embarked feature ---
prop.table(table(full$Embarked, full$Survived),1) # the embarked feature might be worthful
full$Embarked[which(full$Embarked=="")] <- "S" # replacing missing values with most common value

# --- Family and friends ---
companions <- full$SibSp + full$Parch 
hist(companions) # we can learn about the companions distribution and discretize accordingly
full$trav_as <- "alone" # initialize the new column
full$trav_as[companions == 1] <- "couple"
full$trav_as[companions == 2] <- "small_family"
full$trav_as[companions > 2] <- "large_family"
# traveling with family effects your actions in case of disaster, so is traveling as a crew member:
full$trav_as[full$Fare==0] <- "crew"
full$trav_as <- as.factor(full$trav_as)

# being a crew member also effects your cabine location and maybe your economic status
full$Pclass <- as.character(full$Pclass)
full$Pclass[full$Fare==0] <- "crew"
full$Pclass <- as.factor(full$Pclass)

# --- use the title of the person ---
full$Name <- as.character(full$Name) # in order to perform character functions
full$title <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]}) # split the name and take the title
full$title <- sub(' ', '', full$title) # delete the space
full$title <- factor(full$title)
full$Name <- NULL # delete the Name column

# fill the missing values in Age feature based on the average of travelers
# with the same title and traveling status, for instance- unmarried weman who traveled alone might
# not be the same age as unmarried weman who traveld with large families (child)

for(i in which(is.na(full$Age), arr.ind = TRUE, useNames = TRUE)){ # loop over rows of missing values
  full[i,"Age"] <- mean(full[full$title==full[i,"title"] & full$trav_as == full[i,"trav_as"],"Age"],na.rm=TRUE)
}
full$Age <- as.integer(full$Age)

# --- split back to train and test ---
train <- full[1:nrow(train),] 
test <- full[(nrow(train)+1):(nrow(full)),] 

# ------ Part b ------
# --- b2 ---
fit <- rpart(Survived ~ .,data=train, method="class",control=rpart.control(minbucket=1, minsplit=2, cp=0))

# --- b3 ---
rsq.rpart(fit)

# --- b4 ---
fancyRpartPlot(fit)

# --- b5 ---
preds <- predict(fit, newdata = test, type="class")
table(preds,test$Survived) # confusion matrix
(table(preds,test$Survived)[1,2]+table(preds,test$Survived)[2,1])/sum(table(preds,test$Survived)) # error

# --- b6 ---
plotcp(fit)

# --- b7 ---
fit$cptable

# --- b9 ---
# first tree- minbucket=1, minsplit=2, cp=0.1
fit <- rpart(Survived ~ .,data=train, method="class",control=rpart.control(minbucket=1, minsplit=2, cp=0.1))
prp(fit)
preds <- as.integer(predict(fit,type="class", test))-1
true <- as.integer(test$Survived)
error_rate_tree <- (table(preds,true)[1,2]+table(preds,true)[2,1])/sum(table(preds,true))
error_rate_tree

# second tree- minbucket=10, minsplit=6, cp=0.04
fit <- rpart(Survived ~ .,data=train, method="class",control=rpart.control(minbucket=10, minsplit=6, cp=0.04))
prp(fit)
preds <- as.integer(predict(fit,type="class", test))-1
true <- as.integer(test$Survived)
error_rate_tree <- (table(preds,true)[1,2]+table(preds,true)[2,1])/sum(table(preds,true))
error_rate_tree

# find the best tree over a range of 1000 different parameters
min <- 1
for (b in 1:10){
  for (s in 1:10){
    for (c in 1:10){
      buck <- b*5
      split <- s*2
      comp <- c*0.005
      fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + trav_as + title,data=train, method="class",control=rpart.control(minbucket=buck, minsplit=split, cp=comp))
      preds <- predict(fit, newdata = test, type="class")
      if (min > (table(preds,test$Survived)[1,2]+table(preds,test$Survived)[2,1])/sum(table(preds,test$Survived))){
        min <- (table(preds,test$Survived)[1,2]+table(preds,test$Survived)[2,1])/sum(table(preds,test$Survived))
        control = c(buck,split,comp)
      }
    }
  }
}

control
fit <- rpart(Survived ~ .,data=train, method="class",control=rpart.control(minbucket=control[1], minsplit=control[2], cp=control[3]))
prp(fit)
preds <- as.integer(predict(fit,type="class", test))-1
true <- as.integer(test$Survived)
error_rate_tree <- (table(preds,true)[1,2]+table(preds,true)[2,1])/sum(table(preds,true))
error_rate_tree

# --- b10 ---
# we copied this code for the three decision trees, it is presented here only once for estetic reasons
precision <- sum(preds & true) / sum(preds)
recall <- sum(preds & true) / sum(true)
Fmeasure <- 2 * precision * recall / (precision + recall)
precision
recall
Fmeasure

preds <- predict(fit, type="prob", test)[,2]
validate <- prediction(preds, true, label.ordering = NULL)
pref <- performance(validate,"tpr","fpr")
plot(pref,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
auc <- performance(validate,"auc")
auc <- unlist(slot(auc, "y.values"))
auc

# ------ Part c ------

fit <- cforest(as.factor(Survived) ~ ., data=train, controls=cforest_unbiased(ntree=2000, mtry=3))
preds <- as.integer(predict(fit, test, OOB=TRUE, type = "response"))-1
true <- as.integer(test$Survived)
error_rate_party <- (table(preds,true)[1,2]+table(preds,true)[2,1])/sum(table(preds,true))

error_rate_party

fit <- randomForest(as.factor(Survived) ~ ., data=train, importance=TRUE, ntree=2000)
preds <- as.integer(predict(fit, test))-1
true <- as.integer(test$Survived)
error_rate_party <- (table(preds,true)[1,2]+table(preds,true)[2,1])/sum(table(preds,true))

error_rate_party
