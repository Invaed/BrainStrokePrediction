#Install necessary packages
#install.packages("devtools")
#install.packages("klaR")
#install.packages("psych")
# install.packages("ggstatsplot")
# install.packages("tidyverse")
# install.packages("ROSE")
# install.packages("naniar")
library(devtools)  # You need to install this package!
library(DMwR)
library(ggstatsplot)
library(ROSE)
library(adabag)
library(rpart) 
library(caret)
library(klaR)
library(ggplot2)
library(MASS)
library(gains)
library(psych)# for describing the data
library(tidyverse)
library(neuralnet)
library(nnet)
library(caret)
library(e1071)
library(naniar)
library(randomForest)
rm(list = ls())

setwd("D:/GSU/CIS8695_Big Data Analytics/CIS8695_Project")
Stoke.df <- read.csv("healthcare-dataset-stroke-data.csv",stringsAsFactors = F)
str(Stoke.df[,-c(1)])
summary(Stoke.df)
Stoke.df["bmi"][Stoke.df["bmi"] == "N/A"] <- NA
summary(Stoke.df,na.rm=TRUE)
# colSums(is.na(Stoke.df))
Stoke.df$stroke = factor(Stoke.df$stroke,levels=c(0,1),labels=c("0","1"))
levels(Stoke.df$stroke) <- make.names(levels(factor(Stoke.df$stroke)))

#Finding the missing/null values
colSums(is.na(Stoke.df))
gg_miss_var(Stoke.df)

#Removing the columns with NA values and convert the BMI column to numeric
Stoke.df<-Stoke.df[complete.cases(Stoke.df), ]
Stoke.df['bmi']<-as.numeric(unlist(Stoke.df['bmi']))#Convert BMI to 
sapply(Stoke.df, typeof)#Check the type of  column

#Plotting Boxplot for outlier detection
boxplot(Stoke.df[,c("age","avg_glucose_level","bmi")],xlab="Outliers Detection")

#Handling outliers

Q <- quantile(Stoke.df$avg_glucose_level, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(Stoke.df$avg_glucose_level)
upr <-  Q[2]+1.5*iqr 
lowr<- Q[1]-1.5*iqr 
Stoke.df<- subset(Stoke.df, Stoke.df$avg_glucose_level > lowr & Stoke.df$avg_glucose_level <upr)

Q <- quantile(Stoke.df$bmi, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(Stoke.df$bmi)
upr <-  Q[2]+1.5*iqr 
lowr<- Q[1]-1.5*iqr 
Stoke.df<- subset(Stoke.df, Stoke.df$bmi > lowr & Stoke.df$bmi <upr)

#Plotting Boxplot for outlier detection
boxplot(Stoke.df[,c("age","avg_glucose_level","bmi")],xlab="Outliers Detection")

summary(Stoke.df,na.rm=TRUE)

#Finding the missing/null values
colSums(is.na(Stoke.df))
gg_miss_var(Stoke.df)

#Handling Categorical Variables

Stoke.df=Stoke.df %>% mutate(dummy=1) %>%
  spread(key=gender,value=dummy, fill=0)

Stoke.df=Stoke.df %>% mutate(dummy=1) %>%
  spread(key=ever_married,value=dummy, fill=0)

Stoke.df=Stoke.df %>% mutate(dummy=1) %>%
  spread(key=work_type,value=dummy, fill=0)

Stoke.df=Stoke.df %>% mutate(dummy=1) %>%
  spread(key=Residence_type,value=dummy, fill=0)

Stoke.df=Stoke.df %>% mutate(dummy=1) %>%
  spread(key=smoking_status,value=dummy, fill=0)

#Renaming the column names
names(Stoke.df)[names(Stoke.df)=="No"]<-"Unmarried" 
names(Stoke.df)[names(Stoke.df)=="Yes"]<-"Married"
names(Stoke.df)[names(Stoke.df)=="formerly smoked"]<-"formerly_smoked" 
names(Stoke.df)[names(Stoke.df)=="never smoked"]<-"never_smoked"
names(Stoke.df)[names(Stoke.df)=="Self-employed"]<-"Self_employed"

#Balancing the data
#Pie chart depicting Data distribution
pie(table(Stoke.df$stroke),labels=c("Not a Stroke","Stroke"),main="Raw Data Distribution")

Stoke.df[, -c(1,7)] <- scale(Stoke.df[, -c(1,7)])

#Splitting data 
set.seed(1)
train.index <- sample(nrow(Stoke.df), nrow(Stoke.df)*0.7)  
train.df <- Stoke.df[train.index, -c(1)]
valid.df <- Stoke.df[-train.index,-c(1)]


#train.df <- SMOTE(stroke ~ ., train.df, perc.over = 10000,perc.under=150)

train.df <- SMOTE(stroke ~ ., train.df, perc.over = 1000,perc.under=100)
table(train.df$stroke)
pie(table(train.df$stroke),labels=c("Not a Stroke","Stroke"),main="over sampled Data Distribution")


predictors=-c(6)
outcomeName<-c("stroke")

# Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = 'final',
  classProbs = T)

#levels(Stoke.df$stroke) <- make.names(levels(factor(Stoke.df$stroke)))

#Training a logistic model
model_lr<-train(train.df[,predictors],train.df[,outcomeName],method='glm',
                trControl=fitControl,tuneLength=3)
#Predicting using logistic model
valid.df$pred_lr<-predict(object = model_lr,valid.df[,predictors,drop = FALSE])
valid.df$pred_lr.prob<-predict(object = model_lr,valid.df[,predictors,drop = FALSE],type="prob")
#Checking the accuracy of the logistic model
con_lr<-confusionMatrix(valid.df$stroke,valid.df$pred_lr)
con_lr


#Random Forest
model_rf<-train(train.df[,predictors],train.df[,outcomeName],method='rf',
                trControl=fitControl,tuneLength=3)
#Predicting using random forest model
valid.df$pred_rf<-predict(object = model_rf,valid.df[,predictors])
valid.df$pred_rf.prob<-predict(object = model_rf,valid.df[,predictors],type="prob")
#Checking the accuracy of the random forest model
con_rf<-confusionMatrix(valid.df$stroke,valid.df$pred_rf)
con_rf


#KNN
model_knn<-train(train.df[,predictors],train.df[,outcomeName],method='knn',tuneLength=3)
valid.df$pred_knn<-predict(object = model_knn,valid.df[,predictors])
valid.df$pred_knn.prob<-predict(object = model_knn,valid.df[,predictors],type="prob")
con_knn<-confusionMatrix(as.factor(valid.df$stroke),as.factor(valid.df$pred_knn))
con_knn


# Ensemble using Averaging
# Taking average of predicted probabilities
valid.df$pred_avg<-(valid.df$pred_rf.prob$X1+valid.df$pred_lr.prob$X1+valid.df$pred_knn.prob$X1)/3
#Splitting into binary classes at 0.5
valid.df$pred_class<-as.factor(ifelse(valid.df$pred_avg>0.5,'X1','X0'))
ensemble.averaging<-confusionMatrix(valid.df$stroke,valid.df$pred_class)
ensemble.averaging

# Ensemble using Majority Voting
valid.df$pred_majority<-as.factor(ifelse(valid.df$pred_rf=='X1' & valid.df$pred_knn=='X1','X1',
                                         ifelse(valid.df$pred_rf=='X1' & valid.df$pred_lr=='X1','X1',
                                                ifelse(valid.df$pred_knn=='X1' & valid.df$pred_lr=='X1','X1','X0'))))
ensemble.voting<-confusionMatrix(valid.df$stroke,valid.df$pred_majority)
ensemble.voting

#Neural Network
# model_nn<-train(train.df[,predictors],train.df[,outcomeName],method='nnet',
#                 trControl=fitControl,tuneLength=3)
# summary(model_nn)
# valid.df$pred_nn<-predict(object = model_nn,valid.df[,predictors])
# valid.df$pred_nn.prob<-predict(object = model_nn,valid.df[,predictors],type="prob")
# con_nn<-confusionMatrix(as.factor(valid.df$stroke),as.factor(valid.df$pred_nn))
# con_nn
# plot(model_nn)

NN = neuralnet(stroke ~ age+hypertension+heart_disease+avg_glucose_level+bmi+
              Female+Male+Other+Unmarried+Married+children+Govt_job+Never_worked+
              Private+Self_employed+Rural+Urban+formerly_smoked+never_smoked+smokes+Unknown,
               train.df, hidden = 3 , linear.output = T )
# plot neural network
plot(NN)

valid.df$pred_nn<-compute(NN, valid.df)
valid.df$class <-apply(valid.df$pred_nn$net.result,1,which.max)-1
valid.df$class<- as.factor(ifelse(valid.df$class==1,'X1','X0'))
levels(valid.df$class) <- make.names(levels(factor(valid.df$class)))
con_nn <- confusionMatrix(as.factor(valid.df$class),as.factor(valid.df$stroke))
con_nn

#Plotting ROC

library(pROC)
valid.df$stroke_roc = ifelse(valid.df$stroke=='X1', 1, 0)
valid.df$pred_lr_roc = ifelse(valid.df$pred_lr=='X1', 1, 0)
valid.df$pred_rf_roc = ifelse(valid.df$pred_rf=='X1', 1, 0)
valid.df$pred_knn_roc = ifelse(valid.df$pred_knn=='X1', 1, 0)
valid.df$pred_avg_roc = ifelse(valid.df$pred_avg=='X1', 1, 0)
valid.df$pred_vot_roc = ifelse(valid.df$pred_majority=='X1', 1, 0)
valid.df$pred_nn_roc = ifelse(valid.df$pred_nn=='X1', 1, 0)
roc_lr <- plot(roc(valid.df$stroke_roc, valid.df$pred_lr_roc), print.auc = FALSE, col = "blue", lty=1)
roc_rf <- plot(roc(valid.df$stroke_roc, valid.df$pred_rf_roc), print.auc = FALSE, col = "black", add = TRUE, lty=1)
roc_knn <- plot(roc(valid.df$stroke_roc, valid.df$pred_knn_roc), print.auc = FALSE, col = "red", add = TRUE, lty=1)
roc_avg <- plot(roc(valid.df$stroke_roc, valid.df$pred_avg_roc), print.auc = FALSE, col = "pink", add = TRUE, lty=1)
roc_vot <- plot(roc(valid.df$stroke_roc, valid.df$pred_vot_roc), print.auc = FALSE, col = "green", add = TRUE, lty=1)
roc_nn <- plot(roc(valid.df$stroke_roc, valid.df$pred_nn_roc), print.auc = FALSE, col = "cyan", add = TRUE, lty=1)

plot_colors <- c("blue","black","red","pink", "green", "cyan")
legend(x = "bottomright",inset = 0,legend = c(paste("Logistic (Auc=",auc(roc_lr),")"), 
                                      paste("Random Forest (Auc=",auc(roc_rf),")"),
                                      paste("K-Nearest Neighbor (Auc=",auc(roc_knn),")"),
                                      paste("Averaging (Auc=",auc(roc_avg),")"),
                                      paste("Voting (Auc=",auc(roc_vot),")"),
                                      paste("Neural Networks (Auc=",auc(roc_nn),")")),col=plot_colors, lwd=2, cex=0.9, bty='n', horiz = FALSE)


#
c1<-rbind("Logistic Regression","Random Forest","KNN","Averaging","Voting","Nueral Networks")
c2<-rbind(con_lr$overall[1],con_rf$overall[1],con_knn$overall[1],ensemble.averaging$overall[1],ensemble.voting$overall[1], con_nn$overall[1])
c3<-rbind(auc(roc_lr),auc(roc_rf),auc(roc_knn),auc(roc_avg),auc(roc_vot), auc(roc_nn))
D1<-cbind(c1,c2,c3)
D1


