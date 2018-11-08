
#Basic Machine Learning on kaggle titanic dataset
nomsg<-suppressMessages
nomsg(library(tidyverse))
nomsg(library(caret))
nomsg(library(mice))
nomsg(library(xgboost))
library(RANN)
library(caretEnsemble)
library(Amelia)
train<-read.csv("train.csv",stringsAsFactors = F)
train<-as.tibble(train)
#Remove cabin,name,Ticket for now as these may not predict survival 

#View and deal with nas
newtrain<-train %>%
  mutate(PassengerId=as.factor(PassengerId),Pclass=as.factor(Pclass),
         Survived=as.factor(Survived),Embarked=as.factor(Embarked),
         Sex=as.factor(Sex),
         AgeGroup=as.factor(findInterval(Age,c(0,18,35,100)))) %>% 
  select(-PassengerId,-Name,-Ticket,-Cabin)
#Change levels
levels(newtrain$AgeGroup)<-c("Young","Mid Age","Aged")
levels(newtrain$Sex)<-c("F","M")
#Impute median
newtrain_1<-preProcess(newtrain,method="medianImpute")
newtrain_imp<-predict(newtrain_1,newtrain)
#checkNAs
anyNA(newtrain_imp)
#View NAS
newtrain_imp %>% 
  map_dbl(~sort(sum(is.na(.x),decreasing=T)))
#redo levels
newtrain_imp<-newtrain_imp %>% 
  mutate(AgeGroup=as.factor(findInterval(Age,c(0,18,35,100))))
levels(newtrain_imp$AgeGroup)<-c("Young","Mid Age","Aged")
anyNA(newtrain_imp)
#Let's visualise survival by Age Group
newtrain_imp %>% 
  ggplot(aes(Survived,fill=Sex))+geom_histogram(stat="count")+facet_wrap(AgeGroup~Pclass)+
  ggtitle("Survival by class,Agegroup and Gender")+
  theme(plot.title=element_text(hjust=0.5))+
  scale_fill_manual(values=c("orange","steelblue4"))
#The graph does suggest that being of mid Age and embarking in the third class made you more likely to die
#Overall more women than men survived.

#Partition our data into a training and test dataset
#Deep deep water
nomsg(library(h2o))
h2o.init()
#Split our data frame
train_deep<-h2o.splitFrame(as.h2o(newtrain_imp),0.75)
train_set<-train_deep[[1]]
validate_set<-train_deep[[2]]
#Make predictions

y<-"Survived"
x<-setdiff(names(newtrain_imp),y)
gbm_fit<-h2o.gbm(x,y,train_set,nfolds = 10,
                 ntrees=5970,
                 learn_rate = 0.7,
                 max_depth = 50,
                             seed = 2,
                 nbins_cats = 156,
                 keep_cross_validation_predictions = T,
                 keep_cross_validation_fold_assignment = T,
               validation_frame = validate_set
                 )
h2o.confusionMatrix(gbm_fit)
mod1<-h2o.performance(gbm_fit)
h2o.accuracy(mod1)
#make predictions on Test set
test<-read.csv("test.csv",stringsAsFactors = F)
test<-as.tibble(test)
newtest<-test %>% 
  mutate(PassengerId=as.factor(PassengerId),Pclass=as.factor(Pclass),
         Sex=as.factor(Sex),
         Embarked=as.factor(Embarked),
         AgeGroup=as.factor(findInterval(Age,c(0,18,35,100)))) %>% 
  select(-Ticket,-Name,-Cabin)
levels(newtest$Embarked)<-c("","C","Q","S")
levels(newtest$AgeGroup)<-c("Young","Mid Age","Aged")
levels(newtest$Sex)<-c("F","M")
#Find NAs
newtest %>% 
  map_lgl(~anyNA(.x))
#Preprocess and remove NAs from age and Fare
newtest_1<-preProcess(newtest,method="medianImpute")
newtest_imp<-predict(newtest_1,newtest)
#Check Dona
newtest_imp<-newtest_imp %>% 
  mutate(AgeGroup=as.factor(findInterval(Age,c(0,18,35,100))))
levels(newtest_imp$AgeGroup)<-c("Young","Mid Age","Aged")
levels(newtest_imp$Sex)<-c("F","M")
newtest_imp_h2o<-as.h2o(newtest_imp)

h2o_prediction<-h2o.predict(gbm_fit,newtest_imp_h2o)
mymodel<-gbm_fit@model$cross_validation_holdout_predictions_frame_id$name
as.data.frame(gbm_fit@model$variable_importances) %>% 
  ggplot(aes(scaled_importance,variable,fill=variable))+geom_col()+
  labs(x="Relative Importance",y="Feature")+
  ggpubr::theme_cleveland()+
  ggtitle("Feature Importance in The GBM Model")
  
#summary with class predictions= T
predictions<-as.data.frame(h2o_prediction) %>% 
  select(predict)
predictions_h2o<-cbind(newtest_imp,predictions)
names(predictions_h2o)
test_h2o<-predictions_h2o %>%
  rename(Survived=`predict`) %>% 
  select(PassengerId,Survived)
write.csv(test_h2o,"waterwater.csv",row.names = F)
#Check confusion Matrix
