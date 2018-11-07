#Deep Learning with H20
library(tidyverse)
# Format data with no factor
data(BreastCancer, package = 'mlbench') # Load data from mlbench package
dat <- BreastCancer[, -1]  # Remove the ID column
dat[, c(1:ncol(dat))] <- sapply(dat[, c(1:ncol(dat))], as.numeric) # Convert factors into numeric
## Start a local cluster with default parameters
#Make our target variable Categorical
dat<-dat %>% 
  mutate(Class=as.factor(Class))
library(h2o)
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
## Convert Breast Cancer into H2O
dat.h2o <- as.h2o(dat, destination_frame = "midata")
#Set target
y<-"Class"
x<-setdiff(names(dat.h2o),y)
#splitting our data
parts<-h2o.splitFrame(dat.h2o,0.8)
train_set<-parts[[1]]
test_set<-parts[[2]]
m<-h2o.deeplearning(x,y,train_set)
p<-h2o.predict(m,test_set)
#get MSE
h2o.mse(m)
#Confusion Matrix
h2o.performance(m,test_set)

#Scaling
dat1<-dat %>% 
  mutate_all(as.numeric)
shapiro.test(dat1$Class)
#
h2o.describe(dat.h2o)
h2o.levels(dat.h2o)

#some plots
 histpl<-h2o.hist(as.h2o(dat$Marg.adhesion))
 plot(histpl,col=rainbow(7),xlab="Adhesion",ylab="Freq"
      ,main="Adhesion Distribution")
 #Splitting our data into three
 partitioned_data<-h2o.splitFrame(dat.h2o,c(0.65,0.25))
  training<-partitioned_data[[1]]
  validate<-partitioned_data[[2]]
  test<-partitioned_data[[3]]
  y<-"Class"
  x<-setdiff(names(dat.h2o),y)
  #Make predictions
  train_mod<-h2o.deeplearning(x,y,training)
 validation<-h2o.predict(train_mod,validate)
 h2o.confusionMatrix(train_mod,validate)
#Predict on Test
 tested<-h2o.predict(train_mod,test)

