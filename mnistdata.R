#Exploring MNIST
library(h2o)
h2o.init()
mydata <- h2o.importFile("https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/iris_wheader.csv")
#Loading our Data from kaggle
mnist_train<-h2o.importFile("C:\\Users\\Lance Nelson\\Downloads\\mnist-in-csv\\mnist_train.csv")
mnist_test<-h2o.importFile("C:\\Users\\Lance Nelson\\Downloads\\mnist-in-csv\\mnist_test.csv")
#Initial data exploration
x<-1:784
y<-785
mnist_train[,y]<-as.factor(mnist_train[,y])
mnist_test[,y]<-as.factor(mnist_test[,y])
part_train<-h2o.splitFrame(mnist_train,1.0/6.0)
valid<-part_train[[1]]
train<-part_train[[2]]
#Visualization
avg<-matrix(h2o.mean(mnist_train[,x]),nrow=28)
image(avg,col=gray(256:0/256))
#Visualize sd
avgsd<-matrix(sapply(x, function(x)h2o.sd(mnist_train[,x])),nrow=28)
image(avgsd,col=gray(254:0/254))
#Models
NN_mod<-h2o.deeplearning(x,training_frame=train,hidden=c(400,200,2),
                         epochs = 60,
                         activation = "Tanh",
                         autoencoder = T)
nn_features<-h2o.deepfeatures(NN_mod,mnist_train,layer = 2)
#Plot
h2o.describe(nn_features)
qplotdata<-as.data.frame(nn_features)
qplot(DF.L2.C5,DF.L2.C8,data=qplotdata,color=as.character(as.vector(mnist_train[,1])))+
  ggtitle("A deep deep net")+
  ggpubr::theme_cleveland()+theme(plot.title = element_text(hjust=0.5),
                                  legend.title = element_blank())
  


