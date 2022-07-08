# Import the libraries
library(randomForest)
library(caret)

# Fetch environment variables
#data_location<-Sys.getenv("DATA_LOCATION")
data_location="./Data/fetal_health.csv"

# Read the data set
health<-read.csv2(data_location, header = TRUE, sep = ",")
health$fetal_health=as.factor(health$fetal_health)

# Test-train split
set.seed(2022)
ind<-sample(2,nrow(health),replace=TRUE,prob=c(0.7,0.3))
train<-health[ind==1,]
test<-health[ind==2,]

# Create a Random Forest model
set.seed(2022)
rf<-randomForest(fetal_health ~ ., data=train)
print(rf)

# Perform prediction on test data
p2<-predict(rf,test)
confusionMatrix(p2,test$fetal_health)

# Tune the hyper parameters of Random Forest model
tuneRF(train[,-22],train[,22],stepFactor=0.5,plot=FALSE,ntreeTRY=300,trace=TRUE,improve=0.05)

# Update the model
set.seed(2022)
rf1<-randomForest(fetal_health~.,data=train,ntree=300,mtry=8,importance=TRUE,proximity=TRUE)
print(rf1)

# Perform prediction on test data again
p1<-predict(rf1,train)
confusionMatrix(p1,train$fetal_health)

