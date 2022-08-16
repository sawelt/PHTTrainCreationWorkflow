# PHT Train Creation Workflow
To achieve FAIRification of various digital components in a Distributed Analytics (DA) infrastructure, in this case, the Personal Health Train (PHT) infrastructure, two workflows have been designed and developed as part of this project. 
* _Train Creation Workflow_ : The workflow is a step-by-step procedure to build a Train image. A Train image consists of the data analysis task, required dependencies, image definition file (for example, Dockerfile), a list of connection interfaces, and a list of metadata.
* _Train Storehouse Platform_ : The platform is used to visualize, review, and approve published Train images to be deployed in a PHT infrastructure. It consists of a PHT community-driven approval process to check the Train and its contents before releasing it into the PHT ecosystem.

The proposed workflows are evaluated in a three-folded approach. Firstly, a technical evaluation using three data use cases, and later, a user evaluation using a survey-based method. The final assessment is done concerning the FAIR guidelines.

### Usecase-based Evaluation
The purpose of this evaluation is to test the functionalities of the implemented workflows. Three different types of use cases, based on the data set and programming languages used, were chosen to evaluate the flexibility of the workflows.

#### Drug Reviews Sentiment Analysis
We performed sentiment analysis on the drug reviews data set from the [UCI ML repository](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+\%28Drugs.com\%29). This analysis aimed to determine whether a given drug review contains positive or negative user emotions. The data set has more than 200k patient reviews on specific drugs in textual format, their medical conditions, a 10-star rating to reflect their satisfaction with the drug, and a usefulness count by other patients who found the drug review helpful.

The CSV data set was divided into three subsets and distributed to three different Stations to simulate the distributed setting for PHT. Two classification models written in Python, [XGBoostClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html\#xgboost.XGBClassifier) and [LightGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html), were deployed to perform the analysis. The Train image for this analysis was created using the _Train Creation Workflow_. It consists of the data analysis task in Python, _requirements.txt_ for dependencies, a standard template based _Dockerfile_, and three JSON files: connection interfaces, metadata, and user feedback.

#### COVID-19 Image Classification
We performed image classification on chest X-rays or CT images collected at the beginning of the COVID-19 pandemic in 2020 by the [University of Montreal](https://github.com/ieee8023/covid-chestxray-dataset). The analysis aimed to classify the images into Covid, Normal, and Viral Pneumonia. The image data set was divided into specific categories to determine the target label. It has 251 images for training and 66 images for testing the model.

The classification algorithm was written in Python and used Keras to train and build the classification model. The Train image was built and pushed to the Train repository using the _Train Creation Workflow_. It consists of the data classification task in Python, _requirements.txt_ for dependencies, a user-defined _Dockerfile_, and three JSON files: connection interfaces, metadata, and user feedback.

#### Fetal Health Classification
We carried out a 3-way classification to build a model to predict fetal health using the Cardiotocography (CTG) data set from the [UCI ML repository](https://archive.ics.uci.edu/ml/datasets/cardiotocography). The CSV data set consists of uterine contraction (UC) measurements and fetal health rate (FHR) features on a cardiotocograph. The measurements are classified into three target labels by expert obstetricians: Normal, Suspect, and Pathological. It has 2126 measurements and 23 attributes. 

The classification algorithm was written in R and used the [Random Forest Classifier package](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) to train and build the classification model. The Train image was created and deployed to the PHT ecosystem using the _Train Creation Workflow_ and the _Train Storehouse Platform_. The Train consists of the data classification task in R, _installPackages.R_ for dependencies, a standard template based _Dockerfile_, and three JSON files: connection interfaces, metadata, and user feedback.

