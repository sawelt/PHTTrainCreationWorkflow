import warnings
warnings.simplefilter('ignore')
import os
from minio import Minio
import pandas as pd
import numpy as np
import re
import sys
from random import randrange
from string import punctuation
from nltk.stem import SnowballStemmer
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class LoadAndPrepareData:
    def __init__(self, drop_na):
        self.drop_na = drop_na

    @staticmethod
    def getDataFrameFromMinIO():
        # ENV Variables
        """minio_address = str(os.environ['MINIO_ADDRESS'])
        minio_port = str(os.environ['MINIO_PORT'])
        minio_access_key = str(os.environ['MINIO_ACCESS'])
        minio_secret_key = str(os.environ['MINIO_SECRET'])
        bucket_name = str(os.environ['MINIO_BUCKET_NAME'])
        object_name = str(os.environ['MINIO_OBJECT_NAME'])"""

        minio_address = '127.0.0.1'
        minio_port = '9000'
        minio_access_key = 'admin'
        minio_secret_key = 'password'
        bucket_name = 'drugreviews'
        object_name = 'drugs_review_dataset.csv'

        minioClient = Minio(
            '{0}:{1}'.format(minio_address, minio_port),
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )
        res = minioClient.get_object(bucket_name, object_name)
        df_drugs = pd.read_csv(res)

        #df_drugs = pd.read_csv("drugs_review_dataset.csv")
        print("The shape of the dataset is: ", df_drugs.shape)

        return df_drugs

    def prepareData(self, df_drugs):
        # Drop NA rows
        if self.drop_na:
            df_drugs = df_drugs.dropna(how='any', axis=0)
            print("The shape of the dataset after null values removal:", df_drugs.shape)

        # Lowercase column names
        df_drugs.columns = df_drugs.columns.str.lower()

        # Sorting the dataframe based on uniqueID
        df_drugs.sort_values(['uniqueid'], ascending=True, inplace=True)
        df_drugs.reset_index(drop=True, inplace=True)

        # Date format conversion
        df_drugs['date'] = pd.to_datetime(df_drugs['date'])
        return df_drugs


class DataPreProcessingDrugs:
    def __init__(self, sentiment_threshold_rate, df_drugs):
        self.sentiment_threshold_rate = sentiment_threshold_rate
        self.df_drugs = df_drugs

    def addSentimentRateToDataframe(self):
        # Add new column 'sentiment' based on the drug ratings
        # 1: positive sentiment (rating above 6)
        # 0: negative sentiment (rating below or equal to 6)
        self.df_drugs['sentiment_rate'] = self.df_drugs['rating'].apply(
            lambda x: 1 if x > self.sentiment_threshold_rate else 0)

    @staticmethod
    def cleanReviews(user_review):
        # 1. Change user review to lower case
        updated_review = user_review.str.lower()

        # 2. Replace &#039 pattern
        updated_review = updated_review.str.replace("&#039;", "")

        # 3. Remove special chars
        updated_review = updated_review.str.replace(r'[^\w\d\s]', ' ')

        # 4. Remove non-ASCII chars
        updated_review = updated_review.str.replace(r'[^\x00-\x7F]+', ' ')

        # 5. Remove leading and trailing whitespaces
        updated_review = updated_review.str.replace(r'^\s+|\s+?$', '')

        # 6. Replace continuous spaces with a single space
        updated_review = updated_review.str.replace(r'\s+', ' ')

        # 7. Replace repeating punctuations (full-stop)
        updated_review = updated_review.str.replace(r'\.{2,}', ' ')
        return updated_review

    @staticmethod
    def expandShortenedWordsToCompleteWords(review):
        # Shortened words dictionary
        shortened_words = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
            "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
            "doesn't": "does not", "doesn’t": "does not", "don't": "do not", "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he had", "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
            "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
            "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
            "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
            "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
            "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
            "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have", "there's": "there is", "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
            "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
            "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
            "when's": "when is",
            "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
            "why've": "why have",
            "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
            "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y’all": "you all",
            "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have", "ain’t": "am not", "aren’t": "are not", "can’t": "cannot",
            "can’t’ve": "cannot have", "’cause": "because", "could’ve": "could have", "couldn’t": "could not",
            "couldn’t’ve": "could not have", "didn’t": "did not", "don’t": "do not", "hadn’t": "had not",
            "hadn’t’ve": "had not have",
            "hasn’t": "has not", "haven’t": "have not", "he’d": "he had", "he’d’ve": "he would have",
            "he’ll": "he will",
            "he’ll’ve": "he will have", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you",
            "how’ll": "how will",
            "how’s": "how is", "i’d": "i would", "i’d’ve": "i would have", "i’ll": "i will", "i’ll’ve": "i will have",
            "i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have",
            "it’ll": "it will", "it’ll’ve": "it will have", "it’s": "it is", "let’s": "let us", "ma’am": "madam",
            "mayn’t": "may not", "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have",
            "must’ve": "must have", "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not",
            "needn’t’ve": "need not have", "o’clock": "of the clock", "oughtn’t": "ought not",
            "oughtn’t’ve": "ought not have",
            "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have", "she’d": "she would",
            "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",
            "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have",
            "so’ve": "so have", "so’s": "so is", "that’d": "that would", "that’d’ve": "that would have",
            "that’s": "that is",
            "there’d": "there would", "there’d’ve": "there would have", "there’s": "there is", "they’d": "they would",
            "they’d’ve": "they would have", "they’ll": "they will", "they’ll’ve": "they will have",
            "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not", "we’d": "we would",
            "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are",
            "we’ve": "we have",
            "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are",
            "what’s": "what is", "what’ve": "what have", "when’s": "when is", "when’ve": "when have",
            "where’d": "where did",
            "where’s": "where is", "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have",
            "who’s": "who is", "who’ve": "who have", "why’s": "why is", "why’ve": "why have", "will’ve": "will have",
            "won’t": "will not", "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not",
            "wouldn’t’ve": "would not have", "y’all’d": "you all would", "y’all’d’ve": "you all would have",
            "y’all’re": "you all are",
            "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will",
            "you’ll’ve": "you will have", "you’re": "you are", "you’ve": "you have"
        }

        shortened_words_re = re.compile('(%s)' % '|'.join(shortened_words.keys()))

        def replace(match):
            return shortened_words[match.group(0)]

        return shortened_words_re.sub(replace, review)

    @staticmethod
    def getStopWords():
        stop_words = {"isn't", 'until', 'through', 'most', 'after', 'when', 'up', 'a', 'nor', 'we', "hasn't",
            'should', 'which', 's', "wasn't", "shouldn't", 'to', 'themselves', 'under', 'about', 'into', 'me',
            'during', "didn't", "you'd", "needn't", "she's", 'had', 'mightn', 'who', 'while', "mustn't", 'wasn',
            'your', 'there', 'hers', 'hadn', 'ma', 'will', 'with', 'off', 'what', 'been', "shan't", 'if', 'he',
            'why', 'were', 'd', 'being', 'than', 'be', 'of', 'at', "haven't", 'have', 'an', 'ours', 'are',
            'further', 'yourselves', 'or', 'just', "it's", 'm', 'against', 'any', "mightn't", 'them', 'whom',
            "doesn't", 'their', 'you', 'it', 'above', 'don', 'that', 'other', 'now', 'y', "that'll", 'not', 'his',
            'doing', 'needn', 'isn', 'mustn', 'these', 'him', 'herself', 'before', 'then', 'where', 'hasn', 'yourself',
            'itself', 'did', 'i', 'no', 'between', 'yours', 'couldn', 'very', 'was', 'few', 'over', "wouldn't", 'o',
            'can', 'its', "don't", 'am', "weren't", 'in', 'only', 'below', 'both', 'again', "won't", 'haven', 'theirs',
            'too', 'having', 'she', 'as', 'himself', "you'll", 'by', "you've", 'does', 'but', 'all', 'our', 'some', 'such',
            'didn', 'they', 'her', 'down', 'from', 'weren', 'do', 'ain', 'because', 'my', 've', 'here', 'so', 're', 'shan',
            'for', 'those', 'each', "couldn't", 'out', 'on', 'is', 'll', 'shouldn', 'wouldn', 'the', 'how', "hadn't",
            'aren', 'doesn', 'won', "should've", 't', 'and', 'own', 'more', "you're", 'once', "aren't", 'myself', 'same',
            'this', 'has', 'ourselves'}

        return stop_words

    def nlpSpecificDataPreProcessing(self):
        # 1. Clean the reviews
        self.df_drugs['clean_review'] = self.cleanReviews(self.df_drugs['review'])

        # 2. Replace shortened words with complete words
        self.df_drugs['clean_review'] = self.df_drugs['clean_review'].apply(
            lambda x: self.expandShortenedWordsToCompleteWords(x))

        # 3. Remove remaining punctuations
        punctuations = punctuation + '""“”’' + '−‘´°£€\—–&'
        self.df_drugs['clean_review'] = self.df_drugs['clean_review'].apply(
            lambda x: ''.join(word for word in x if word not in punctuations))

        # 4. Remove stopwords
        stop_words = self.getStopWords()
        self.df_drugs['clean_review'] = self.df_drugs['clean_review'].apply(
            lambda x: ' '.join(word for word in x.split() if word not in stop_words))

        # 5. Remove word stems using the Snowball Stemmer
        Snow_ball = SnowballStemmer("english")
        self.df_drugs['clean_review'] = self.df_drugs['clean_review'].apply(
            lambda x: " ".join(Snow_ball.stem(word) for word in x.split()))

        # 6. Partition date into day, month and year columns
        #self.df_drugs['date'] = pd.to_datetime(self.df_drugs['date'], errors='coerce')
        self.df_drugs['day'] = self.df_drugs['date'].dt.day
        self.df_drugs['month'] = self.df_drugs['date'].dt.month
        self.df_drugs['year'] = self.df_drugs['date'].dt.year

    @staticmethod
    def performSentimentPolarity(cleaned_review):
        sentiment_polarity = []
        for i in cleaned_review:
            analysis = TextBlob(i)
            sentiment_polarity.append(analysis.sentiment.polarity)
        return sentiment_polarity

    def addMoreStatisticsFeatures(self):
        # 1. Add word count column for each cleaned review
        self.df_drugs['count_word'] = self.df_drugs["clean_review"].apply(lambda x: len(str(x).split()))

        # 2. Add unique word count column for each cleaned review
        self.df_drugs['count_unique_word'] = self.df_drugs["clean_review"].apply(lambda x: len(set(str(x).split())))

        # 3. Add letters count column for each cleaned review
        self.df_drugs['count_letters'] = self.df_drugs["clean_review"].apply(lambda x: len(str(x)))

        # 4. Add punctuations count column for each original review
        self.df_drugs["count_punctuations"] = self.df_drugs["review"].apply(
            lambda x: len([c for c in str(x) if c in punctuation]))

        # 5. Add uppercase word count column for each original review
        self.df_drugs["count_words_upper"] = self.df_drugs["review"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()]))

        # 6. Add title case word count column for each original review
        self.df_drugs["count_words_title"] = self.df_drugs["review"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()]))

        # 7. Add stopwords count column for each original review
        stop_words = self.getStopWords()
        self.df_drugs["count_stopwords"] = self.df_drugs["review"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

        # 8. Add average word length column for each cleaned review
        self.df_drugs["mean_word_len"] = self.df_drugs["clean_review"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    def addColumnLabelEncoding(self):
        # Label Encoding Drugname and Conditions
        label_encoder_feat = {}
        for feature in ['drugname', 'condition']:
            label_encoder_feat[feature] = LabelEncoder()
            self.df_drugs[feature] = label_encoder_feat[feature].fit_transform(self.df_drugs[feature])

    def dataPreProcessing(self):
        print("Starting data pre-processing ...")
        # 1. Add new column sentiment rate
        self.addSentimentRateToDataframe()

        # 2. NLP specific pre processing
        self.nlpSpecificDataPreProcessing()

        # 3. Sentiment Polarity
        self.df_drugs['sentiment'] = self.performSentimentPolarity(self.df_drugs['clean_review'])

        # 4. Add more features to data set
        self.addMoreStatisticsFeatures()

        # 5. Add column label encoding
        self.addColumnLabelEncoding()
        print("Data pre-processing is complete!!!")

        return self.df_drugs


class AnalysisModels:
    def __init__(self, test_size, random_state, n_estimators, learning_rate, df_drugs):
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.df_drugs = df_drugs

    def defineFeaturesAndSplitDataset(self):
        features = self.df_drugs[['condition', 'usefulcount', 'day', 'month', 'year',
                                 'sentiment', 'count_word', 'count_unique_word', 'count_letters',
                                 'count_punctuations', 'count_words_upper', 'count_words_title',
                                 'count_stopwords', 'mean_word_len']]

        target = self.df_drugs['sentiment_rate']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size,
                                                            random_state=self.random_state)
        print("Training data set size: ", X_train.shape)
        print("Testing data set size: ", X_test.shape)
        return X_train, X_test, y_train, y_test

    def XGBClassifierModelResults(self, X_train, X_test, y_train, y_test):
        # Model I: XGBClassifier Training
        xgb_clf = XGBClassifier(n_estimators=self.n_estimators,
                                learning_rate=self.learning_rate,
                                max_depth=5,
                                eval_metric='error')

        model_xgb = xgb_clf.fit(X_train, y_train)

        # Predict and Results
        predictions_2 = model_xgb.predict(X_test)
        print("Model I XGBClassifier Accuracy: ", accuracy_score(y_test, predictions_2), '\n')
        print("The confusion Matrix is: \n")
        print(confusion_matrix(y_test, predictions_2), '\n')
        print(classification_report(y_test, predictions_2))

    def LGBMClassifierModelResults(self, X_train, X_test, y_train, y_test):
        # Model II: LightGBM
        clf = LGBMClassifier(n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate,
                             num_leaves=30,
                             subsample=.9,
                             max_depth=7,
                             reg_alpha=.1,
                             reg_lambda=.1,
                             min_split_gain=.01,
                             min_child_weight=2,
                             verbose=-1,
                             )
        model = clf.fit(X_train, y_train)

        # Predict and Results
        predictions = model.predict(X_test)
        print("Model II LightGBM Accuracy: ", accuracy_score(y_test, predictions), '\n')
        print("Confusion matrix is: \n")
        print(confusion_matrix(y_test, predictions), '\n')
        print(classification_report(y_test, predictions))

    def callAnalysisModels(self):
        X_train, X_test, y_train, y_test = self.defineFeaturesAndSplitDataset()
        # Call Model I
        self.XGBClassifierModelResults(X_train, X_test, y_train, y_test)

        # Call Model II
        self.LGBMClassifierModelResults(X_train, X_test, y_train, y_test)


class Run:
    def __init__(self, sentiment_threshold_rate: int, drop_na: bool, test_size: float,
                 random_state: int, n_estimators: int, learning_rate: float):
        self.sentiment_threshold_rate = sentiment_threshold_rate
        self.drop_na = drop_na
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def run(self):
        # Fetch data and prepare for analysis
        df_drugs = LoadAndPrepareData.getDataFrameFromMinIO()
        prepare_data = LoadAndPrepareData(self.drop_na)
        df_drugs = prepare_data.prepareData(df_drugs)

        # Data Preprocessing
        data_preprocess = DataPreProcessingDrugs(self.sentiment_threshold_rate, df_drugs)
        data_preprocess.dataPreProcessing()

        # Analysis Models
        analysis_model = AnalysisModels(self.test_size, self.random_state,
                                        self.n_estimators, self.learning_rate, df_drugs)
        analysis_model.callAnalysisModels()


if __name__ == "__main__":
    random_suffix = randrange(1000)
    file_name = 'sentiment_analysis_results_' + str(random_suffix) + '.txt'
    sys.stdout = open(file_name, 'w')
    Run(6, True, 0.3, 48, 500, 0.10).run()
    sys.stdout.close()
