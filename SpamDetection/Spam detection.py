import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_selection import chi2, SelectKBest
import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer

#Read the Dataset
Dataset = pd.read_csv ('/SpamDetection/spam.csv', encoding = "ISO-8859-1")

#Replace ham and spam with 1s and 0s
Dataset.loc[Dataset['v1'] == 'spam', 'v1'] = 1
Dataset.loc[Dataset['v1'] == 'ham', 'v1'] = 0

#Change datatype of labes to int
Dataset['v1'] = Dataset['v1'].astype('int')

#Split Dataset into trainning and testing
Training_data = Dataset.sample(frac=0.9, random_state=25)
Testing_data = Dataset.drop(Training_data.index)

#Applying Tfidf Vectorizer on Dataset
tf_vectorizer = TfidfVectorizer()
#For digram
#ngram_range = (2,2)


trainTfidf = tf_vectorizer.fit_transform(Training_data['v2']).toarray()
testTfidf = tf_vectorizer.transform(Testing_data['v2'])

#Applying Naive Bayes model
naive_bayes_classifier = MultinomialNB(alpha=1)
naive_bayes_classifier.fit(trainTfidf, Training_data['v1'])

#predicting results using Naive Bayes
TfidfPredictions = naive_bayes_classifier.predict(testTfidf)


#Comparing model accuracy with test data results
print("Tfidf results")
print(metrics.accuracy_score(Testing_data['v1'], TfidfPredictions))
print(metrics.confusion_matrix(Testing_data['v1'], TfidfPredictions))

#Applying Count Vectorizer on Dataset
count_vectorizer = CountVectorizer()
trainCount = count_vectorizer.fit_transform(Training_data['v2'])
trainCountDF = pd.DataFrame(trainCount.toarray(), columns=count_vectorizer.get_feature_names_out())
testCHI2 = count_vectorizer.transform(Testing_data['v2'])

#Selecting most important features to determine ham or spam
chi2_selector = SelectKBest(chi2, k=1000)
top = chi2_selector.fit_transform(trainCount, Training_data['v1'])
cols_indexs = chi2_selector.get_support(indices=True)
features_df_new = trainCountDF.iloc[:,cols_indexs]
res = chi2_selector.transform(testCHI2)


#Applying Naive Bayes model
naive_bayes_classifier = MultinomialNB(alpha=1)
naive_bayes_classifier.fit(features_df_new, Training_data['v1'])

#predicting results using Naive Bayes
TfidfPredictions = naive_bayes_classifier.predict(res)

#Comparing model accuracy with test data results
print("Chi-squared results")
print(metrics.accuracy_score(Testing_data['v1'], TfidfPredictions))
print(metrics.confusion_matrix(Testing_data['v1'], TfidfPredictions))

for i in len(Training_data):
  tokenizer = Tokenizer()
  xd = tokenizer.fit_on_texts(Training_data['v2'])