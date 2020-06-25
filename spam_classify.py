import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def Pretreatment(text):
    STEMMER = PorterStemmer()
    PUNCT_TO_REMOVE = string.punctuation
    STOPWORDS = set(stopwords.words("english"))
    text = text.lower()
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([STEMMER.stem(word) for word in text.split()])
    return text


#   读取数据
Entire_Dataset = pd.read_csv("data/train.csv", encoding='utf-8')

#   预处理
Entire_Dataset['Email'] = Entire_Dataset['Email'].apply(Pretreatment)

#   构造训练集和验证集
train, test = train_test_split(Entire_Dataset, random_state=1212, test_size=0.2, shuffle=True)

#   特征提取使用词袋模型
vectorizer = CountVectorizer()
train_cnt = vectorizer.fit_transform(train.Email)
test_cnt = vectorizer.transform(test.Email) 

#   根据词频矩阵转化TF-IDF矩阵
transformer = TfidfTransformer()
train_tfidf = transformer.fit_transform(train_cnt)
test_tfidf = transformer.transform(test_cnt)

#   贝叶斯
clf = MultinomialNB()
clf.fit(train_cnt, train.Label)
print("NB score: ", clf.score(test_cnt, test.Label))
clf.fit(train_tfidf, train.Label)
print("NB(TF-IDF) score: ", clf.score(test_tfidf, test.Label))

#   SVM
svm = LinearSVC()
svm.fit(train_cnt, train.Label)
print("SVM score: ", svm.score(test_cnt, test.Label))
svm.fit(train_tfidf, train.Label)
print("SVM(TF-IDF) score: ", svm.score(test_tfidf, test.Label))

#   Logistic
lr_crf = LogisticRegression(max_iter=150, penalty='l2', solver='lbfgs', random_state=0)
lr_crf.fit(train_cnt, train.Label)
print("LR score: ", lr_crf.score(test_cnt, test.Label))
lr_crf.fit(train_tfidf, train.Label)
print("LR(TF-IDF) score: ", lr_crf.score(test_tfidf, test.Label))

#   随机森林
rf = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None, verbose=0, n_jobs=-1)
rf.fit(train_cnt, train.Label)
print("RF score: ", rf.score(test_cnt, test.Label))
rf.fit(train_tfidf, train.Label)
print("RF(TF-IDF) score: ", rf.score(test_tfidf, test.Label))
