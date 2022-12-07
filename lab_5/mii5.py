import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#загрузка датасета
dataset = pd.read_csv(r"emotions.csv")
#выделение целевого столбца
train_labels = dataset['Emotion']

#очистка данных
def remove_un(data):

    data = re.sub(r'\W', ' ', str(data))
    data = re.sub(r'\s+[a-zA-Z]\s+', ' ', data)
    data = re.sub(r'\^[a-zA-Z]\s+', ' ', data)
    data = re.sub(r'\s+', ' ', data, flags=re.I)
    data = re.sub(r'^b\s+', '', data)
    data = data.lower()
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize(data)

    return data

dataset['Text'] = dataset['Text'].apply(remove_un)

#разделение датасета на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(dataset['Text'], train_labels, test_size=0.1, random_state=0)
#векторизация данных
tfidf = TfidfVectorizer(stop_words = "english")
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

#обучение модели
sgd = SGDClassifier (loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
sgd.fit(tfidf_train, y_train)
y_pred = sgd.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Точность SGD-классификатора: {round(score * 100, 2)}%')

plt.subplot(1, 2, 1)
plt.hist(y_test)
plt.subplot(1, 2, 2)
plt.hist(y_pred)
plt.show()

svc = LinearSVC()
svc.fit(tfidf_train, y_train)
y_pred = svc.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Точность SVC-классификатора: {round(score * 100, 2)}%')

plt.subplot(1, 2, 1)
plt.hist(y_test)
plt.subplot(1, 2, 2)
plt.hist(y_pred)
plt.show()

#обучение модели
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Точность пассивно-агрессивного классификатора: {round(score * 100, 2)}%')

plt.subplot(1, 2, 1)
plt.hist(y_test)
plt.subplot(1, 2, 2)
plt.hist(y_pred)
plt.show()
