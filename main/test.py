import time
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn import svm
from sklearn.metrics import accuracy_score
from random import shuffle
from collections import OrderedDict
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_union
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import vstack
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import os
import glob
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import operator

path = '/home/frosk/Development/Star_Prediction/scaledata/merged'
feelings = "/home/frosk/Development/Star_Prediction/emotion_lex"

files = []

first = "/home/frosk/Development/Star_Prediction/scaledata/merged/merged_dennis_schwartz.txt"
files.append(first)
second = "/home/frosk/Development/Star_Prediction/scaledata/merged/merged_james_berardinelli.txt"
files.append(second)
third = "/home/frosk/Development/Star_Prediction/scaledata/merged/merged_scott_renshaw.txt"
files.append(third)
fourth = "/home/frosk/Development/Star_Prediction/scaledata/merged/merged_steve_rhodes.txt"
files.append(fourth)




def readin(filename):
    with open(filename,"r") as subjtext:
        for doc in subjtext.read().split("\n"):
            raw_data.append(doc)
        subjtext.close()

    for field in raw_data:
        try:
            raw_corpus[field.split("\t")[0]] = field.split("\t")[1:]
        except IndexError:
            continue

    items = raw_corpus.items()
    shuffle(list(items))
    new_dic = OrderedDict(items)

    for i in new_dic.items():
        try:
            corpus_combined.append(i[1][3])
            labels_combined.append(i[1][0])

        except IndexError:
            continue


def fill_feeling_dic():
    for filename in glob.glob(os.path.join(feelings, '*.txt')):
        with open(filename) as infile:
            for word in infile.read().split("\n"):
                try:
                    feeling_dic[word] += [filename[52:-4]]
                except KeyError:
                    feeling_dic[word] = [filename[52:-4]]


class EmotionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, corpus_combined):
        features_list = []

        for doc in corpus_combined:
            feat = {
            "anger" : 0,
            "disgust" : 0,
            "fear" : 0,
            "joy" : 0,
            "sadness" : 0,
            "surprise" : 0
            }
            feat2= {}

            for word in doc.split():
                try:
                    for feeling in feeling_dic[word]:
                        if feeling =="anger":
                            feat["anger"] +=1
                        elif feeling =="disgust":
                            feat["disgust"] +=1
                        elif feeling =="fear":
                            feat["fear"] +=1
                        elif feeling =="joy":
                            feat["joy"] +=1
                        elif feeling =="sadness":
                            feat["sadness"] +=1
                        elif feeling =="surprise":
                            feat["surprise"] +=1
                except KeyError:
                    continue

            maxi = max(feat.items(), key=operator.itemgetter(1))[0]
            feat2["max"] = maxi
            features_list.append(feat2)

        return features_list


def fill_polarity_dic():
    with open("/home/frosk/Development/Star_Prediction/negative-words.txt") as nega, open("/home/frosk/Development/Star_Prediction/positive-words.txt") as posi:
        for word in nega.read().split("\n"):
            try:
                polarity_dic[word] += ["neg"]
            except KeyError:
                polarity_dic[word] = ["neg"]

        for word in posi.read().split("\n"):
            try:
                polarity_dic[word] += ["pos"]
            except KeyError:
                polarity_dic[word] = ["pos"]
        nega.close()
        posi.close()


class PolarityTransformer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, corpus_combined):
        features_list = []
        for doc in corpus_combined:
            feat = {}
            pos_count = 0
            neg_count = 0
            for word in doc.split():
                try:
                    if polarity_dic[word][0]=="neg":
                        neg_count +=1
                    elif polarity_dic[word][0]=="pos":
                        pos_count +=1
                    elif len(polarity_dic[word]) > 1:
                        continue
                except KeyError:
                    continue
            if pos_count > neg_count:
                feat["pol"]="pos"
            elif pos_count < neg_count:
                feat["pol"]="neg"
            elif pos_count == neg_count:
                feat["pol"]="neut"

            features_list.append(feat)
        return features_list


def transform_and_classify():
    bigram_vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=1 )
    transformer = TfidfVectorizer(min_df=1)
    combined_features = FeatureUnion([("univ_select", transformer) ,("grams",bigram_vectorizer),
                                        ("polarity",Pipeline([
                                            ("poli", PolarityTransformer()),
                                            ("vect", DictVectorizer())
                                        ])),
                                        ("emotion",Pipeline([
                                            ("poli", EmotionTransformer()),
                                            ("vect", DictVectorizer())
                                        ]))
                                    ])


    text_clf = Pipeline([
                         ('features', combined_features),
                         # ('select',SelectKBest(chi2,k=5000)),
                         # ('norm', Normalizer(norm='l2')),
                         ('clf', OneVsOneClassifier(LinearSVC(random_state=0, C=15)))
                         ])

    scores = cross_validation.cross_val_score(text_clf, corpus_combined, labels_combined, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def classify(filename):
    readin(filename)
    print(len(labels_combined))
    print(len(corpus_combined))
    transform_and_classify()


if __name__ == '__main__':
    start_time = time.time()

    corpus_combined = []
    labels_combined = []
    corpus = []
    labels = []
    raw_data = []
    raw_corpus = {}
    polarity_dic = {}
    feeling_dic = {}

    fill_feeling_dic()
    fill_polarity_dic(

        for filename in files:
        print("CLASSIFICATION FOR :"+filename)
        print("#####################################")
        classify(filename)

        corpus_combined = []
        labels_combined = []
        raw_data = []
        raw_corpus = {}
    print("--- %s seconds ---" % round((time.time() - start_time),3))
