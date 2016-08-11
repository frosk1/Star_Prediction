from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from source.transformer import *
from random import shuffle
from collections import OrderedDict
import os
import glob
import source.resource as res


class Corpus:

    def __init__(self, filename):
        self.corpus = []
        self.labels = []
        self.raw_data = []
        self.raw_corpus = {}
        self.numb_class = 3
        self.read_in(filename)

    def read_in(self, filename):
        with open(filename, "r") as subjtext:
            for doc in subjtext.read().split("\n"):
                self.raw_data.append(doc)
            subjtext.close()

        for field in self.raw_data:
            try:
                self.raw_corpus[field.split("\t")[0]] = field.split("\t")[1:]
            except IndexError:
                continue

        items = self.raw_corpus.items()
        shuffle(list(items))
        new_dic = OrderedDict(items)

        for item in new_dic.items():
            try:
                if self.numb_class == 3:
                    self.corpus.append(item[1][3])
                    self.labels.append(item[1][0])
                elif self.numb_class == 4:
                    self.corpus.append(item[1][3])
                    self.labels.append(item[1][1])
                else:
                    print("Use 3 class data or 4 class data!")

            except IndexError:
                continue


class Lexicon:

    def __init__(self):
        self.emotion = self.fill_emotion_dic()
        self.polarity = self.fill_polarity_dic()

    def fill_emotion_dic(self,):
        emotion_dic = {}
        for filename in glob.glob(os.path.join(res.feelings, '*.txt')):
            with open(filename) as infile:
                for word in infile.read().split("\n"):
                    try:
                        emotion_dic[word] += [filename[52:-4]]
                    except KeyError:
                        emotion_dic[word] = [filename[52:-4]]
        return emotion_dic

    def fill_polarity_dic(self,):
        polarity_dic = {}
        with open(res.neg) as negative, open(res.pos) as positive:
            for word in negative.read().split("\n"):
                try:
                    polarity_dic[word] += ["neg"]
                except KeyError:
                    polarity_dic[word] = ["neg"]

            for word in positive.read().split("\n"):
                try:
                    polarity_dic[word] += ["pos"]
                except KeyError:
                    polarity_dic[word] = ["pos"]
            negative.close()
            positive.close()
        return polarity_dic


class Classifier:

    def __init__(self, corpus, emotion, polarity):
        self.corpus = corpus.corpus
        self.labels = corpus.labels
        self.emotion = emotion
        self.polarity = polarity

    def transform_and_classify(self,):
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        combined_features = FeatureUnion([("TF-IDF", TfidfVectorizer(min_df=1)),
                                          ("Unigrams, Bigrams",CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
                                          ("Polarity Lexicon", Pipeline([
                                            ("poli", PolarityTransformer(self.polarity)),
                                            ("vect", DictVectorizer())
                                            ])),
                                          ("Emotion Lexicon", Pipeline([
                                            ("poli", EmotionTransformer(self.emotion)),
                                            ("vect", DictVectorizer())
                                            ]))
                                        ])
        print("Feature Set:")
        for feature in combined_features.transformer_list:
            print(feature[0])


        text_clf = Pipeline([
                             ('features', combined_features),
                             # ('select',SelectKBest(chi2,k=5000)),
                             # ('norm', Normalizer(norm='l2')),
                             ('clf', OneVsOneClassifier(LinearSVC(random_state=0, C=15)))
                             ])

        scores = cross_validation.cross_val_score(text_clf, self.corpus, self.labels, cv=10)

        print("\n" + "Accuracy crosval. :"+ str(scores))
        print("Mean Accuracy: %0.2f (+/- %0.2f) " % (scores.mean(), scores.std() * 2))
        print("\n")
