from sklearn.base import BaseEstimator, TransformerMixin
import operator


class PolarityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, polarity_dic):
        self.polarity_dic = polarity_dic

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
                    if self.polarity_dic[word][0] == "neg":
                        neg_count += 1
                    elif self.polarity_dic[word][0] == "pos":
                        pos_count += 1
                    elif len(self.polarity_dic[word]) > 1:
                        continue
                except KeyError:
                    continue
            if pos_count > neg_count:
                feat["pol"] = "pos"
            elif pos_count < neg_count:
                feat["pol"] = "neg"
            elif pos_count == neg_count:
                feat["pol"] = "neut"

            features_list.append(feat)
        return features_list


class EmotionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, emotion_dic):
        self.emotion_dic = emotion_dic

    def fit(self, x, y=None):
        return self

    def transform(self, corpus_combined):
        features_list = []

        for doc in corpus_combined:
            feat = {
                "anger": 0,
                "disgust": 0,
                "fear": 0,
                "joy": 0,
                "sadness": 0,
                "surprise": 0
            }
            feat2 = {}

            for word in doc.split():
                try:
                    for feeling in self.emotion_dic[word]:
                        if feeling == "anger":
                            feat["anger"] += 1
                        elif feeling == "disgust":
                            feat["disgust"] += 1
                        elif feeling == "fear":
                            feat["fear"] += 1
                        elif feeling == "joy":
                            feat["joy"] += 1
                        elif feeling == "sadness":
                            feat["sadness"] += 1
                        elif feeling == "surprise":
                            feat["surprise"] += 1
                except KeyError:
                    continue

            maxi = max(feat.items(), key=operator.itemgetter(1))[0]
            feat2["max"] = maxi
            features_list.append(feat2)

        return features_list
