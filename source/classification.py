"""
Classification Module
"""
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
    """
    Corpus class for data structure used in this Text classification-system.
    The data structure contains the documents and its labels, given by the
    data set of Bo Pang and Lilian Lee.

    Parameter
    ---------
    filename : String, obligatory
        Contains the relative path of the merged data set for an
        author.
    numb_class : int
        Represents the number of labels for classification. There are two
        options for a 3 class or a 4 class task.(Default=3)

    Attributes
    ----------
    corpus : array
        Contains every document from the corpus, stored as string.

    labels : array
        Contains every label for the corresponding document.

    raw_data : array
        Contains every information from the raw_corpus file as string.
        e.g. ID,Label3,Label4,Labelpro,Review

    raw_corpus : hash
        The raw data is wrapped in a hash map, for the reason of random
        selection. (The acutal data is sorted by rating scales)
        Key:review_ID Value:raw_information

    numb_class : int
        Represents the number of labels for classification. There are two
        options for a 3 class or a 4 class task.
    """
    def __init__(self, filename, numb_class=3):
        self.corpus = []
        self.labels = []
        self.raw_data = []
        self.raw_corpus = {}
        self.numb_class = numb_class
        self.read_in(filename)

    def read_in(self, filename):
        """File Reader for merged corpus file.

        Open a merged corpus file and write information into
        the data structure.

        Parameter
        ---------
        filename : string
            Contains the relative path of a merged corpus file.
        """
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
        shuffle(list(items))                            # Shuffling the items
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
                    break

            except IndexError:
                continue


class Lexicon:
    """
    Lexicon class for a data structure used in the lexicon
    feature extraction approach.

    Notes
    -----
    The methods used in this class could also be static, but we
    stick to a OOP way!

    Attributes
    ----------
    emotion : hash
        Inverted index, contains words with the corresponding emotion.
        Build with the emotion_lex dir.

    polarity : hash
        Inverted index, contains words with the corresponding polarity.
        Build with the polarity_lex dir.
    """

    def __init__(self):
        self.emotion = self.fill_emotion_dic()
        self.polarity = self.fill_polarity_dic()

    def fill_emotion_dic(self,):
        """Setter emotion lexicon.

        Walk through the emotion_lex dir and build an inverted index
        containing the words as keys and the corresponding emotions as
        values.

        Returns
        -------
        emotion_dic : hash
            Inverted index, contains words with the corresponding emotion.
            Build with the emotion_lex dir.
        """
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
        """Setter polarity lexicon.

        Walk through the polarity_lex dir and build an inverted index
        containing the words as keys and the corresponding polarity as
        values.

        Returns
        -------
        polarity_dic : hash
            Inverted index, contains words with the corresponding polarity.
            Build with the polarity_lex dir.
        """
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
    """
    Text-classification class with scikit-learn.
    For reference see: http://scikit-learn.org/stable/

    This Classifier class is based on the Corpus and Lexicon
    classes. It uses an One vs. One classification approach based
    on a SVM(Linear Kernel). For Evaluation a 10 Fold Cross-validation
    is used.

    Parameter
    ---------
    corpus_obj : Corpus, obligatory
        Contains a corpus Object with the documents and the
        corresponding labels.

    lexicon_obj : Lexicon, obligatory
        Contains a lexicon Object with the emotion and polarity
        lexicon entries.

    Attributes
    ----------
    labels : array
        Contains the labels as strings.

    corpus : array
        Contains the document as strings.

    emotion: hash
        Inverted index, contains words with the corresponding emotion.

    polarity: hash
        Inverted index, contains words with the corresponding polarity.
    """

    def __init__(self, corpus_obj, lexicon_obj):
        self.corpus = corpus_obj.corpus
        self.labels = corpus_obj.labels
        self.emotion = lexicon_obj.emotion
        self.polarity = lexicon_obj.polarity

    def transform_and_classify(self,):
        """ Feature Extraction, Classification and Evaluation method.

        This method uses a classification pipeline from scikit learn.
        There is also an option to add Feature Selection and Normalization
        to the pipeline. First the combined features are applied to the
        data set, then the classification algorithm (OnevsOne with LinearSVM).

        For reference see: http://scikit-learn.org/stable/modules/pipeline.html

        The actual output is written to standard out.
        """
        print("Items: " + str(len(self.corpus))+"\n")

        combined_features = FeatureUnion([("TF-IDF", TfidfVectorizer(min_df=1)),
                                          ("Unigrams, Bigrams", CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
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
                             # ('select',SelectKBest(chi2,k=5000)), # Feautre Selection (with chi2)
                             # ('norm', Normalizer(norm='l2')),     # Normalization (to l2 norm)
                             ('clf', OneVsOneClassifier(LinearSVC(random_state=0, C=15)))
                             ])

        # 10fold cross-validation with the text_clf pipeline and the given corpus / labels
        scores = cross_validation.cross_val_score(text_clf, self.corpus, self.labels, cv=10)

        print("\n" + "Accuracy crosval. :" + str(scores))
        print("Mean Accuracy: %0.2f (+/- %0.2f) " % (scores.mean(), scores.std() * 2))
        print("\n")
