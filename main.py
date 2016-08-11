import time
from source.resource import files
from source.classification import *


def classification(file_path):
    reviews = Corpus(file_path)
    lexicon = Lexicon()
    print("Items: " + str(len(reviews.corpus))+"\n")
    classifier = Classifier(reviews, lexicon.emotion, lexicon.polarity)
    classifier.transform_and_classify()

if __name__ == '__main__':
    start_time = time.time()

    for filename in files:
        print("10 Fold Cross-Validation: "+filename[65:])
        print("############################################")
        classification(filename)

    print("--- %s seconds ---" % round((time.time() - start_time), 3))
