"""
Main Module

This is the start point of the multi-class classification system.

For every corpus(author) from the scale_date dir:

    1. Define a Classifier object
    - with Corpus and Lexicon objects

    2. Use the transform_and_classify() method

Notes
-------
This mini System is build in an OOP way. Without OOP, it could be definetly
less lines of code, but we chose the more structured kind of java way.

(Never the less there is more than one class in one module, so
we guess it is also pythonic.
Like PEP 8 says:
"A style guide is about consistency.
Consistency with this style guide is important.
Consistency within a project is more important.
Consistency within one module or function is the most important.")


Options
-------
For changing the 3 class to the 4 class task, just set
the parameter 'numb_class' of the corpus object to 4.
(The default value is 3.)

For using feature selection or/and normalization,
uncomment the pipeline items in the 'transform_and_classifiy'
method of the Classifier class.
"""
import time
from source.resource import files
from source.classification import *


if __name__ == '__main__':
    start_time = time.time()

    for filename in files:
        print("10 Fold Cross-Validation: "+filename[65:])
        print("############################################")

        classifier = Classifier(Corpus(filename, numb_class=4), Lexicon())
        classifier.transform_and_classify()

    print("--- %s seconds ---" % round((time.time() - start_time), 3))
