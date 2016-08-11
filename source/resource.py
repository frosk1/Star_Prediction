"""
Resource Module

Contains the relative paths of the lexicon and corpus dirs.
On every run the lexicons and corpora are build up once and
written to the ram. There is no serialization for lexicon and
corpus objects in the classification system. (Cause there is no
need for runtime improvement)
"""
import os
from os import listdir

base_path = os.path.dirname(__file__)

corpus = os.path.abspath(os.path.join(base_path, "..", "scale_data/merged/"))
feelings = os.path.abspath(os.path.join(base_path, "..", "emotion_lex"))
neg = os.path.abspath(os.path.join(base_path, "..", "polarity_lex/negative-words.txt"))
pos = os.path.abspath(os.path.join(base_path, "..", "polarity_lex/positive-words.txt"))

files = [os.path.join(corpus, f) for f in listdir(corpus)]
