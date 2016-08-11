import os
from os import listdir

base_path = os.path.dirname(__file__)

corpus = os.path.abspath(os.path.join(base_path, "..", "scale_data/merged/"))
feelings = os.path.abspath(os.path.join(base_path, "..", "emotion_lex"))
neg = os.path.abspath(os.path.join(base_path, "..", "polarity_lex/negative-words.txt"))
pos = os.path.abspath(os.path.join(base_path, "..", "polarity_lex/positive-words.txt"))

files = [os.path.join(corpus, f) for f in listdir(corpus)]
