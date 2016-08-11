Sentiment Analysis Task (Rating Inference Problem)
==========================================
 
Bitbucket: https://bitbucket.org/IMS_CREW/star_prediction

 
About
=====

As part of the Sentiment Analysis class at the IMS, this repo contains a
multi-class classification system. It addresses the rating inference problem
as solved in the paper of Bo Pang and Lilian Lee (2005): "Seeing stars: Exploiting 
class relationships for sentiment categorization with respect to rating scales".
This project can be seen as an extension of the methods used in the paper by
Bo Pang and Lilian Lee.

Class Relations
===============
The acutal rating scales of the authors are mapped to:

- 3 class task: 0,1,2 (2 is the best rating)
- 4 class task: 0,1,2,3 (3 is the best rating)

Ressource 
=============
The scale data set in this repo is published by Bo Pang and Lillian Lee.
Scale dataset v1.0:
https://www.cs.cornell.edu/people/pabo/movie-review-data/

- movie reviews containing only subjective sentences
- 4 different authors (4 different corpora)

The Polarity Lexicon is given by:

- Hu and Liu (2004): "Mining and summarizing customer reviews." 

The Emotion Lexicon is given by:

- National Research Council Canada (NRC)
http://www.purl.com/net/lexicons 

Evaluation
==========
10 Fold Crossvalidation on every authors reviews given by the
data set.

Classification
===================
- One vs. One Approach
- SVM (Linear Kernel)

Features
========
- Unigrams
- Bigrams
- Tf_idf
- Polarity Lexicon
- Emotion Lexicon

Requirements-Python and Modules
===============================
- Python 3.5
- scikit-learn 0.17.1