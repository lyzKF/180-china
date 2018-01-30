#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-16 16:11
# * Last modified : 2018-01-16 16:11
# * Filename      : pyLDAvis.py
# * Description   : 
# *********************************************************

import numpy as np
import logging
import json
import pyLDAvis.gensim

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
import sys

topic_number = sys.argv[1]
#
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
#logging.debug("test")

texts = [['human', 'interface', 'computer'],
         ['survey', 'user', 'computer', 'system', 'response', 'time'],
         ['eps', 'user', 'interface', 'system'],
         ['system', 'human', 'system', 'eps'],
         ['user', 'response', 'time'],
         ['trees'],
         ['graph', 'trees'],
         ['graph', 'minors', 'trees'],
         ['graph', 'minors', 'survey']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# coloring words
# bow_water = ['bank','water','river', 'tree']
# color_words(goodLdaModel, bow_water)

# pyLDAvis
goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=100, num_topics=topic_number)
badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=10, num_topics=topic_number)

vis_good = pyLDAvis.gensim.prepare(goodLdaModel, corpus, dictionary)

vis_bad = pyLDAvis.gensim.prepare(badLdaModel, corpus, dictionary)

pyLDAvis.show(vis_good)  
# pyLDAvis.show(vis_bad)  

#pyLDAvis.save_html(vis_good, 'lda_good.html')  

#pyLDAvis.save_html(vis_bad, 'lda_bad.html')  

# Quantitative approach
goodcm = CoherenceModel(model=goodLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')

badcm = CoherenceModel(model=badLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')

print (goodcm.get_coherence())
print (badcm.get_coherence())
