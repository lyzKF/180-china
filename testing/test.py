import logging
import engine_module_wordcloud


logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s%(thread)d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%Y %b %d %H:%M:%S',
        filename='log_test'
        )

from gensim.models import Word2Vec
wv_model_local = Word2Vec.load("./package_nlp/wv/word2vec_wx")
print wv_model_local

engine_module_wordcloud._calculate_sim()
