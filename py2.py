import word2vec
from gensim.models import word2vec

sen = word2vec.LineSentence('data/develop.word')
model = word2vec.Word2Vec(sentences=sen)




