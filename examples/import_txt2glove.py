from glove import Glove
from glove import Corpus
import pandas as pd
import numpy as np
import os
import sys
from os.path import expanduser
import gensim

home = expanduser("~")
glove = Glove.load(home+'/glove-python/examples/imdb_glove.model')

fname = sys.argv[1]

with open(fname) as fp:
     data = [list(map(float, line.strip().split())) for line in fp]

vectors = []
for i in range(len(data)):
     vectors.append(np.asarray(data[i]))

vectors=np.asarray(vectors)

glove_imported = Glove(no_components=100, learning_rate=0.05)

glove_imported.word_vectors=vectors

glove_imported.dictionary=glove.dictionary

glove_imported.inverse_dictionary = glove.inverse_dictionary

new_fname=os.path.basename(fname)

glove_imported.save(home+'/glove-python/examples/imported_'+new_fname[:-4]+'.model')

