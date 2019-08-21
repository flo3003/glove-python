from __future__ import print_function
import argparse
import pprint
import gensim
import pandas as pd
import re
import os
import numpy as np
from glove import Glove
from glove import Corpus


isNumber = re.compile(r'\d+.*')

def norm_word(word):
    if isNumber.search(word.lower()):
    	return '---num---'
    elif re.sub(r'\W+', '', word) == '':
    	return '---punc---'
    else:
    	return word.lower()


def read_corpus(filename):

    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)

    with open(filename, 'r') as datafile:
        for line in datafile:
	    yield line.lower().translate(None, delchars).strip().split()

def read_wikipedia_corpus(filename):

    # We don't want to do a dictionary construction pass.
    corpus = gensim.corpora.WikiCorpus(filename, dictionary={})
    for text in corpus.get_texts():
        yield text


if __name__ == '__main__':

    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Create the necessary files for the database')

    parser.add_argument('--create', '-c', action='store',
                        default=None,
                        help=('The filename of the corpus to construct the co-occurrence matrix'))
    parser.add_argument('-wiki', '-w', action='store_true',
                        default=False,
                        help=('Assume the corpus input file is in the '
                              'Wikipedia dump format'))
    parser.add_argument('--load', '-l', action='store',
                        default=False,
                        help=('If true then a corpus model and a glove model will be loaded from the disk'))
    parser.add_argument('--ontology', '-o', action='store',
                        default=None,
                        help=('The filename of the ontology to construct the word-to-word semantic relationship'))
    parser.add_argument('--emb_dim', '-d', action='store',
                        default=100,
                        help=('Embedding dimensions'))
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help=('Number of parallel threads to use for training'))

    args = parser.parse_args()

    if args.load:
        print('Loading saved models..')
        corpus_model = Corpus.load('corpus.model')
        glove = Glove.load('glove.model')
        print('Saved models loaded..')
    else:
        if args.create:
            # Build the corpus dictionary and the cooccurrence matrix.
            print('Pre-processing corpus')

            if args.wiki:
                print('Using wikipedia corpus')
                get_data = read_wikipedia_corpus
                filename=args.wiki
            else:
                get_data = read_corpus
                filename=args.create

            corpus_model = Corpus()
            corpus_model.fit(get_data(args.create), window=5)
            corpus_model.save('corpus.model')

            print('Dict size: %s' % len(corpus_model.dictionary))
            print('Collocations: %s' % corpus_model.matrix.nnz)

            corpus_model.matrix.sum_duplicates()
            R=[]
            for row, col, value in zip(corpus_model.matrix.row, corpus_model.matrix.col, corpus_model.matrix.data):
                R.append((row+1, col+1, value, np.log(value)))

            RR=pd.DataFrame(R, columns=('word_a', 'word_b', 'cooccurrence', 'log_cooccurrence'))
            RR.to_csv("coo_matrix.csv", sep=',', index=False)
            print("Cooccurrence Matrix Saved")

            if not args.create:
                # Try to load a corpus from disk.
                print('Reading corpus statistics')
                corpus_model = Corpus.load('corpus.model')

                print('Dict size: %s' % len(corpus_model.dictionary))
                print('Collocations: %s' % corpus_model.matrix.nnz)

            glove = Glove(no_components=int(args.emb_dim))
            glove.fit(corpus_model.matrix, epochs=1,
                      no_threads=args.parallelism)
            glove.add_dictionary(corpus_model.dictionary)
            glove.save('glove.model')

    	# SAVE THE WORD MAPPING
    	dd=[]
    	for k, v in glove.dictionary.iteritems():
      		dd.append((v+1,k))

    	ddf=pd.DataFrame(dd, columns=('word_id', 'name'))
    	ddf.to_csv("word_mapping.csv", sep=',', index=False)
        print("Word Mapping Saved")

    if args.ontology:
    	#CREATE AND SAVE THE ONTOLOGY
        ontology_filename=args.ontology
    	lexicon = {}
    	cnt = 0
    	for line in open(ontology_filename, 'r'):
      		words = line.lower().strip().split()
      		lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]

    	d = []
    	for q in range(len(lexicon.keys())):
      		if (lexicon.keys()[q] == '---num---' or  lexicon.keys()[q] == '---punc---'):
                    continue
      		for w in range(len(lexicon.values()[q])):
                    if (lexicon.values()[q][w] == '---num---' or  lexicon.values()[q][w] == '---punc---'):
                        continue
                    if (lexicon.keys()[q] in glove.dictionary and  lexicon.values()[q][w] in glove.dictionary):
                        d.append((glove.dictionary[lexicon.keys()[q]]+1, glove.dictionary[lexicon.values()[q][w]]+1, "1"))

        new_fname=os.path.basename(ontology_filename)
    	df=pd.DataFrame(d, columns=('source_id', 'target_id', 'semantic_value'))
    	df.to_csv(new_fname[:-4]+".csv", sep=',', index=False)
        print("Lexicon Saved")
