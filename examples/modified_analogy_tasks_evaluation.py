import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import csv
from os.path import expanduser
from glove import Glove, metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Evaluate a trained GloVe '
                                                  'model on an analogy task.'))
    parser.add_argument('--test', '-t', action='store',
                        required=True,
                        help='The filename of the analogy test set.')
    parser.add_argument('--model', '-m', action='store',
                        required=True,
                        help='The filename of the stored GloVe model.')
    parser.add_argument('--encode', '-e', action='store_true',
                        default=False,
                        help=('If True, words from the '
                              'evaluation set will be utf-8 encoded '
                              'before looking them up in the '
                              'model dictionary'))
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help=('Number of parallel threads to use'))

    args = parser.parse_args()

    # Load the GloVe model
    glove = Glove.load(args.model)
    filename = str(args.model)
    home = expanduser("~")

    if "framenet" in filename:
	my_lex = home+"/LexiconFALCON/database/framenet.csv"
    elif "wordnet" in filename:
	my_lex = home+"/LexiconFALCON/database/wordnet.csv"
    elif "ppdb" in filename:
	my_lex = home+"/LexiconFALCON/database/ppdb.csv"

    if args.encode:
        encode = lambda words: [x.lower().encode('utf-8') for x in words]
    else:
        encode = lambda words: [unicode(x.lower()) for x in words]


    lexicon = []
    for w in range(len(glove.word_vectors)):
         lexicon.append([])

    with open(my_lex, 'r') as csvfile:
         myfile = csv.reader(csvfile, delimiter=',')
         next(myfile)
         for row in myfile:
             lexicon[int(row[0])-1].append(int(row[1])-1)


    # Load the analogy task dataset. One example can be obtained at
    # https://word2vec.googlecode.com/svn/trunk/questions-words.txt
    sections = defaultdict(list)
    evaluation_words = [sections[section].append(encode(words)) for section, words in
                        metrics.read_analogy_file(args.test)]

    section_ranks = []
    neighbor_ranks = []

    for section, words in sections.items():
        evaluation_ids = metrics.construct_analogy_test_set(words, glove.dictionary, ignore_missing=True)

        ranks , neighbors = metrics.modified_analogy_rank_score(evaluation_ids, glove.word_vectors, lexicon,
                                           no_threads=int(args.parallelism))
        section_ranks.append(ranks)
        neighbor_ranks.append(neighbors)

        print('Section %s mean rank: %s, accuracy: %s, sum_ranks: %s' % (section, ranks.mean(),
                                                          (ranks == 0).sum() / float(len(ranks)), (glove.word_vectors.shape[0]*ranks).sum()))

        if (float((neighbors!=0).sum()) != 0):
            print('Section %s mean neighbors_per_failure: %s, sum_neighbors: %s' % (section, float(neighbors.sum())/float((neighbors!=0).sum()), np.sum(neighbors) ))
        else:
            print('Section %s mean neighbors_per_failure: %s, sum_neighbors: %s' % (section, float((neighbors!=0).sum()), np.sum(neighbors) ))
    ranks = np.hstack(section_ranks)
    neighbor_ranks = np.hstack(neighbor_ranks)

    print('Overall rank: %s, accuracy: %s' % (ranks.mean(),
                                              (ranks == 0).sum() / float(len(ranks))))

    print('Overall neighbors_rank: %s, neighbors_accuracy: %s' % (neighbors.mean(), np.sum(neighbors)))
