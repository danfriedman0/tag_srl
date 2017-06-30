# Functions for loading data from files
from __future__ import print_function

import numpy as np


def get_word_embeddings(idx_to_word, fn='data/embeddings/sskip.100.vectors'):
    """
    Load word embeddings from a file and store them in an embedding matrix
    indexed by the indices in idx_to_word.
    """
    word_to_vec = {}
    with open(fn, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            word_to_vec[word] = vec
    num_words = max(idx_to_word.keys()) + 1
    embed_size = len(word_to_vec.values()[0])
    embeddings = np.zeros((num_words, embed_size), dtype=np.float32)
    for idx, word in idx_to_word.iteritems():
        embeddings[idx, :] = word_to_vec[word]
    return embeddings

