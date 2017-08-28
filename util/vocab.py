# vocab.py
# A class for storing vocabulary information and encoding words as integers
from __future__ import print_function
from __future__ import division

import numpy as np
from collections import Counter

class Vocab(object):
    def __init__(self, fn, zero=None, unk=None, alpha=0.25):
        """
        fn should be a path to a vocab file in data/vocab
        The Vocab class assigns word indices in the order the words appear
          in the file
        Word indices begin at index 1.
        If zero is provided, index 0 is reserved for zero.
        If unk is provided, unk is added to vocab.
        alpha is an optional hyperparameter for word dropout.
        """
        self.zero = zero
        self.unk = unk
        self.alpha = alpha

        with open(fn, 'r') as f:
            lines = f.readlines()
        counts = [line.strip().split(' ') for line in lines]
        self.counts = {w: int(c) for w, c in counts}
        
        if self.zero is not None:
            self.idx_to_word = {}
            self.idx_to_word[0] = self.zero
            for w, _ in counts:
                if w != self.zero:
                    self.idx_to_word[len(self.idx_to_word)] = w
        else:
            self.idx_to_word = {i: w for i, (w, _) in
                                enumerate(counts, start=0)}

        if self.unk is not None:
            self.idx_to_word[len(self.idx_to_word)] = self.unk

        self.word_to_idx = {w: i for i, w in self.idx_to_word.iteritems()}
        if self.unk is not None:
            self.unk_idx = self.word_to_idx[self.unk]
        else:
            self.unk_idx = -1

        self.size = len(self.idx_to_word)


    def encode(self, word, use_dropout=False):
        """
        Encodes a word with an integer idx.
        Optionally applies word dropout
          (see Marcheggiani et al, 2017, section 3)
        """
        if word not in self.word_to_idx:
            if self.unk is None:
                raise KeyError("{} not found".format(word))
            word = self.unk
        # if use_dropout and word != self.unk:
        #     drop_prob = self.alpha / (self.counts[word] + self.alpha)
        #     if np.random.random() < drop_prob:
        #         word = self.unk
        return self.word_to_idx[word]

    def get_freqs(self, words):
        freqs = []
        for word in words:
            if word in self.counts:
                freqs.append(self.counts[word])
            else:
                freqs.append(0)
        return freqs

    def encode_sequence(self, words, use_dropout=False):
        return [self.encode(word, use_dropout) for word in words]

    def decode(self, idx):
        return self.idx_to_word[idx]

    def decode_sequence(self, idxs):
        return [self.decode(idx) for idx in idxs]

    def __contains__(self, word):
        return word in self.word_to_idx


    
def get_vocabs(stag_type='ud'):
    """
    Returns a dictionary of Vocab objects for words, parts of speech,
      predicate lemmas, and semantic role labels
    """
    vocab_types = ['words', 'pos', 'lemmas', 'labels', 'stags', 'predicates']
    vocabs = {}
    for vocab_type in vocab_types:
        if vocab_type == 'stags':
            fn = 'data/vocab/stags.{}.txt'.format(stag_type)
        else:
            fn = 'data/vocab/{}.txt'.format(vocab_type)
        if vocab_type == 'labels':
            zero = None
            unk = None
        elif vocab_type == 'lemmas' or vocab_type == 'predicates':
            zero = '_'
            unk = '<unk>'
        else:
            zero = "<zero>"
            unk = "<unk>"
        vocabs[vocab_type] = Vocab(fn, zero=zero, unk=unk)
    return vocabs
    
