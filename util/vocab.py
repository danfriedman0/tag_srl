# vocab.py
# A class for storing vocabulary information and encoding words as integers
from __future__ import print_function

from collections import Counter

class Vocab(object):
    def __init__(self, fn, zero=None, unk=None):
        """
        fn should be a path to a vocab file in data/vocab
        The Vocab class assigns word indices in the order the words appear
          in the file
        Word indices begin at index 1.
        If zero is provided, index 0 is reserved for zero.
        If unk is provided, unk is added to vocab.
        """
        self.zero = zero
        self.unk = unk
        
        with open(fn, 'r') as f:
            lines = f.readlines()
        counts = [line.strip().split(' ') for line in lines]
        self.counts = {w: int(c) for w, c in counts}

        if self.zero is not None:
            self.idx_to_word = {i: w for i, (w, _) in
                                enumerate(counts, start=1)}
            self.idx_to_word[0] = self.zero
        else:
            self.idx_to_word = {i: w for i, (w, _) in
                                enumerate(counts, start=0)}

        if self.unk is not None:
            self.idx_to_word[len(self.idx_to_word)] = self.unk

        self.word_to_idx = {w: i for i, w in self.idx_to_word.iteritems()}

        self.size = len(self.idx_to_word)


    def encode(self, word):
        if word not in self.word_to_idx:
            if self.unk is None:
                raise KeyError("{} not found".format(word))
            word = self.unk
        return self.word_to_idx[word]

    def encode_sequence(self, words):
        return [self.encode(word) for word in words]

    def decode(self, idx):
        return self.idx_to_word[idx]

    def __contains__(self, word):
        return word in self.word_to_idx

    
def get_vocabs():
    """
    Returns a dictionary of Vocab objects for words, parts of speech,
      predicate lemmas, and semantic role labels
    """
    vocab_types = ['words', 'pos', 'lemmas', 'labels']
    vocabs = {}
    for vocab_type in vocab_types:
        fn = 'data/vocab/{}.txt'.format(vocab_type)
        if vocab_type == 'labels':
            zero = None
            unk = None
        else:
            zero = "<zero>"
            unk = "<unk>"
        vocabs[vocab_type] = Vocab(fn, zero=zero, unk=unk)
    return vocabs
    
