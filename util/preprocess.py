# preprocess.py
# Scripts to preprocess CoNLL data (e.g., vocabulary information)
from __future__ import print_function

import os

from collections import Counter
from util.data_loader import conll09_generator


def get_vocab_counts(vocab_type, lowercase=False):
    """
    vocab_type must be "words", "lemmas", or "pos"
    returns a counter
    """
    if vocab_type == 'roles':
        return get_role_vocab_counts()
    fn_in = 'data/conll2009/CoNLL2009-ST-English-train.txt'
    counts = Counter()
    with open(fn_in, 'r') as f:
        for sent in conll09_generator(f):
            words = getattr(sent, vocab_type)
            if lowercase:
                words = [word.lower() for word in words]
            counts.update(words)
    return counts


def get_role_vocab_counts():
    fn_in = 'data/conll2009/CoNLL2009-ST-English-train.txt'
    counts = Counter()
    with open(fn_in, 'r') as f:
        for sent in conll09_generator(f):
            pred_lists = sent.pred_lists
            args = [arg for pred_list in pred_lists
                    for arg in pred_list.arg_seq]
            counts.update(args)
    return counts


def preprocess_vocab(vocab_type):
    print('{}...'.format(vocab_type))
    counts = get_vocab_counts(vocab_type, vocab_type=='words')
    if not os.path.exists('data/vocab/'):
        os.makedirs('data/vocab')
    fn_out = 'data/vocab/{}.txt'.format(vocab_type)
    with open(fn_out, 'w') as f_out:
        for word, count in counts.most_common():
            f_out.write('{} {}\n'.format(word, count))
    if vocab_type == 'words':
        filter_words()


def filter_words():
    vocab = set()
    with open('data/embeddings/sskip.100.vectors', 'r') as f:
        for line in f:
            vocab.add(line.split(' ')[0])
    fn = 'data/vocab/words.txt'
    with open(fn, 'r') as f_in:
        lines = f_in.readlines()
    with open(fn, 'w') as f_out:
        for line in lines:
            if line.split(' ')[0] in vocab:
                f_out.write(line)
    f_out.close()


if __name__ == '__main__':
    vocab_types = ['words', 'lemmas', 'pos', 'roles']
    for vocab_type in vocab_types:
        preprocess_vocab(vocab_type)
