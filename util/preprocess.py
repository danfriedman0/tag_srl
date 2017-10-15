# preprocess.py
# Scripts to preprocess CoNLL data (e.g., vocabulary information)
from __future__ import print_function

import os
import sys
import xml.etree.ElementTree as ET

from collections import Counter
from util.data_loader import conll09_generator
from subprocess import check_output


def get_vocab_counts(vocab_type, stag_type='ud'):
    """
    vocab_type must be "words", "lemmas", "pos", or "stags"
    returns a counter
    """
    if vocab_type == 'labels':
        return get_label_vocab_counts()
    fn_txt = 'data/eng/conll09/train.txt'
    fn_preds = 'data/eng/conll09/pred/train_predicates.txt'
    fn_stags = 'data/eng/conll09/pred/train_stags_model1.txt'
    counts = Counter()
    for sent in conll09_generator(fn_txt, fn_preds, fn_stags,
                                  only_sent=True):
        words = getattr(sent, vocab_type)
        counts.update(words)
    return counts


def get_label_vocab_counts():
    fn_in = 'data/eng/conll09/train.txt'
    counts = Counter()
    with open(fn_in, 'r') as f:
        for sent in conll09_generator(f):
            pred_lists = sent.parent.pred_lists
            args = [arg for pred_list in pred_lists
                    for arg in pred_list.arg_seq]
            counts.update(args)
    return counts


def preprocess_vocab(vocab_type, stag_type='ud'):
    print('Getting vocab for {}...'.format(vocab_type))
    counts = get_vocab_counts(vocab_type, stag_type)
    if not os.path.exists('data/eng/vocab/'):
        os.makedirs('data/eng/vocab/')
    if vocab_type == 'stags':
        fn_out = 'data/vocab/{}.{}.txt'.format(vocab_type, stag_type)
    fn_out = 'data/eng/vocab/{}.txt'.format(vocab_type)
    with open(fn_out, 'w') as f_out:
        for word, count in counts.most_common():
            f_out.write('{} {}\n'.format(word, count))


def get_stag_vocab_counts(stag_type='model1'):
    fn = 'data/eng/stags/{}.txt'.format(stag_type)
    with open(fn, 'r') as f:
        counts = Counter(f.read().split('\n'))
    fn_out = 'data/vocab/stags.{}.txt'.format(stag_type)
    with open(fn_out, 'w') as f_out:
        for word, count in counts.most_common():
            f_out.write('{} {}\n'.format(word, count))
            

            
            
if __name__ == '__main__':
    # vocab_types = ['words', 'lemmas', 'pos', 'labels', 'stags']
    vocab_types = ['lemmas']
    for vocab_type in vocab_types:
        preprocess_vocab(vocab_type)

    # fns = ['train', 'dev', 'test', 'ood']
    # for fn in fns:
    #     print(fn + ':')
        
    #     print('Getting stags...')        
    #     fn_pred = 'data/conll09/pred/{}.tag'.format(fn)
    #     stag_sents = get_stags(fn_pred)

    #     print('Writing to file...')
    #     fn_in = 'data/conll09/gold/{}.txt'.format(fn)
    #     fn_out = 'data/conll09/gold/{}.tag'.format(fn)
    #     add_stags(fn_in, fn_out, stag_sents)

    # get_stag_vocab_counts('model2')
    preprocess_vocab('words')
    
