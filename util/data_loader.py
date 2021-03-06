# Functions for reading CoNLL-2009 data
from __future__ import print_function
from __future__ import division

import numpy as np

from util.conll_io import CoNLL09_Sent, CoNLL09_Sent_with_Pred
from util.conll_io import conll09_generator, get_lemma_to_preds



def make_batch_labels_masks(sents, vocab):
    masks = np.zeros((len(sents), vocab.size), dtype=np.float32)
    for i, sent in enumerate(sents):
        if len(sent.frame) == 0:
            masks[i, :] = np.ones(vocab.size, dtype=np.float32)
        for label in sent.frame:
            if label in vocab:
                masks[i, vocab.encode(label)] = 1.0
    return masks


def make_batch_field_sequence(sents, field, seq_length, vocab,
                              use_dropout=False):
    data = np.zeros((len(sents), seq_length), dtype=np.int32)
    for i, sent in enumerate(sents):
        words = getattr(sent, field)
        data[i, :len(words)] = vocab.encode_sequence(words, use_dropout)
    return data


def make_batch_field_single(sents, field, vocab=None):
    data = np.zeros((len(sents),), dtype=np.int32)
    for i, sent in enumerate(sents):
        val = getattr(sent, field)
        if vocab:
            val = vocab.encode(val)
        data[i] = val
    return data


def make_batch_freqs(sents, seq_length, vocab):
    data = np.zeros((len(sents), seq_length), dtype=np.int32)
    for i, sent in enumerate(sents):
        data[i, :len(sent)] = vocab.get_freqs(sent.words)
    return data
    
            
def make_batch(sents, vocabs, train):
    """
    A batch contains seven things:
      words_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
      freqs_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
      pos_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
      lemmas_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
      preds_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
      preds_idx_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
      labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
      labels_mask_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, vocabs['labels'].size))    
      stags_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
    in that order.
    There is one predicate for each sentence.
      preds_placeholder is a list of encoded predicates
      preds_idx_placeholder is the index of each predicate in the
        corresponding sentence
    """
    seq_length = max(len(sent) for sent in sents)
    words = make_batch_field_sequence(sents, 'words',
                                      seq_length, vocabs['words'],
                                      use_dropout=False)
    freqs = make_batch_freqs(sents, seq_length, vocabs['words'])
    pos = make_batch_field_sequence(sents, 'pos',
                                    seq_length, vocabs['pos'])
    lemmas = make_batch_field_sequence(sents, 'lemmas',
                                       seq_length, vocabs['lemmas'])
    preds = make_batch_field_single(sents, 'pred', vocab=vocabs['lemmas'])
    preds_idx = make_batch_field_single(sents, 'pred_idx')
    labels = make_batch_field_sequence(sents, 'labels',
                                       seq_length, vocabs['labels'])
    labels_mask_placeholder = make_batch_labels_masks(sents, vocabs['labels'])
    stags = make_batch_field_sequence(sents, 'stags',
                                      seq_length, vocabs['stags'])
    seq_lengths = make_batch_field_single(sents, 'length')
    return (words, freqs, pos, lemmas, preds, preds_idx,
            labels, labels_mask_placeholder, stags, seq_lengths)

    
def batch_producer(batch_size, vocabs, fn_txt, fn_preds, fn_stags,
                   language, train=True):
    """
    vocabs should be a dictionary of Vocab objects keyed "words", "pos", etc.
    See `make_batch` for details about what's in a batch.
    Returns the batch and also the corresponding list of sentence objects
      (useful for evaluation)
    """
    all_sents = [s for s in conll09_generator(
        fn_txt, fn_preds, fn_stags, language)]
    # if train:
    #     all_sents = sorted(all_sents, key=lambda s: -len(s))
    sents = []
    for sent in all_sents: #conll09_generator(fn_txt, fn_preds, fn_stags):
        # During training, skip sentences without predicates
        # (but include them during testing to make it simpler to write
        # predictions to file)
        if sent.pred is not None or not train:
            sents.append(sent)
        # sents.append(sent)
        if len(sents) == batch_size:
            yield sents, make_batch(sents, vocabs, train)
            sents = []
    # Fill out the last batch if the data doesn't evenly divide
    if len(sents) > 0 and len(sents) < batch_size:
        sents += [sents[0] for _ in xrange(batch_size - len(sents))]
        yield sents, make_batch(sents, vocabs, train)

def make_fill_preds_batch(sents, seq_length):
    batch = np.zeros((len(sents), seq_length), dtype=np.int32)
    for i, sent in enumerate(sents):
        for j, fp in enumerate(sent.fill_preds):
            if fp == 'Y':
                batch[i, j] = 1
    return batch
        

def make_disamb_batch(sents, vocabs, train):
    seq_length = max(len(sent) for sent in sents)
    words = make_batch_field_sequence(sents, 'words',
                                      seq_length, vocabs['words'],
                                      use_dropout=False)
    pos = make_batch_field_sequence(sents, 'pos',
                                    seq_length, vocabs['pos'])
    lemmas = make_batch_field_sequence(sents, 'plemmas',
                                       seq_length, vocabs['plemmas'])
    labels = make_batch_field_sequence(sents, 'predicates',
                                       seq_length, vocabs['predicates'])
    stags = make_batch_field_sequence(sents, 'stags',
                                      seq_length, vocabs['stags'])
    fill_preds = make_fill_preds_batch(sents, seq_length)
    return words, pos, lemmas, labels, stags, fill_preds
    


def disamb_batch_producer(batch_size, vocabs, fn_txt, fn_stags,
                          language, train=True):
    all_sents = [s for s in conll09_generator(
        fn_txt, fn_stags, fn_stags, language, only_sent=True)]
    if train:
        all_sents = sorted(all_sents, key=lambda s: -len(s))
    sents = []
    for sent in all_sents:
        sents.append(sent)
        if len(sents) == batch_size:
            yield sents, make_disamb_batch(sents, vocabs, train)
            sents = []
    # Fill out the last batch if the data doesn't evenly divide
    if len(sents) > 0 and len(sents) < batch_size:
        sents += [all_sents[i] for i in xrange(batch_size - len(sents))]
        yield sents, make_disamb_batch(sents, vocabs, train)
            
