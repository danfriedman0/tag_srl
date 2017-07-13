# preprocess.py
# Scripts to preprocess CoNLL data (e.g., vocabulary information)
from __future__ import print_function

import os
import sys
import xml.etree.ElementTree as ET

from collections import Counter
from util.data_loader import conll09_generator


def get_vocab_counts(vocab_type):
    """
    vocab_type must be "words", "lemmas", or "pos"
    returns a counter
    """
    if vocab_type == 'labels':
        return get_label_vocab_counts()
    fn_in = 'data/conll09/pred/train.tag'
    counts = Counter()
    with open(fn_in, 'r') as f:
        for sent in conll09_generator(f):
            words = getattr(sent, vocab_type)
            counts.update(words)
    return counts


def get_label_vocab_counts():
    fn_in = 'data/conll09/pred/train.tag'
    counts = Counter()
    with open(fn_in, 'r') as f:
        for sent in conll09_generator(f):
            pred_lists = sent.parent.pred_lists
            args = [arg for pred_list in pred_lists
                    for arg in pred_list.arg_seq]
            counts.update(args)
    return counts


def preprocess_vocab(vocab_type):
    print('Getting vocab for {}...'.format(vocab_type))
    counts = get_vocab_counts(vocab_type)
    if not os.path.exists('data/vocab/'):
        os.makedirs('data/vocab')
    fn_out = 'data/vocab/{}.txt'.format(vocab_type)
    with open(fn_out, 'w') as f_out:
        for word, count in counts.most_common():
            f_out.write('{} {}\n'.format(word, count))
    # if vocab_type == 'words':
    #     filter_words()


def filter_words():
    print('Filtering words...')
    vocab = set()
    with open('data/embeddings/sskip.100.vectors', 'r') as f:
        for line in f:
            vocab.add(line.split(' ')[0])
    fn = 'data/vocab/words.txt'
    with open(fn, 'r') as f_in:
        lines = f_in.readlines()
    with open(fn, 'w') as f_out:
        i = 0
        for line in lines:
            if line.split(' ')[0] in vocab:
                f_out.write(line)
                i += 1
    print('{}/{} words in sskip.100.vectors'.format(i, len(lines)))
    f_out.close()


def get_rolesets(fn):
    try:
        tree = ET.parse(fn)
    except:
        return []
    root = tree.getroot()
    d = {}
    for pred in root.iter('predicate'):
        for roleset in pred.iter('roleset'):
            lemma = roleset.attrib['id'].split('.')[0]
            if lemma not in d:
                d[lemma] = set()
            for role in roleset.iter('role'):
                n = role.attrib['n']
                d[lemma].add('A' + n)
                d[lemma].add('R-A' + n)
                d[lemma].add('C-A' + n)
    rolesets = []
    for lemma, roles in d.iteritems():
        rolesets.append([lemma] + list(roles))
    return rolesets
    

def make_frame_file(frame_type):
    """frame_type should be 'p' for propbank or 'n' for nombank"""
    modifs = 'AM-LOC AM-DIR AM-MNR AM-EXT AM-REC AM-CAU AM-DIS AM-ADV AM-PNC AM-MOD AM-NEG AM-SLC AM-TMP AM-PRT AM-PRD R-AM-PNC R-AM-EXT R-AM-MNR R-AM-LOC R-AM-ADV R-AM-DIR R-AM-TMP R-AM-CAU C-R-AM-TMP C-AM-ADV C-AM-TMP C-AM-EXT C-AM-NEG C-AM-PNC C-AM-DIR C-AM-LOC C-AM-MNR C-AM-CAU C-AM-DIS'
    root_dir = '/Users/danfriedman/resources/CoNLL2009/LDC2012T04-CoNLL2009-Shared-Task-Part2/data/CoNLL2009-ST-English/{}b_frames/'.format(frame_type)
    frame_files = os.listdir(root_dir)
    fn_out = 'data/{}b_frames.txt'.format(frame_type)
    with open(fn_out, 'a') as f:
        sys.stdout.write('[' + ' ' * 77 + ']')
        tick = len(frame_files) // 77
        for i, fn in enumerate(frame_files):
            path = os.path.join(root_dir, fn)
            rolesets = get_rolesets(path)
            for roleset in rolesets:
                line = ' '.join(roleset)
                f.write(line + ' ' + modifs + '\n')
            if i % tick == 0:
                num_ticks = i // tick
                bar = '[' + '='*num_ticks + ' '*(77-num_ticks) + ']'
                sys.stdout.write('\r')
                sys.stdout.write(bar)
                sys.stdout.flush()


def get_stags(fn):
    stag_sents = []
    with open(fn, 'r') as f:
        for sent in conll09_generator(f, only_sent=True):
            stag_sents.append(sent.stags)
    return stag_sents

def add_stags(fn_in, fn_out, stag_sents):
    f_out = open(fn_out, 'w')
    sent_idx = 0
    line_idx = 0
    with open(fn_in, 'r') as f:
        for line in f:
            if line == '\n':
                sent_idx += 1
                line_idx = 0
                f_out.write('\n')
                continue
            parts = line.strip().split('\t')
            parts.append(stag_sents[sent_idx][line_idx])
            f_out.write('\t'.join(parts) + '\n')
            line_idx += 1
    f_out.close()

    
if __name__ == '__main__':
    # vocab_types = ['words', 'lemmas', 'pos', 'labels', 'stags']
    # for vocab_type in vocab_types:
    #     preprocess_vocab(vocab_type)

    fns = ['train', 'dev', 'test', 'ood']
    for fn in fns:
        print(fn + ':')
        
        print('Getting stags...')        
        fn_pred = 'data/conll09/pred/{}.tag'.format(fn)
        stag_sents = get_stags(fn_pred)

        print('Writing to file...')
        fn_in = 'data/conll09/gold/{}.txt'.format(fn)
        fn_out = 'data/conll09/gold/{}.tag'.format(fn)
        add_stags(fn_in, fn_out, stag_sents)
    
