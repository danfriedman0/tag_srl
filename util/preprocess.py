# preprocess.py
# Scripts to preprocess CoNLL data (e.g., vocabulary information)
from __future__ import print_function

import os
import sys
import xml.etree.ElementTree as ET

from collections import Counter
from util.data_loader import conll09_generator


def get_vocab_counts(vocab_type, lowercase=False):
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
            if lowercase:
                words = [word.lower() for word in words]
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
        print(fn)
        return []
    root = tree.getroot()
    rolesets = []    
    for pred in root.iter('predicate'):
        for roleset in pred.iter('roleset'):
            role_id = roleset.attrib['id']
            line = [role_id]
            for role in roleset.iter('role'):
                n = role.attrib['n']
                line.append('A' + n)
                line.append('R-A' + n)
                line.append('C-A' + n)
            rolesets.append(line)
    return rolesets
    

def make_frame_file():
    modifs = 'AM-LOC AM-DIR AM-MNR AM-EXT AM-REC AM-CAU AM-DIS AM-ADV AM-PNC AM-MOD AM-NEG AM-SLC AM-TMP AM-PRT AM-PRD R-AM-PNC R-AM-EXT R-AM-MNR R-AM-LOC R-AM-ADV R-AM-DIR R-AM-TMP R-AM-CAU C-R-AM-TMP C-AM-ADV C-AM-TMP C-AM-EXT C-AM-NEG C-AM-PNC C-AM-DIR C-AM-LOC C-AM-MNR C-AM-CAU C-AM-DIS'
    root_dir = '/Users/danfriedman/resources/CoNLL2009/LDC2012T04-CoNLL2009-Shared-Task-Part2/data/CoNLL2009-ST-English/nb_frames/'
    frame_files = os.listdir(root_dir)
    fn_out = 'data/frames.txt'
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
            


if __name__ == '__main__':
    vocab_types = ['words', 'lemmas', 'pos', 'labels', 'stags']
    for vocab_type in vocab_types:
        preprocess_vocab(vocab_type)
