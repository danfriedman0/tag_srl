# Code to extract the depencency tree supertags of Ouchi et al. 2014
# from CoNLL-format data
from __future__ import print_function

from collections import defaultdict
import sys
import os


def conll_line_to_dict(line):
    '''
    Stores a CoNLL word in named fields for more transparent access.
    `line` should be a string.
    ConLL-09:
        0:ID    1:FORM   2:LEMMA   3:PLEMMA    4:POS        5:PPOS
        6:FEAT  7:PFEAT  8:HEAD    9:PHEAD     10:DEPREL     11:PDEPREL
        12:FILLPRED      13:PRED   14:APRED1   15:APRED2    ...
    '''
    fields = [(0, 'idx', int), (4, 'pos', str),
              (8, 'head', int), (10, 'deprel', str)]
    # fields = [(0, 'idx', int), (3, 'pos', str),
    #           (6, 'head', int), (7, 'deprel', str)]
    
    parts = line.split('\t')
    d = {}
    for idx, field, dtype in fields:
        d[field] = dtype(parts[idx])
    return d


def get_model0_stag(word, dep_table):
    # This is the model from Foth et al. (2006).
    # The template is:
    #   [left_dependents]+DEPREL/DIRECTION+[right_dependents]
    # left_dependents and right_dependents are comma-separated lists of
    # the deprels of dependents.
    # DIRECTION is the direction (L or R) to the head, or N if it's the root.
    head = word['deprel']
    if word['deprel'] == 'ROOT':
        head += '/N'
    else:
        head += '/R' if word['head'] > word['idx'] else '/L'
    l_deps = [w['deprel'] for w in dep_table[word['idx']]['L']]
    r_deps = [w['deprel'] for w in dep_table[word['idx']]['R']]
    stag = '+'.join([','.join(l_deps), head, ','.join(r_deps)])
    return stag


def get_model1_stag(word, dep_table):
    # In model 1, the template is
    #   DEPREL/DIRECTION+L_R,
    # DIRECTION is direction (L or R) to head.
    # The L and R after the + sign are binary features indicating whether or
    # not the word has any dependents to the left or right.
    head = word['deprel']
    if word['deprel'] != 'ROOT':
        head += '/R' if word['head'] > word['idx'] else '/L'
    deps = []
    if len(dep_table[word['idx']]['L']) > 0:
        deps.append('L')
    if len(dep_table[word['idx']]['R']) > 0:
        deps.append('R')
    if len(deps) > 0:
        stag = head + '+' + '_'.join(deps)
    else:
        stag = head
    return stag


def get_model2_stag(word, dep_table):
    # Model 2 is the same as model 1, only verbs are additionally labeled
    # with their obligatory arguments (any argument labeled SBJ, OBJ, PRD,
    # or VC). If a verb has only non-obligatory arguments, the supertag
    # just indicates the direction, like in model1.
    
    arg_list = ['SBJ', 'OBJ', 'PRD', 'VC']
    # arg_list = ['nsubj', 'csubj', 'dobj', 'iobj', 'ccomp', 'xcomp']
    
    if word['pos'][0] != 'V':
        return get_model1_stag(word, dep_table)

    head = word['deprel']
    if word['deprel'] != 'ROOT':
        head += '/R' if word['head'] > word['idx'] else '/L'
        

    l_deps = [w['deprel'] + '/L' for w in dep_table[word['idx']]['L']
              if w['deprel'] in arg_list]
    r_deps = [w['deprel'] + '/R' for w in dep_table[word['idx']]['R']
              if w['deprel'] in arg_list]
    
    if len(dep_table[word['idx']]['L']) > 0 and len(l_deps) == 0:
        l_deps.append('L')
    if len(dep_table[word['idx']]['R']) > 0 and len(r_deps) == 0:
        r_deps.append('R')
        
    deps = l_deps + r_deps
    if len(deps) > 0:
        stag = head + '+' + '_'.join(deps)
    else:
        stag = head
    return stag



def get_stags_from_sent(words, model_number):
    '''
    Given a conll sentence, returns a list of supertags.
    `words` should be a list of conll word dicts.
    `model_number` should be 0, 1, or 2:
      0 is the model from Foth et al (2006);
      1 and 2 are from Ouchi et al.
    '''

    # Build an adjacency matrix for left and right dependencies
    dep_table = defaultdict(lambda: defaultdict(list))
    for i, word in enumerate(words):
        if words[word['head'] - 1]['idx'] > word['idx']:
            dep_table[word['head']]['L'].append(word)
        else:
            dep_table[word['head']]['R'].append(word)

    # Get the stags according to the model number
    get_stag_fns = [get_model0_stag, get_model1_stag, get_model2_stag]
    get_stag = get_stag_fns[model_number]
    stags = [get_stag(word, dep_table) for word in words]

    return stags


def extract_stags(fn_in, fn_out, model_number, stag_to_name):
    """
    fn_in: CoNLL file to tag
    fn_out: file to write to
    model_number: 0, 1, or 2
    stag_to_name: a dictionary mapping stag strings to names (s0, s1, ...);
      initially pass an empty dictionary.
    returns the updated stag_to_name
    """
    f_out = open(fn_out, 'w')
    with open(fn_in, 'r') as f_in:
        lines = []
        for i, line in enumerate(f_in):
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
            else:
                words = [conll_line_to_dict(l) for l in lines]
                stags = get_stags_from_sent(words, model_number)
                for stag in stags:
                    if stag not in stag_to_name:
                        stag_to_name[stag] = 's{}'.format(len(stag_to_name))
                for line, stag in zip(lines, stags):
                    # f_out.write('\t'.join([line, stag_to_name[stag]]))
                    f_out.write(stag_to_name[stag])
                    f_out.write('\n')
                f_out.write('\n')
                lines = []
    f_out.close()
    return stag_to_name
    

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python extract.py model_number (file | directory) ...')
        exit

    model = int(sys.argv[1])
    if os.path.isdir(sys.argv[2]):
        dir_name = sys.argv[2]
        files = os.listdir(dir_name)
        fn_ins = [os.path.join(dir_name, f) for f in files]
    else:
        fn_ins = [sys.argv[2]]
    stag_to_name = {}
    
    for fn_in in fn_ins:
        if not os.path.isfile(fn_in):
            continue
        print(fn_in)
        # Not ideal, but get the path to match what srl.py expects...
        path = fn_in.split('/')
        data_split = path[-1].split('.')[0]
        stag_fn = data_split + '_stags_model{}.txt'.format(model)
        fn_out = '/'.join(path[:-1] + [stag_fn])
        stag_to_name = extract_stags(fn_in, fn_out, model, stag_to_name)
        
    print('Extracted {} stags'.format(len(stag_to_name)))
    fn_stags = 'stags.model{}.txt'.format(model)
    if os.path.isdir(sys.argv[2]):
        fn_stags = os.path.join(sys.argv[2], fn_stags)
    stags = sorted(stag_to_name.iteritems(), key=lambda x: int(x[1][1:]))
    with open(fn_stags, 'w') as f:
        for stag, name in stags:
            f.write(name + ' ' + stag + '\n')
            
