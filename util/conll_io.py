# conll_io.py
# Classes for reading and writing data in CoNLL-09 format
import numpy as np
import cPickle as pickle
from itertools import izip

class CoNLL09_Pred_List(object):
    """
    Stores information about a predicate and its arguments.
    Attributes are:
      pred_lemma (e.g. 'elect')
      pred_idx: the index of the predicate in the sentence
      arg_seq: a list of argument labels ('A3', 'AM-TMP', '_', ...),
        one for each word in the sentence
    """
    def __init__(self, full_pred, pred_lemma, pred_idx, arg_seq):
        self.full_pred = full_pred
        self.pred_lemma = pred_lemma
        self.pred_idx = pred_idx
        self.arg_seq = arg_seq

    def __str__(self):
        pred_idx = str(self.pred_idx)
        arg_seq = ' '.join(self.arg_seq)
        return '\t'.join([self.pred_lemma, pred_idx, arg_seq])


    
class CoNLL09_Sent(object):
    def __init__(self, lines):
        """
        Takes a list of lists in CoNLL format:
          0:ID    1:FORM   2:LEMMA   3:PLEMMA    4:POS        5:PPOS
          6:FEAT  7:PFEAT  8:HEAD    9:PHEAD     10:DPREL     11:PDEPREL
          12:FILLPRED      13:PRED   14:APRED1   15:APRED2    ...
        and stores it in a more convenient format for batch processing.
        Attributes are:
          words
          pos: parts of speech
          lemmas (only for predicates)
          pred_lists: a list of CoNLL09_Pred_List objects (see above), one
            for each predicate in the sentence
          stags: supertags
          preds: a list of the predicates (or '_' for non-predicates)
        """
        self.lines = [line[:13] for line in lines]
        self.words = [self.normalize(line[1]) for line in lines]
        self.pos = [line[5] for line in lines]
        self.stags = [line[-1] for line in lines]
        self.preds = [line[-2] for line in lines]

        # If line[13] is the predicate in the form "pred_lemma.xx")
        self.lemmas = []        
        for line in lines:
            if line[12] == 'Y':
                self.lemmas.append(line[-2].split('.')[0])
            else:
                self.lemmas.append('_')
        self.plemmas = [line[3] for line in lines]
        self.predicates = [line[13] for line in lines]
        self.fill_preds = [line[12] for line in lines]
        self.fill_preds_b = [fp == 'Y' for fp in self.fill_preds]
        self.predicted_predicates = ['_' for line in lines]

        # Add a predicate info list for each predicate in the sentence
        self.pred_lists = []
        num_preds = len(lines[0]) - 14
        pred_num = 0
        for i, line in enumerate(lines):
            if line[12] == 'Y':
                full_pred = line[-2]
                pred_lemma = line[-2].split('.')[0]
                pred_idx = i

                # `arg_seq` is the semantic dependency relation of each word
                # in the to the current predicate
                arg_seq = [line[14 + pred_num] for line in lines]
                
                self.pred_lists.append(
                    CoNLL09_Pred_List(
                        full_pred, pred_lemma, pred_idx, arg_seq))
                
                pred_num += 1

        self.num_preds = len(self.pred_lists)

        # Initialize the predictions list to all underscores
        # `predictions_list` is a sent_length x number_of_predicates matrix,
        # corresponding to CoNLL columns 14+. So
        #   self.predictions_list[i][j]
        # is the predicted semantic relation of word i to predicate j.
        self.predictions_list = [['_' for _ in xrange(self.num_preds)]
                                 for _ in xrange(len(self.words))]        
        

    def normalize(self, token):
        """From Marcheggiani et al"""
        penn_tokens = {
            '-LRB-': '(',
            '-RRB-': ')',
            '-LSB-': '[',
            '-RSB-': ']',
            '-LCB-': '{',
            '-RCB-': '}' 
        }
        if token in penn_tokens:
            return penn_tokens[token]

        token = token.lower()
        try:
            int(token)
            return "<NUM>"
        except:
            pass
        try:
            float(token.replace(',', ''))
            return "<FLOAT>"
        except:
            pass
        return token
   
        
    def __str__(self):
        out = []
        for i, line in enumerate(self.lines):
            line_out = line + [self.preds[i]] + self.predictions_list[i]
            out.append('\t'.join(line_out))
        return '\n'.join(out) + '\n'


    def __len__(self):
        return len(self.words)
    

    def add_predicted_predicates(self, probs, vocab,
                                 fill_all=True, lemma_to_preds=None):
        # Add predicted predicates to self
        for i in xrange(len(self.fill_preds)):
            if fill_all or self.fill_preds[i] == 'Y':
                if (lemma_to_preds is not None and
                    self.plemmas[i] in lemma_to_preds):
                    possibilities = list(lemma_to_preds[self.plemmas[i]])
                    idxs = vocab.encode_sequence(possibilities)
                    probabilities = [probs[i][j] for j in idxs]
                    prediction = possibilities[np.argmax(probabilities)]
                    self.predicted_predicates[i] = prediction
                else:
                    pred_id = np.argmax(probs[i])
                    prediction = vocab.idx_to_word[pred_id]
                    self.predicted_predicates[i] = prediction
        return self.predicted_predicates


class CoNLL09_Sent_with_Pred(object):
    """
    This is like a CoNLL09_Sent but encoded for a specific predicate.
    It has fields for words, pos, lemmas, stags, etc., and also a pointer
    to the parent CoNLL09_Sent.
    """
    def __init__(self, sent, pred_num, pred_to_frame):
        self.words = sent.words
        self.pos = sent.pos
        self.lemmas = sent.lemmas
        self.stags = sent.stags
        self.length = len(self.words)
        self.frame = []

        # pred_num == -1 means that this is a dummy object for a sentence
        # with no predicates
        if pred_num != -1:
            pred_list = sent.pred_lists[pred_num]
            self.pred = pred_list.pred_lemma
            self.full_pred = pred_list.full_pred
            self.pred_idx = pred_list.pred_idx
            self.labels = pred_list.arg_seq
            if self.pred in pred_to_frame:
                self.frame = pred_to_frame[self.pred]
            self.count = len([l for l in self.labels if l not in self.frame])
        else:
            self.count = 0
            pred_list = []
            self.pred = None
            self.pred_idx = 0
            self.labels = []
            
        self.predictions = []
        self.pred_num = pred_num
        self.parent = sent


    def __len__(self):
        return len(self.words)


    def add_predictions(self, probs, vocab, restrict_labels=False):
        """
        probabilities is a matrix shaped (seq_length, num_labels)
          containing a probability distribution over labels for each
          word in the sequence
        add_predictions picks the most probable label for each
          word and also adds the predictions to the parent sentence
          for writing to file later
        If restrict_labels is set to True then only labels from the predicate's
          frame are allowed (note that the predicate frame information is
          not totally complete or accurate)
        """
        if self.pred_num == -1:
            return

        # Decode predictions
        raw_predictions = np.argmax(probs, axis=1)
        predictions = [vocab.decode(p) for p in raw_predictions]
        self.predictions = predictions

        # Add the predictions to the parent's predictions list
        for i, row in enumerate(self.parent.predictions_list):
            row[self.pred_num] = predictions[i]


def get_pred_to_frame(language):
    """Returns a dictionary mapping predicates to allowable frames"""
    fn_in = 'data/{}/frames.txt'.format(language)
    pred_to_frame = {}
    with open(fn_in, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if parts[0] in pred_to_frame:
                pred_to_frame[parts[0]] = list(set(pred_to_frame[parts[0]] +
                                                   parts[1:]))
            else:
                pred_to_frame[parts[0]] = parts[1:]
    return pred_to_frame


def get_lemma_to_preds(fn='data/eng/conll09/train.txt'):
    """
    Given a file (default: English training set), returns a dictionary
    mapping each lemma to the set of predicates that are associated with
    that lemma in the dataset.
    `d` is a dictionary mapping strings to sets.
    """
    d = {}
    with open(fn, 'r') as f:
        for line in f:
            if line == '\n':
                continue
            parts = line.split('\t')
            if parts[12] == 'Y':
                lemma = parts[3]
                predicate = parts[13]
                if lemma not in d:
                    d[lemma] = []
                if predicate not in d[lemma]:
                    d[lemma].append(predicate)
    return d


def _get_lemma_to_preds(fn='data/eng/frames/lemma_to_preds.pkl'):
    with open(fn, 'r') as f:
        d = pickle.load(f)
    return d
    
            
def conll09_generator(fn_txt, fn_preds, fn_stags,
                      language='eng', only_sent=False):
    """
    Generator for reading data in CoNLL format.
    Given a file object, yields CoNLL09_Sent_with_Pred objects.
    """
    pred_to_frame = get_pred_to_frame(language)
    fs = [open(fn, 'r') for fn in [fn_txt, fn_preds, fn_stags]]
    lines = []
    for line, pred, stag in izip(*fs):
        if line == '\n':
            sent = CoNLL09_Sent(lines)
            lines = []
            if only_sent:
                yield sent
                continue
            for i in xrange(sent.num_preds):
                yield CoNLL09_Sent_with_Pred(sent, i, pred_to_frame)
            if sent.num_preds == 0:
                yield CoNLL09_Sent_with_Pred(sent, -1, pred_to_frame)
        else:
            line = line.strip().split('\t')
            line.append(pred.strip())
            line.append(stag.strip())
            lines.append(line)
    [f.close() for f in fs]


def test_frames():
    fn_txt = 'data/conll09/train.txt'
    fn_preds = 'data/conll09/gold/train_predicates.txt'
    fn_stags = 'data/conll09/pred/train_stags_model1.txt'
    count = 0
    sent_count = 0
    for sent in conll09_generator(fn_txt, fn_preds, fn_stags):
        if sent.count > 0:
            # print(sent.pred)
            # print(sent.labels)
            # print(sent.frame)
            # return
            sent_count += 1
            count += sent.count

    print('{} sentences with errors, {} total errors'.format(sent_count,count))


    
def get_baseline_predictions():
    """
    This is a baseline for predicted predicates:
      - Get a dictionary from training data mapping lemmas to predicates
      - Pick randomly from the allowable predicates
    F1 is ~67 (50% of lemma instances have only one allowable predicate)
    """
    total = 0
    matches = 0
    unk = 0
    
    lemma_to_preds = get_lemma_to_preds()
    fn = 'data/eng/conll09/dev.txt'
    with open(fn, 'r') as f:
        for line in f:
            if line == '\n':
                print('')
                continue
            parts = line.split('\t')
            lemma = parts[3]
            fill_pred = parts[12] == 'Y'
            if fill_pred and lemma in lemma_to_preds:
                total += 1
                preds = lemma_to_preds[lemma]
                if len(preds) == 1:
                    matches += 1
                print(np.random.choice(preds))
            elif fill_pred:
                total += 1
                unk += 1
                print('UNKNOWN')
            else:
                print('_')
    

if __name__ == '__main__':
    get_baseline_predictions()
