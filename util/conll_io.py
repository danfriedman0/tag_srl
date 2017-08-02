# conll_io.py
# Classes for reading and writing data in CoNLL-09 format
import numpy as np
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


class CoNLL09_Sent_with_Pred(object):
    """
    This is like a CoNLL09_Sent but encoded for a specific predicate.
    It has fields for words, pos, lemmas, stags, etc., and also a pointer
    to the parent CoNLL09_Sent.
    """
    def __init__(self, sent, pred_num):
        self.words = sent.words
        self.pos = sent.pos
        self.lemmas = sent.lemmas
        self.stags = sent.stags
        self.length = len(self.words)

        # pred_num == -1 means that this is a dummy object for a sentence
        # with no predicates
        if pred_num != -1:
            pred_list = sent.pred_lists[pred_num]
            self.pred = pred_list.pred_lemma
            self.full_pred = pred_list.full_pred
            self.pred_idx = pred_list.pred_idx
            self.labels = pred_list.arg_seq
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


def get_pred_to_frame(fn_in='data/frames.txt'):
    """Returns a dictionary mapping predicates to allowable frames"""
    pred_to_frame = {}
    with open(fn_in, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            pred_to_frame[parts[0]] = parts[1:]
    return pred_to_frame

            
def _conll09_generator(f, only_sent=False):
    """
    Generator for reading data in CoNLL format.
    Given a file object, yields CoNLL09_Sent_with_Pred objects.
    """
    pred_to_frame = get_pred_to_frame()
    lines = []
    for line in f:
        if line == '\n':
            sent = CoNLL09_Sent(lines)
            lines = []            
            if only_sent:
                yield sent
            else:
                for i in xrange(sent.num_preds):
                    yield CoNLL09_Sent_with_Pred(sent, i, pred_to_frame)
                if sent.num_preds == 0:
                    yield CoNLL09_Sent_with_Pred(sent, -1, pred_to_frame)
        else:
            lines.append(line.strip().split('\t'))

            
def conll09_generator(fn_txt, fn_preds, fn_stags):
    """
    Generator for reading data in CoNLL format.
    Given a file object, yields CoNLL09_Sent_with_Pred objects.
    """
    fs = [open(fn, 'r') for fn in [fn_txt, fn_preds, fn_stags]]
    lines = []
    for line, pred, stag in izip(*fs):
        if line == '\n':
            sent = CoNLL09_Sent(lines)
            lines = []
            for i in xrange(sent.num_preds):
                yield CoNLL09_Sent_with_Pred(sent, i)
            if sent.num_preds == 0:
                yield CoNLL09_Sent_with_Pred(sent, -1)
        else:
            line = line.strip().split('\t')
            line.append(pred.strip())
            line.append(stag.strip())
            lines.append(line)
    [f.close() for f in fs]


def test_frames():
    fn = 'data/conll09/pred/train.tag'
    count = 0
    sent_count = 0
    with open(fn, 'r') as f:
        for sent in conll09_generator(f):
            if sent.count > 0:
                sent_count += 1
                count += sent.count

    print('{} sentences with errors, {} total errors'.format(sent_count,count))


if __name__ == '__main__':
    test_frames()
