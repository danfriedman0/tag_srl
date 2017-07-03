# conll_io.py
# Classes for reading and writing data in CoNLL-09 format

class CoNLL09_Pred_List(object):
    """
    Stores information about a predicate and its arguments.
    Attributes are:
      pred_lemma (e.g. 'elect')
      pred_idx: the index of the predicate in the sentence
      arg_seq: a list of argument labels ('A3', 'AM-TMP', '_', ...),
        one for each word in the sentence
    """
    def __init__(self, pred_lemma, pred_idx, arg_seq):
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
          pos (parts of speech)
          lemmas (only for predicates)
          pred_lists: a list of CoNLL09_Pred_List objects (see above), one
            for each predicate in the sentence
        """
        self.lines = [line[:14] for line in lines]
        self.words = [line[1].lower() for line in lines]
        self.pos = [line[5] for line in lines]
        # self.lemmas = [line[3] for line in lines if line[12] == 'Y' else '_']
        self.lemmas = []
        for line in lines:
            if line[12] == 'Y':
                self.lemmas.append(line[3])
            else:
                self.lemmas.append('_')

        # Add a predicate info list for each predicate in the sentence
        self.pred_lists = []
        num_preds = len(lines[0]) - 14
        pred_num = 0
        for i,line in enumerate(lines):
            if line[12] == 'Y':
                pred_lemma = line[3]
                pred_idx = i
                arg_seq = [line[14 + pred_num] for line in lines]
                self.pred_lists.append(
                    CoNLL09_Pred_List(pred_lemma, pred_idx, arg_seq))
                pred_num += 1

        self.num_preds = len(self.pred_lists)
        self.predictions_list = [['_' for _ in xrange(self.num_preds)]
                                 for _ in xrange(len(self.words))]
        

    def __str__(self):
        out = []
        pred_idxs = [p_list.pred_idx for p_list in self.pred_lists]
        for i, line in enumerate(self.lines):
            line_out = line + self.predictions_list[i]
            out.append('\t'.join(line_out))
        return '\n'.join(out) + '\n'


    def __len__(self):
        return len(self.words)


class CoNLL09_Sent_with_Pred(object):
    """Same as a CoNLL09_Sent but specifies an active predicate"""
    def __init__(self, sent, pred_num):
        self.words = sent.words
        self.pos = sent.pos
        self.lemmas = sent.lemmas

        # pred_num < 0 means that this is a dummy object for a sentence
        # with no predicates
        if pred_num >= 0:
            pred_list = sent.pred_lists[pred_num]
            self.pred = pred_list.pred_lemma
            self.pred_idx = pred_list.pred_idx
            self.labels = pred_list.arg_seq
        else:
            pred_list = []
            self.pred = None
            self.pred_idx = -1
            self.labels = []
            
        self.predictions = []
        self.pred_num = pred_num
        self.parent = sent


    def __len__(self):
        return len(self.words)


    def add_predictions(self, raw_predictions, vocab):
        if self.pred_num < 0:
            return
        predictions = [vocab.decode(p) for p in raw_predictions]
        self.predictions = predictions
        for i, row in enumerate(self.parent.predictions_list):
            row[self.pred_num] = predictions[i]

        
def conll09_generator(f):
    """
    Generator for reading data in CoNLL format.
    Given a file object, yields CoNLL09_Sent_with_Pred objects.
    """
    lines = []
    for line in f:
        if line == '\n':
            sent = CoNLL09_Sent(lines)
            for i in xrange(sent.num_preds):
                yield CoNLL09_Sent_with_Pred(sent, i)
            if sent.num_preds == 0:
                yield CoNLL09_Sent_with_Pred(sent, -1)
            lines = []
        else:
            lines.append(line.strip().split('\t'))
