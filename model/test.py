# Test the model so far.

from model import srl

class Vocab(object):
    def __init__(self, size=10):
        self.size = size
        self.idx_to_word = None

    def add_idx_to_word(self, idx_to_word):
        self.idx_to_word = idx_to_word
        self.size = max(idx_to_word.keys()) + 1

        
class Args(object):
    def __init__(self):
        self.word_embed_size = 16
        self.pos_embed_size = 4
        self.lemma_embed_size = 8
        self.state_size = 32
        self.batch_size = 50
        self.num_layers = 3
        self.dropout = 0.7
        self.role_embed_size = 8
        self.output_lemma_embed_size = 12


def get_idx_to_word():
    fn = 'data/embeddings/sskip.100.vectors'
    idx_to_word = {}
    i = 1
    with open(fn, 'r') as f:
        for line in f:
            word = line.split(' ')[0]
            idx_to_word[i] = word
            i += 1
    return idx_to_word
        
        
def test():
    vocabs = {
        'words': Vocab(),
        'lemmas': Vocab(),
        'pos': Vocab(),
        'roles': Vocab()
    }
    args = Args()

    vocabs['words'].add_idx_to_word(get_idx_to_word())
    
    srl.SRL_Model(vocabs, args)


if __name__ == '__main__':
    test()
