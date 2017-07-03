# train.py
# Trains the SRL model in srl.py
from __future__ import print_function
from __future__ import division

import argparse
import tensorflow as tf
from timeit import default_timer as timer
from subprocess import check_output

from model.srl import SRL_Model
from util import vocab

parser = argparse.ArgumentParser(
    description="Hyperparameters for training an SRL model")
parser.add_argument("--word_embed_size",
                    help="Embedding size for words",
                    default=100, type=int)
parser.add_argument("--pos_embed_size",
                    help="Embedding size for parts of speech",
                    default=16, type=int)
parser.add_argument("--lemma_embed_size",
                    help="Embedding size for (input) lemmas",
                    default=100, type=int)
parser.add_argument("--state_size",
                    help="Size of LSTM hidden state",
                    default=512, type=int)
parser.add_argument("--role_embed_size",
                    help="Embedding size for roles (for projection weights)",
                    default=128, type=int)
parser.add_argument("--output_lemma_embed_size",
                    help="Embedding size for (output/projection) lemmas",
                    default=128, type=int)
parser.add_argument("--batch_size",
                    help="Batch size",
                    default=100, type=int)
parser.add_argument("--num_layers",
                    help="Number of layers in the BiLSTM",
                    default=4, type=int)
parser.add_argument("--dropout",
                    help="Dropout probability (between LSTM layers)",
                    default=1.0, type=float)
parser.add_argument("--max_epochs",
                    help="Maximum number of epochs to train for",
                    default=10, type=int)
parser.add_argument("--debug",
                    help="Use a smaller configuration for debuggin",
                    action="store_true", default=False)


class Debug_Args(object):
    def __init__(self):
        self.word_embed_size = 16
        self.pos_embed_size = 4
        self.lemma_embed_size = 8
        self.state_size = 32
        self.batch_size = 50
        self.num_layers = 2
        self.dropout = 0.7
        self.role_embed_size = 8
        self.output_lemma_embed_size = 12
        self.max_epochs = 2


def run_evaluation_script(fn_gold, fn_sys):
    args = ['perl', 'eval/eval09.pl', '-g', fn_gold, '-s', fn_sys, '-q']
    print('Running the CoNLL evaluation script...')
    output = check_output(args)
    print(output)
    
    

def train(args):
    fn_train = 'data/conll2009/CoNLL2009-ST-English-train.txt'
    fn_valid = 'data/conll2009/CoNLL2009-ST-English-development.txt'
    fn_sys = 'output/predictions_dev.txt'
    
    vocabs = vocab.get_vocabs()

    print("Building model...")
    model = SRL_Model(vocabs, args)

    with tf.Session() as session:
        best_loss = float('inf')
        bad_streak = 0

        session.run(tf.global_variables_initializer())

        print('-' * 78)
        print('Validating...')
        valid_loss = model.run_testing_epoch(session, vocabs, fn_valid)
        print('Validation loss: {}'.format(valid_loss))
        print('-' * 78)
        
        run_evaluation_script(fn_valid, fn_sys)
        
        for i in xrange(args.max_epochs):
            print('-' * 78)
            print('Epoch {}'.format(i))
            start = timer()
            train_loss = model.run_training_epoch(session, vocabs, fn_train)
            end = timer()
            print('Done with epoch {}'.format(i))
            print('Avg loss: {}, total time: {}'.format(train_loss, end-start))

            print('-' * 78)
            print('Validating...')
            valid_loss = model.run_testing_epoch(session, vocabs, fn_valid)
            print('Validation loss: {}'.format(valid_loss))
            print('-' * 78)


            if valid_loss < best_loss:
                best_loss = valid_loss
            elif valid_loss > best_loss:
                bad_streak += 1
                if bad_streak >= 2:
                    print('Validation loss deteriorated for two epochs \
                    in a row, stopping early.')
                    break

if __name__ == '__main__':
    args = parser.parse_args()
    if args.debug:
        args = Debug_Args()
    train(args)
    
