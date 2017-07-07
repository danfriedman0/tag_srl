# train.py
# Trains the SRL model in srl.py
from __future__ import print_function
from __future__ import division

import argparse
import os
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
parser.add_argument("--learning_rate",
                    help="Learning rate",
                    default=0.01, type=float)
parser.add_argument("--max_epochs",
                    help="Maximum number of epochs to train for",
                    default=25, type=int)
parser.add_argument("--restrict_labels",
                    help="Only allow labels from a predicate's frame",
                    action="store_true", default=False)
parser.add_argument("--use_gold_preds",
                    help="Use gold predicates instead of predicted",
                    action="store_true", default=False)
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
        self.learning_rate = 0.01
        self.role_embed_size = 8
        self.output_lemma_embed_size = 12
        self.max_epochs = 2


def run_evaluation_script(fn_gold, fn_sys, print_output=False):
    args = ['perl', 'eval/eval09.pl', '-g', fn_gold, '-s', fn_sys, '-q']
    with open(os.devnull, 'w') as devnull:
        output = check_output(args, stderr=devnull)
    if print_output:
        print(output)
    # Just want to return labeled and unlabeled semantic F1 scores
    lines = output.split('\n')
    lf1_line = [line for line in lines if line.startswith('  Labeled F1')][0]
    labeled_f1 = float(lf1_line.strip().split(' ')[-1])
    uf1_line = [line for line in lines if line.startswith('  Unlabeled F1')][0]
    unlabeled_f1 = float(uf1_line.strip().split(' ')[-1])    
    return labeled_f1, unlabeled_f1
    

def train(args):
    if args.use_gold_preds:
        fn_train = 'data/conll09/gold/train.txt'
        fn_valid = 'data/conll09/gold/dev.txt'
    else:
        fn_train = 'data/conll09/pred/train.txt'
        fn_valid = 'data/conll09/pred/dev.txt'


    model_suffix = ''
    if args.restrict_labels:
        model_suffix += '_rl'
    if args.use_gold_preds:
        model_suffix += '_gp'
    else:
        model_suffix += '_pp'
    fn_sys = 'output/predictions/dev{}.txt'.format(model_suffix)
    
    vocabs = vocab.get_vocabs()

    print("Building model...")
    model = SRL_Model(vocabs, args)

    saver = tf.train.Saver(max_to_keep=1)
    model_dir = 'output/models/srl' + model_suffix + '/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    with tf.Session() as session:
        best_f1 = 0
        bad_streak = 0

        session.run(tf.global_variables_initializer())

        print('Saving model to', model_dir)
        saver.save(session, model_dir, global_step=0)
        
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
            print('Running evaluation script...')
            labeled_f1, unlabeled_f1 = run_evaluation_script(fn_valid, fn_sys)
            print('Labeled F1:    {0:.2f}'.format(labeled_f1))
            print('Unlabeled F1:  {0:.2f}'.format(unlabeled_f1))

            if labeled_f1 > best_f1:
                best_f1 = labeled_f1
                print('Saving model to', model_dir)
                saver.save(session, model_dir, global_step=i)
            elif best_f1 < labeled_f1:
                bad_streak += 1
                if bad_streak >= 3:
                    print('F1 decreased for 3 epochs in a row, stopping early')
                    break
            

if __name__ == '__main__':
    args = parser.parse_args()
    if args.debug:
        args = Debug_Args()
    train(args)
    
