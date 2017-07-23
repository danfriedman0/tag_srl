# train.py
# Trains the SRL model in srl.py
from __future__ import print_function
from __future__ import division

import argparse
import os
import tensorflow as tf
import numpy as np
import cPickle as pickle
from timeit import default_timer as timer

from model.srl import SRL_Model
from eval.eval import run_evaluation_script
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
parser.add_argument("--training_split",
                    help="Train on gold or predicted predicates",
                    choices=['gold', 'pred'],
                    default='gold')
parser.add_argument("--testing_split",
                    help="Test on gold or predicted predicates",
                    choices=['gold', 'pred'],
                    default='pred')
parser.add_argument("--use_stags",
                    help="Use UD supertags in the word representation",
                    action="store_true", default=False)
parser.add_argument("--stag_embed_size",
                    help="Embedding size for supertags",
                    default=50, type=int)
parser.add_argument("--use_basic_classifier",
                    help="Use the basic role classifier",
                    action="store_true", default=False)
parser.add_argument("--early_stopping",
                    help="Stop after n epochs of no improvement",
                    default=8, type=int)
parser.add_argument("--seed",
                    help="Random seed for tensorflow and numpy",
                    default=89, type=int)
parser.add_argument("--use_tf_lstm",
                    help="Use default tensorflow LSTM implementation",
                    action="store_true", default=False)
parser.add_argument("--debug",
                    help="Use a smaller configuration for debugging",
                    action="store_true", default=False)


class Debug_Args(object):
    def __init__(self):
        self.word_embed_size = 16
        self.pos_embed_size = 4
        self.lemma_embed_size = 8
        self.state_size = 32
        self.batch_size = 10
        self.num_layers = 2
        self.dropout = 0.7
        self.learning_rate = 0.01
        self.role_embed_size = 8
        self.output_lemma_embed_size = 12
        self.max_epochs = 10
        self.use_stags = True
        self.stag_embed_size = 16
        self.use_gold_preds = False
        self.restrict_labels = False
        self.use_basic_classifier = False
        self.early_stopping = 3
        self.seed = 89
        self.use_tf_lstm = False
        self.training_split = 'gold'
        self.testing_split = 'pred'
    

def train(args):
    fn_train = 'data/conll09/{}/train.tag'.format(args.training_split)
    fn_valid = 'data/conll09/{}/dev.tag'.format(args.testing_split)
    fn_gold = 'data/conll09/gold/dev.txt'

    # Come up with a model name based on the hyperparameters
    model_suffix = '_'
    model_suffix += 'g' if args.training_split == 'gold' else 'p'
    model_suffix += 'g' if args.testing_split == 'gold' else 'p'
    if args.restrict_labels:
        model_suffix += '_rl'
    if args.use_stags:
        model_suffix += '_st'
    if args.dropout < 1.0:
        model_suffix += '_dr{}'.format(args.dropout)
    if args.use_basic_classifier:
        model_suffix += '_bc'
    if args.use_tf_lstm:
        model_suffix += '_tf'
    if args.seed != 89:
        model_suffix += '_s{}'.format(args.seed)
    fn_sys = 'output/predictions/dev{}.txt'.format(model_suffix)

    # Prepare for saving the model
    model_dir = 'output/models/srl' + model_suffix + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Saving args to', model_dir + 'args.pkl')
    with open(model_dir + 'args.pkl', 'w') as f:
        pickle.dump(args, f)

    vocabs = vocab.get_vocabs()

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)

        print("Building model...")        
        model = SRL_Model(vocabs, args)
        saver = tf.train.Saver(max_to_keep=1)
        
        with tf.Session() as session:
            best_f1 = 0
            bad_streak = 0

            session.run(tf.global_variables_initializer())

            for i in xrange(args.max_epochs):
                print('-' * 78)
                print('Epoch {}'.format(i))
                start = timer()
                train_loss = model.run_training_epoch(
                    session, vocabs, fn_train)
                end = timer()
                print('Done with epoch {}'.format(i))
                print('Avg loss: {}, total time: {}'.format(
                    train_loss, end-start))

                print('-' * 78)
                print('Validating...')
                valid_loss = model.run_testing_epoch(
                    session, vocabs, fn_valid, fn_sys)
                print('Validation loss: {}'.format(valid_loss))

                print('-' * 78)
                print('Running evaluation script...')
                labeled_f1, unlabeled_f1 = run_evaluation_script(
                    fn_gold, fn_sys)
                print('Labeled F1:    {0:.2f}'.format(labeled_f1))
                print('Unlabeled F1:  {0:.2f}'.format(unlabeled_f1))

                if labeled_f1 > best_f1:
                    best_f1 = labeled_f1
                    bad_streak = 0
                    print('Saving model to', model_dir + 'model')
                    saver.save(session, model_dir + 'model')
                else:
                    print('F1 deteriorated (best score: {})'.format(best_f1))
                    bad_streak += 1
                    if bad_streak >= args.early_stopping:
                        print('No F1 improvement for %d epochs, stopping early'
                              % args.early_stopping)
                        print('Best F1 score: {0:.2f}'.format(best_f1))
                        break
                

if __name__ == '__main__':
    args = parser.parse_args()
    if args.debug:
        args = Debug_Args()
    train(args)
    
