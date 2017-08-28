# train.py
# Trains the predicate disambiguation model in disamb.py
from __future__ import print_function
from __future__ import division

import argparse
import os
import tensorflow as tf
import numpy as np
import cPickle as pickle
from timeit import default_timer as timer

from model.disamb.disamb import DisambModel
from util import vocab


parser = argparse.ArgumentParser(
    description="Hyperparameters for training an SRL model")
parser.add_argument("--word_embed_size",
                    help="Embedding size for words",
                    default=100, type=int)
parser.add_argument("--pos_embed_size",
                    help="Embedding size for parts of speech",
                    default=16, type=int)
parser.add_argument("--state_size",
                    help="Size of LSTM hidden state",
                    default=100, type=int)
parser.add_argument("--batch_size",
                    help="Batch size",
                    default=100, type=int)
parser.add_argument("--num_layers",
                    help="Number of layers in the BiLSTM",
                    default=2, type=int)
parser.add_argument("--dropout",
                    help="Dropout keep probability (between LSTM layers)",
                    default=1.0, type=float)
parser.add_argument("--recurrent_dropout",
                    help="Dropout keep probability (between LSTM cells)",
                    default=1.0, type=float)
parser.add_argument("--use_word_dropout",
                    help="Use word dropout",
                    action="store_true")
parser.add_argument("--learning_rate",
                    help="Learning rate",
                    default=0.01, type=float)
parser.add_argument("--max_epochs",
                    help="Maximum number of epochs to train for",
                    default=50, type=int)
parser.add_argument("--restrict_labels",
                    help="Only allow labels from a predicate's frame",
                    action="store_true", default=False)
parser.add_argument("--use_stags",
                    help="Use supertags",
                    action="store_true", default=False)
parser.add_argument("--stag_type",
                    help="Choice of supertags: ud, model1, model2, or None",
                    choices=['model1', 'model2'],
                    default='model1')
parser.add_argument("--use_gold_stags",
                    help="Train with gold supertags",
                    action="store_true", default=False)
parser.add_argument("--stag_embed_size",
                    help="Embedding size for supertags",
                    default=50, type=int)
parser.add_argument("--use_stag_features",
                    help="Use feature based supertag embeddings",
                    action="store_true", default=False)
parser.add_argument("--stag_feature_embed_size",
                    help="Size of feature-based supertag embeddings",
                    default=25, type=int)
parser.add_argument("--early_stopping",
                    help="Stop after n epochs of no improvement",
                    default=8, type=int)
parser.add_argument("--seed",
                    help="Random seed for tensorflow and numpy",
                    default=47, type=int)
parser.add_argument("--use_highway_lstm",
                    help="Use LSTM with highway connections",
                    action="store_true")
parser.add_argument("--optimizer",
                    help="Choice of optimizer",
                    choices=['adam', 'adadelta'], default='adam')
parser.add_argument("--debug",
                    help="Use a smaller configuration for debugging",
                    action="store_true", default=False)


class Debug_Args(object):
    def __init__(self):
        self.word_embed_size = 16
        self.pos_embed_size = 4
        self.state_size = 32
        self.batch_size = 50
        self.num_layers = 1
        self.dropout = 1.0
        self.recurrent_dropout = 0.9
        self.learning_rate = 0.01
        self.max_epochs = 10
        self.use_stags = False
        self.stag_embed_size = 16
        self.restrict_labels = False
        self.early_stopping = 3
        self.seed = 89
        self.stag_type = 'model1'
        self.use_gold_stags = True
        self.use_stag_features = True
        self.stag_feature_embed_size = 8
        self.use_word_dropout = True
        self.use_highway_lstm = True
        self.optimizer = 'adam'
    

def train(args):
    # Set the filepaths for training and validation
    fn_txt_train = 'data/conll09/train.txt'
    fn_preds_train = 'data/conll09/gold/train_predicates.txt'
    if args.use_stags:
        fn_stags_train = 'data/conll09/{}/train_stags_{}.txt'.format(
            ('gold' if args.use_gold_stags else 'pred'),
            args.stag_type)
    else:
        fn_stags_train = fn_preds_train
        
    fn_txt_valid = 'data/conll09/dev.txt'
    fn_preds_valid = 'data/conll09/gold/dev_predicates.txt'
    if args.use_stags:
        fn_stags_valid = 'data/conll09/pred/dev_stags_{}.txt'.format(
            args.stag_type)
    else:
        fn_stags_valid = fn_preds_valid
    

    # Come up with a model name based on the hyperparameters
    model_suffix = '_'
    if args.restrict_labels:
        model_suffix += '_rl'
    if args.use_stags:
        model_suffix += '_st{}_{}'.format(args.stag_embed_size, args.stag_type)
        model_suffix += 'g' if args.use_gold_stags else 'p'
        if args.use_stag_features:
            model_suffix += 'f{}'.format(args.stag_feature_embed_size)
    if args.dropout < 1.0:
        model_suffix += '_dr{}'.format(args.dropout)
    if args.recurrent_dropout < 1.0:
        model_suffix += '_rdr{}'.format(args.recurrent_dropout)
    if args.use_word_dropout:
        model_suffix += '_wdr'
    if args.use_highway_lstm:
        model_suffix += '_hw'
    if args.optimizer != 'adam':
        model_suffix += '_' + args.optimizer
    if args.seed != 89:
        model_suffix += '_s{}'.format(args.seed)
    fn_sys = 'output/predictions/dev{}.txt'.format(model_suffix)

    # Prepare for saving the model
    model_dir = 'output/models/disamb/' + model_suffix + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Saving args to', model_dir + 'args.pkl')
    with open(model_dir + 'args.pkl', 'w') as f:
        pickle.dump(args, f)

    vocabs = vocab.get_vocabs(args.stag_type)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)

        print("Building model...")        
        model = DisambModel(vocabs, args)
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
                    session, vocabs, fn_txt_train, fn_stags_train)
                end = timer()
                print('Done with epoch {}'.format(i))
                print('Avg loss: {}, total time: {}'.format(
                    train_loss, end-start))

                print('-' * 78)
                print('Validating...')
                valid_loss, labeled_f1, unlabeled_f1 = model.run_testing_epoch(
                    session, vocabs, fn_txt_valid, fn_stags_valid)
                print('Validation loss: {}'.format(valid_loss))

                print('-' * 78)
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
    
