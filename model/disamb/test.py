# Test a trained SRL model
from __future__ import print_function
from __future__ import division

import os
import argparse
import tensorflow as tf
import numpy as np
import cPickle as pickle

from model.disamb.disamb import DisambModel
from util import vocab


parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Directory containing the saved model")
parser.add_argument("data", help="train, test, dev, or ood",
                    choices=['train', 'test', 'dev', 'ood'])
parser.add_argument("--fill_all",
                    help="Guess all predicates (not just when fill_pred=Y)",
                    action="store_true", default=False)
parser.add_argument("--fn_out",
                    help="Name of the file to write predictions to",
                    default=None)


def test(args):
    model_dir = args.model_dir    
    with open(os.path.join(model_dir, 'args.pkl'), 'r') as f:
        model_args = pickle.load(f)
        
    fn_txt_valid = 'data/{}/conll09/{}.txt'.format(
        model_args.language, args.data)
    fn_preds_gold = 'data/{}/conll09/gold/{}_predicates.txt'.format(
        model_args.language, args.data)
    fn_stags_valid = 'data/{}/conll09/pred/{}_stags_{}.txt'.format(
        model_args.language, args.data, model_args.stag_type)
    fn_sys = 'output/predictions/{}.txt'.format(args.data)
    if args.fn_out is not None:
        fn_sys = args.fn_out
    
    vocabs = vocab.get_vocabs(model_args.language, model_args.stag_type)

    with tf.Graph().as_default():
        tf.set_random_seed(model_args.seed)
        np.random.seed(model_args.seed)    
    
        print("Building model...")
        model = DisambModel(vocabs, model_args)

        saver = tf.train.Saver()

        with tf.Session() as session:
            print('Restoring model...')
            saver.restore(session, tf.train.latest_checkpoint(model_dir))

            print('-' * 78)
            print('Validating...')
            valid_loss, labeled_f1, unlabeled_f1 = model.run_testing_epoch(
                session, vocabs, fn_txt_valid, fn_stags_valid,
                fn_sys, fn_preds_gold, model_args.language, args.fill_all)
            print('Validation loss: {}'.format(valid_loss))
            print('Labeled F1:    {0:.2f}'.format(labeled_f1))
            print('Unlabeled F1:  {0:.2f}'.format(unlabeled_f1))


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
