# Test a trained SRL model
from __future__ import print_function
from __future__ import division

import os
import argparse
import tensorflow as tf
import cPickle as pickle

from util import vocab


parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Directory containing the saved model")
parser.add_argument("data", help="train, test, dev, or ood",
                    choices=['train', 'test', 'dev', 'ood'])
parser.add_argument("-rl", "--restrict_labels", dest="restrict_labels",
                    help="Only allow valid labels",
                    action="store_true", default=False)


def test(args):
    fn_valid = 'data/conll09/pred/{}.tag'.format(args.data)
    fn_gold = 'data/conll09/gold/{}.txt'.format(args.data)
    fn_sys = 'output/predictions/{}.txt'.format(args.data)
    model_dir = args.model_dir
    
    vocabs = vocab.get_vocabs()

    with open(os.path.join(model_dir, 'args.pkl'), 'r') as f:
        model_args = pickle.load(f)
    model_args.restrict_labels = args.restrict_labels
    
    print("Building model...")
    model = SRL_Model(vocabs, model_args)

    saver = tf.train.Saver(max_to_keep=1)
    
    with tf.Session() as session:
        print('Restoring model...')
        saver.restore(session, tf.train.latest_checkpoint(model_dir))
        
        print('-' * 78)
        print('Validating...')
        valid_loss = model.run_testing_epoch(session, vocabs,
                                             fn_valid, fn_sys)
        print('Validation loss: {}'.format(valid_loss))

        print('-' * 78)
        print('Running evaluation script...')
        labeled_f1, unlabeled_f1 = run_evaluation_script(fn_gold, fn_sys)
        print('Labeled F1:    {0:.2f}'.format(labeled_f1))
        print('Unlabeled F1:  {0:.2f}'.format(unlabeled_f1))


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
