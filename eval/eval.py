# Wrapper for the CoNLL evaluation script
from __future__ import division

import os
import sys
from subprocess import check_output

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


def get_f1(predicted, gold):
    """
    predicted and gold should both be lists of strings,
    '_' means no prediction
    returns labeled_f1, unlabeled_f1
    """
    correct_labeled = 0
    correct_unlabeled = 0
    num_predicted = 0
    num_gold = 0

    for p, g in zip(predicted, gold):
        if p != '_':
            num_predicted += 1
            if p == g:
                correct_labeled += 1
                correct_unlabeled += 1
            elif g != '_':
                correct_unlabeled += 1

        if g != '_':
            num_gold += 1

    if num_predicted == 0 or num_gold == 0:
        return 0, 0

    labeled_precision = correct_labeled / num_predicted
    labeled_recall = correct_labeled / num_gold
    labeled_f1 = (2 * labeled_precision * labeled_recall /
                  (labeled_precision + labeled_recall))

    unlabeled_precision = correct_unlabeled / num_predicted
    unlabeled_recall = correct_unlabeled / num_gold
    unlabeled_f1 = (2 * unlabeled_precision * unlabeled_recall /
                    (unlabeled_precision + unlabeled_recall))

    return 100 * labeled_f1, 100 * unlabeled_f1


def get_f1_from_files(fn_pred, fn_gold):
    with open(fn_pred, 'r') as f_pred:
        lines = f_pred.read().split('\n')
        predicted = [line for line in lines if line != '']
    with open(fn_gold, 'r') as f_gold:
        lines = f_gold.read().split('\n')
        gold = [line for line in lines if line != '']
    return get_f1(predicted, gold)
    
    

if __name__ == '__main__':
    fn_pred = sys.argv[1]
    fn_gold = sys.argv[2]
    labeled_f1, unlabeled_f1 = get_f1_from_files(fn_pred, fn_gold)
    print('Labeled F1:    {0:.2f}'.format(labeled_f1))
    print('Unlabeled F1:  {0:.2f}'.format(unlabeled_f1))
