# Wrapper for the CoNLL evaluation script
import os
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
