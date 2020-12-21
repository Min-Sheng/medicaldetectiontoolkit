import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.exp_utils as utils
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')

parser = argparse.ArgumentParser()

parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                    help='specifies, from which source experiment to load configs and data_loader.')
parser.add_argument('--exp_dir', type=str, help='path to experiment dir. will be created if non existent.')
parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
parser.add_argument('--min_det_thresh', type=float, default=0.3, help='minimum confidence value to select predictions for evaluation.')
def main():
    args = parser.parse_args()

    cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
    cf.min_det_thresh = args.min_det_thresh
    print(cf.min_det_thresh)
    save_dir = args.exp_dir
    submission_path = os.path.join(save_dir, 'fold_0/results_0.csv')
    eval_dir = os.path.join(save_dir, 'FROC')
    
    # Start evaluating
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    noduleCADEvaluation('/home/vincentwu/LIDC/LIDC_annotations.csv', None,
    cf.file_csv_dict['test_csv'], submission_path, eval_dir, cf.min_det_thresh)
        
    print

if __name__ == '__main__':
    main()