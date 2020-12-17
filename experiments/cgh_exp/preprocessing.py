#!/usr/bin/env python
import os, sys
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import subprocess
import pickle

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
import utils.exp_utils as utils


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))


if __name__ == "__main__":

    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)
