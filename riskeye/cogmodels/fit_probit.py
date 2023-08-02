import numpy as np
import argparse
from riskeye.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import bambi
import pandas as pd

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-riskeye'):

    df = get_data(model_label, bids_folder)
    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.8

    model = build_model(model_label, df)
    trace = model.fit(burnin, samples, init='adapt_diag', target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'model-{model_label}_trace.netcdf'))

def build_model(model_label, df):
    if model_label == 'probit1':
        model = bambi.Model('chose_risky ~ x*risky_left*exptype*C(n_safe) + (x*risky_left*exptype*C(n_safe)|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit2':
        model = bambi.Model('chose_risky ~ x*risky_left*exptype + (x*risky_left*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(model_label=None, bids_folder='/data/ds-riskeye'):
    df = get_all_behavior(bids_folder=bids_folder)
    df['x'] = df['log(risky/safe)']
    df['log_n_safe'] = np.log(df['n_safe'])
    df['risky_left'] = df['p_right'] == 1.0


    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-riskeye')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)