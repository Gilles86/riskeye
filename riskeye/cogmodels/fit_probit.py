import numpy as np
import argparse
from riskeye.utils.data import get_all_behavior, get_all_eyepos_info
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
    if model_label == 'probit00':
        model = bambi.Model('chose_risky ~ x*exptype + (x*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit0':
        model = bambi.Model('chose_risky ~ x*exptype*C(n_safe) + (x*exptype*C(n_safe)|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit0b':
        model = bambi.Model('chose_risky ~ x*exptype*n_safe + (x*exptype*n_safe|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit1':
        model = bambi.Model('chose_risky ~ x*risky_left*exptype*C(n_safe) + (x*risky_left*exptype*C(n_safe)|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit2':
        model = bambi.Model('chose_risky ~ x*risky_left*exptype + (x*risky_left*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_risky_looked_at_first_simple':
        model = bambi.Model('chose_risky ~ x*risky_seen_first*exptype + (x*risky_seen_first*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_risky_looked_at_first':
        model = bambi.Model('chose_risky ~ x*risky_seen_first*C(n_safe)*exptype + (x*risky_seen_first*C(n_safe)*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_risky_looked_at_last_simple':
        model = bambi.Model('chose_risky ~ x*risky_seen_last*exptype + (x*risky_seen_last*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_risky_looked_at_last':
        model = bambi.Model('chose_risky ~ x*risky_seen_last*C(n_safe)*exptype + (x*risky_seen_last*C(n_safe)*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_prop_looked_at_risky_simple':
        model = bambi.Model('chose_risky ~ x*risky_duration_prop*exptype + (x*risky_duration_prop*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_prop_looked_at_risky':
        model = bambi.Model('chose_risky ~ x*risky_duration_prop*C(n_safe)*exptype + (x*risky_duration_prop*C(n_safe)*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_prop_looked_at_risky_mediansplit':
        model = bambi.Model('chose_risky ~ x*risky_duration_prop_split*C(n_safe)*exptype + (x*risky_duration_prop_split*C(n_safe)*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_prop_looked_at_risky_mediansplit_simple':
        model = bambi.Model('chose_risky ~ x*risky_duration_prop_split*exptype + (x*risky_duration_prop_split*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_total_gaze_duration':
        model = bambi.Model('chose_risky ~ x*risky_duration*exptype + x*safe_duration*exptype + (x*risky_duration*exptype + x*safe_duration*exptype|subject)', df.reset_index(), link='probit', family='bernoulli')
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(model_label=None, bids_folder='/data/ds-riskeye'):
    df = get_all_behavior(bids_folder=bids_folder)
    df['x'] = df['log(risky/safe)']
    df['log_n_safe'] = np.log(df['n_safe'])
    
    if model_label in ['probit1', 'probit2']:
        df['risky_left'] = df['p_right'] == 1.0

    if model_label in ['probit_prop_looked_at_risky_simple', 'probit_prop_looked_at_risky', 'probit_prop_looked_at_risky_mediansplit', 'probit_prop_looked_at_risky_mediansplit_simple', 'probit_total_gaze_duration']:
        df = df[~df.isnull()['first_saccade']]

    if model_label in ['probit_prop_looked_at_risky_mediansplit', 'probit_prop_looked_at_risky_mediansplit_simple']:
        df = df[~df['risky_duration_prop_split'].isnull()]


    if model_label in ['probit_total_gaze_duration']:
        df['risky_duration'] = df['left_duration'].where(df['p_left'] == 0.55, df['right_duration'])
        df['safe_duration'] = df['left_duration'].where(df['p_left'] == 1.0, df['right_duration'])

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-riskeye')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)