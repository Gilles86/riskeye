import argparse
from bauer.models import RiskModel, RiskRegressionModel
from riskeye.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import numpy as np

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-riskeye'):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.8

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'model-{model_label}_trace.netcdf'))

def build_model(model_label, df):
    if model_label == '1':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'exptype',
         'n2_evidence_sd':'exptype', 'risky_prior_mu':'exptype', 'risky_prior_std':'exptype',
          'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'},
         prior_estimate='full')
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(bids_folder='/data/ds-tmsrisk', model_label=None):

    df = get_all_behavior(bids_folder=bids_folder)
    df = df.reset_index('exptype')

    df['n1'], df['n2'] = df['n_left'], df['n_right']
    df['p1'], df['p2'] = df['p_left'], df['p_right']
    df['choice'] = df['leftRight'] == 1

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-riskeye')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)