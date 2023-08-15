import argparse
from bauer.models import RiskModel, RiskRegressionModel, FlexibleSDRiskRegressionModel
from riskeye.utils.data import get_all_behavior, get_all_eyepos_info
import os.path as op
import os
import arviz as az
import numpy as np

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-riskeye'):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

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
    elif model_label == '2':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'exptype',
         'n2_evidence_sd':'exptype', 'risky_prior_mu':'exptype', 'risky_prior_std':'exptype',
          'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'},
         prior_estimate='full')
    elif model_label == '3':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'exptype*seen_risky_first',
         'n2_evidence_sd':'exptype*seen_risky_first', 'risky_prior_mu':'exptype*seen_risky_first', 'risky_prior_std':'exptype*seen_risky_first',
          'safe_prior_mu':'exptype*seen_risky_first', 'safe_prior_std':'exptype*seen_risky_first'},
         prior_estimate='full')

    elif model_label == 'flexible1':
        model = FlexibleSDRiskRegressionModel(df, regressors={'n1_evidence_sd_poly0':'exptype',
                                        'n1_evidence_sd_poly1':'exptype',
                                        'n1_evidence_sd_poly2':'exptype',
                                        'n1_evidence_sd_poly3':'exptype',
                                        'n1_evidence_sd_poly4':'exptype',
                                        'n2_evidence_sd_poly0':'exptype',
                                        'n2_evidence_sd_poly1':'exptype',
                                        'n2_evidence_sd_poly2':'exptype',
                                        'n2_evidence_sd_poly3':'exptype',
                                        'n2_evidence_sd_poly4':'exptype',}, 
                                        prior_estimate='full', polynomial_order=5, bspline=True)
    elif model_label == 'flexible2':
        model = FlexibleSDRiskRegressionModel(df, regressors={'n1_evidence_sd_poly0':'exptype',
                                        'n1_evidence_sd_poly1':'exptype',
                                        'n1_evidence_sd_poly2':'exptype',
                                        'n1_evidence_sd_poly3':'exptype',
                                        'n1_evidence_sd_poly4':'exptype',
                                        'n2_evidence_sd_poly0':'exptype',
                                        'n2_evidence_sd_poly1':'exptype',
                                        'n2_evidence_sd_poly2':'exptype',
                                        'n2_evidence_sd_poly3':'exptype',
                                        'n2_evidence_sd_poly4':'exptype',}, 
                                        prior_estimate='shared', polynomial_order=5, bspline=True)
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(bids_folder='/data/ds-tmsrisk', model_label=None, exclude_outliers=True):

    df = get_all_behavior(bids_folder=bids_folder, exclude_outliers=exclude_outliers)

    if model_label in ['1', 'flexible1', '3', 'flexible2']:
        df['n1'], df['n2'] = df['n_left'], df['n_right']
        df['p1'], df['p2'] = df['p_left'], df['p_right']
        df['choice'] = df['leftRight'] == -1

    if model_label in ['2', '3']:
        df = df[~df['first_saccade'].isnull()]

    if model_label == '2':
        df['n1'] = df['n_left'].where(df['first_saccade'] == 'left_option', df['n_right'])
        df['p1'] = df['p_left'].where(df['first_saccade'] == 'left_option', df['p_right'])
        df['n2'] = df['n_left'].where(df['first_saccade'] == 'right_option', df['n_right'])
        df['p2'] = df['p_left'].where(df['first_saccade'] == 'right_option', df['p_right'])
        df['chose_risky'] = df['chose_risky'].astype(bool)
        df['choice'] = ((df['p2'] == 0.55) & df['chose_risky']) | ((df['p1'] == 0.55) & ~df['chose_risky'])
        df['risky_first'] = df['p1'] == 0.55


    df = df.reset_index('exptype')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-riskeye')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)