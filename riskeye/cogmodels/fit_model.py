import datetime
import argparse
from bauer.models import RiskModel, RiskRegressionModel, FlexibleSDRiskRegressionModel
from riskeye.utils.data import get_all_behavior, get_all_eyepos_info
import os.path as op
import os
import arviz as az
import numpy as np
import pandas as pd

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-riskeye'):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)
    try:
        az.to_netcdf(trace,
                    op.join(target_folder, f'model-{model_label}_trace.netcdf'))

    except Exception as e:
        print(e)
        print('Saving to random filename instead: ')
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        az.to_netcdf(op.join(target_folder, f'model-{model_label}_trace_{suffix}.netcdf'))

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
    elif model_label == '4':
        model = RiskRegressionModel(df, regressors={'n1_evidence_mu':'0 + first_saccade_left11+exptype:first_saccade_left11',
                                                    'n2_evidence_mu':'0 + first_saccade_left11+exptype:first_saccade_left11',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype', 'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '5':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'exptype*left_duration_prop', 'n2_evidence_sd':'exptype*right_duration_prop', 'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '6':
        model = RiskRegressionModel(df, regressors={'n1_evidence_mu':'0 + first_saccade_left11+exptype:first_saccade_left11',
                                                    'n2_evidence_mu':'0 + first_saccade_left11+exptype:first_saccade_left11',
                                                    'n1_evidence_sd':'exptype*first_saccade', 'n2_evidence_sd':'exptype*first_saccade',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '7':
        model = RiskRegressionModel(df, regressors={'n1_evidence_mu':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'n2_evidence_mu':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'n1_evidence_sd':'exptype*left_duration_prop', 'n2_evidence_sd':'exptype*left_duration_prop',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '8':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'evidence_sd_diff':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '8b':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'evidence_sd_diff':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='shared')
    elif model_label == '9':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'0 + first_saccade_left11+exptype:first_saccade_left11',
                                                    'evidence_sd_diff':'0 + first_saccade_left11+exptype:first_saccade_left11',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '10':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'exptype',
         'n2_evidence_sd':'exptype', 'prior_mu':'exptype', 'prior_std':'exptype'},
         prior_estimate='shared')

    elif model_label == '11':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'0 + left_duration_prop+exptype:left_duration_prop',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '12':
        model = RiskRegressionModel(df, regressors={'evidence_sd_diff':'0+risky_first*exptype',
                                                    'n1_evidence_sd':'exptype',
         'n2_evidence_sd':'exptype', 'prior_mu':'exptype', 'prior_std':'exptype'},
         prior_estimate='shared')
    elif model_label == '13':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'left_duration_prop*exptype',
                                                    'n2_evidence_mu':'0+safe_duration + safe_duration:exptype',
                                                    'n1_evidence_sd':'left_duration*exptype', 'n2_evidence_sd':'right_duration*exptype',
                                                    'risky_prior_mu':'exptype', 'risky_prior_std':'exptype', 'safe_prior_mu':'exptype', 'safe_prior_std':'exptype'}, prior_estimate='full')
    elif model_label == '14':
        model = RiskRegressionModel(df, regressors={'evidence_sd_diff':'0+risky_order_11*exptype',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype', 'prior_mu':'exptype', 'prior_std':'exptype'}, prior_estimate='shared')
    elif model_label == '15':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype', 'prior_mu':'exptype', 'prior_std':'exptype'}, prior_estimate='shared')

    elif model_label == '16':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'0+saw_risky_first_11',
                                                    'n1_evidence_sd':'exptype*risky_duration', 'n2_evidence_sd':'exptype*safe_duration', 'prior_mu':'exptype', 'prior_std':'exptype'}, prior_estimate='shared')
    elif model_label == '17':
        model = RiskRegressionModel(df, regressors={'evidence_mu_diff':'0+risky_duration_prop+risky_duration_prop:exptype',
                                                    'evidence_sd_diff':'0+risky_duration_prop+risky_duration_prop:exptype',
                                                    'n1_evidence_sd':'exptype', 'n2_evidence_sd':'exptype', 'prior_mu':'exptype', 'prior_std':'exptype'}, prior_estimate='shared')

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
    elif model_label == 'flexible3':
        model = FlexibleSDRiskRegressionModel(df, regressors={'evidence_sd_poly0':'exptype',
                                        'evidence_sd_poly1':'exptype',
                                        'evidence_sd_poly2':'exptype',
                                        'evidence_sd_poly3':'exptype',
                                        'evidence_sd_poly4':'exptype', 'prior_mu':'exptype', 'prior_std':'exptype'}, 
                                        prior_estimate='shared', polynomial_order=5, bspline=True,  fit_seperate_evidence_sd=False)
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(bids_folder='/data/ds-riskeye', model_label=None, exclude_outliers=True):

    df = get_all_behavior(bids_folder=bids_folder, exclude_outliers=exclude_outliers)

    if model_label in ['15', '16', '17']:
        df['n1'] = df['n_left'].where(df['p_left'] == 0.55, df['n_right'])
        df['p1'] = 0.55
        df['n2'] = df['n_left'].where(df['p_left'] == 1.0, df['n_right'])
        df['p2'] = 1.0
        df['chose_risky'] = df['chose_risky'].astype(bool)
        df['choice'] = ((df['p2'] == 0.55) & df['chose_risky']) | ((df['p1'] == 0.55) & ~df['chose_risky'])

    elif model_label == '2':
        df['n1'] = df['n_left'].where(df['first_saccade'] == 'left_option', df['n_right'])
        df['p1'] = df['p_left'].where(df['first_saccade'] == 'left_option', df['p_right'])
        df['n2'] = df['n_left'].where(df['first_saccade'] == 'right_option', df['n_right'])
        df['p2'] = df['p_left'].where(df['first_saccade'] == 'right_option', df['p_right'])
        df['chose_risky'] = df['chose_risky'].astype(bool)
        df['choice'] = ((df['p2'] == 0.55) & df['chose_risky']) | ((df['p1'] == 0.55) & ~df['chose_risky'])
    else:
        df['n1'], df['n2'] = df['n_left'], df['n_right']
        df['p1'], df['p2'] = df['p_left'], df['p_right']
        df['choice'] = df['leftRight'] == -1

    df['risky_first'] = df['p1'] == 0.55

    if model_label in ['4', '6', '9']:
        df['first_saccade_left11'] = 1
        df['first_saccade_left11'] = df['first_saccade_left11'].where(df['first_saccade'] == 'left_option', -1)

    if model_label in ['5', '7', '8', '11', '13']:
        df['left_duration_prop'] = df.groupby(['subject', 'exptype'], group_keys=False)['left_duration_prop'].transform(lambda x: (x - x.mean()) / x.std())
        df['right_duration_prop'] = df.groupby(['subject', 'exptype'], group_keys=False)['right_duration_prop'].transform(lambda x: (x - x.mean()) / x.std())

    if model_label in ['13']:
        df['left_duration'] = df.groupby(['subject', 'exptype'], group_keys=False)['left_duration'].transform(lambda x: (x - x.mean()) / x.std())
        df['right_duration'] = df.groupby(['subject', 'exptype'], group_keys=False)['right_duration'].transform(lambda x: (x - x.mean()) / x.std())

    if model_label in ['14']:
        df['risky_order_11'] = df['risky_first'].map({True:1, False:-1})

    if model_label in ['16']:
        df['saw_risky_first'] = ((df['first_saccade'] == 'left_option') & (df['p_left']==0.55)) | ((df['first_saccade'] == 'right_option') & (df['p_right']==.55))
        df['saw_risky_first_11'] = df['saw_risky_first'].map({True:1, False:-1})
        df['risky_duration'] = df['left_duration'].where(df['p_left'] == 0.55, df['right_duration'])
        df['safe_duration'] = df['left_duration'].where(df['p_left'] == 1.0, df['right_duration'])

        df['risky_duration_prop'] = df.groupby(['subject', 'exptype'], group_keys=False)['risky_duration_prop'].transform(lambda x: (x - x.mean()) / x.std())
        df['safe_duration_prop'] = df.groupby(['subject', 'exptype'], group_keys=False)['safe_duration_prop'].transform(lambda x: (x - x.mean()) / x.std())

    if model_label in ['2', '3', '4', '5', '6', '7', '8', '8b', '9', '11', '13', '16', '17']:
        df = df[~df['first_saccade'].isnull()]



    df = df.reset_index('exptype')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-riskeye')
    parser.add_argument('--n_burnin', default=1000, type=int)
    parser.add_argument('--n_samples', default=1000, type=int)
    args = parser.parse_args()

    main(args.model_label, burnin=args.n_burnin, samples=args.n_samples, bids_folder=args.bids_folder)