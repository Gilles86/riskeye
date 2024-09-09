import arviz as az
import numpy as np
import scipy.stats as ss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def invprobit(x):
    return ss.norm.ppf(x)

def extract_intercept_gamma(trace, model, data, group=False):

    fake_data = get_fake_data(data, group)

    pred = model.predict(trace, 'mean', fake_data, inplace=False, include_group_specific=not group)['posterior']['chose_risky_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    # return pred

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept

    intercept = pd.concat((intercept.droplevel(0, 1),), keys=['intercept'], axis=1)
    gamma = pd.concat((gamma.droplevel(0, 1),), keys=['gamma'], axis=1)

    return intercept, gamma


def get_fake_data(data, group=False):

    data = data.reset_index()

    if group:
        permutations = [[1]]
    else:
        permutations = [data['subject'].unique()]

    permutations += [np.array([0., 1.]), data['n_safe'].unique(), ['symbolic', 'non-symbolic'], [0.0], [0.0]]
    names=['subject', 'x', 'n_safe', 'exptype', 'risky_duration', 'safe_duration']

    for key in ['risky_left', 'risky_seen_first', 'risky_seen_last', 'risky_duration_prop_split']:
        if key in data.columns:
            if key in ['risky_duration_prop_split']:
                permutations.append(['low', 'high'])
            else:
                permutations.append([True, False])

            names.append(key)

    if 'risky_duration_prop' in data.columns:
        permutations.append([0.0])
        names.append('risky_duration_prop')

    if 'total_duration' in data.columns:
        permutations.append([0.0])
        names.append('total_duration')

    fake_data = pd.MultiIndex.from_product(permutations, names=names).to_frame().reset_index(drop=True)
    fake_data['Experiment'] = fake_data['exptype'].map({'symbolic':'Symbols', 'non-symbolic':'Coin clouds'})

    return fake_data

# def get_ppc(trace, model):

def get_rnp(intercept, gamma):
    rnp = np.exp(intercept['intercept']/gamma['gamma'])

    rnp = pd.concat((rnp, ), keys=['rnp'], axis=1)
    return rnp


def format_bambi_ppc(trace, model, df):

    preds = []
    for key, kind in zip(['ll_bernoulli', 'p'], ['pps', 'mean']):
        pred = model.predict(trace, kind=kind, inplace=False) 
        if kind == 'pps':
            pred = pred['posterior_predictive']['chose_risky'].to_dataframe().unstack(['chain', 'draw'])['chose_risky']
        else:
            pred = pred['posterior']['chose_risky_mean'].to_dataframe().unstack(['chain', 'draw'])['chose_risky_mean']
        pred.index = df.index
        pred = pred.set_index(pd.MultiIndex.from_frame(df), append=True)
        preds.append(pred)

    pred = pd.concat(preds, keys=['ll_bernoulli', 'p'], names=['variable'])
    return pred

def summarize_ppc(ppc, groupby=None):

    if groupby is not None:
        ppc = ppc.groupby(groupby).mean()

    e = ppc.mean(1).to_frame('p_predicted')
    hdi = pd.DataFrame(az.hdi(ppc.T.values), index=ppc.index,
                       columns=['hdi025', 'hdi975'])

    return pd.concat((e, hdi), axis=1)


def cluster_offers(d, n=6, key='log(risky/safe)'):
    return pd.qcut(d[key], n, duplicates='drop').apply(lambda x: x.mid)

def plot_ppc(df, ppc, plot_type=1, var_name='ll_bernoulli', level='subject', col_wrap=5, legend=True, **kwargs):

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()

    df = df.copy()

    # Make sure that we group data from (Roughly) same fractions
    if not (df.groupby(['subject', 'log(risky/safe)']).size().groupby('subject').size() < 7).all():
        df['log(risky/safe)'] = df.groupby(['subject'],
                                        group_keys=False).apply(cluster_offers)

    if level == 'group':
        df['log(risky/safe)'] = df['bin(risky/safe)']
        ppc = ppc.reset_index('log(risky/safe)')
        ppc['log(risky/safe)'] = ppc.index.get_level_values('bin(risky/safe)')
        ppc.set_index('log(risky/safe)', append=True, inplace=True)

    if plot_type == 0:
        groupby = ['log(risky/safe)', 'Experiment']
    elif plot_type == 1:
        groupby = ['n_safe', 'Experiment']
    elif plot_type in [2, 3]:
        groupby = ['n_safe', 'log(risky/safe)', 'Experiment']
    elif plot_type in [4]:
        groupby = ['seen_risky_first', 'log(risky/safe)', 'Experiment']
    elif plot_type in [5]:
        groupby = ['seen_risky_first', 'n_safe', 'Experiment']
    elif plot_type in [6]:
        groupby = ['risky_first', 'n_safe', 'Experiment']
    elif plot_type in [7]:
        groupby = ['risky_duration_prop_split', 'n_safe', 'Experiment']
    elif plot_type in [8]:
        groupby = ['risky_duration_prop_split', 'log(risky/safe)', 'Experiment']
    elif plot_type in [9]:
        groupby = ['risky_duration_prop_split', 'n_safe', 'log(risky/safe)', 'Experiment']
    else:
        raise NotImplementedError

    if level == 'group':
        ppc = ppc.groupby(['subject']+groupby).mean()

    if level == 'subject':
        groupby = ['subject'] + groupby

    ppc_summary = summarize_ppc(ppc, groupby=groupby)
    p = df.groupby(groupby)[['chose_risky']].mean()
    ppc_summary = ppc_summary.join(p).reset_index()

    if 'risky_first' in ppc_summary.columns:
        ppc_summary['Order'] = ppc_summary['risky_first'].map({True:'Risky first', False:'Safe first'})

    if 'risky_duration_prop_split' in ppc_summary.columns:
        ppc_summary['Risky dwell time'] = ppc_summary['risky_duration_prop_split'].map({'low':'Short', 'high':'Long'})

    if 'n_safe' in groupby:
        ppc_summary['Safe offer'] = ppc_summary['n_safe'].astype(int)

    ppc_summary['Prop. chosen risky'] = ppc_summary['chose_risky']

    if 'log(risky/safe)' in groupby:
        if level == 'group':
            ppc_summary['Predicted acceptance'] = ppc_summary['log(risky/safe)']
        else:
            ppc_summary['Log-ratio offer'] = ppc_summary['log(risky/safe)']

    if plot_type in [1, 5, 6, 7]:
        x = 'Safe offer'
    else:
        if level == 'group':
            x = 'Predicted acceptance'
        else:
            x = 'Log-ratio offer'


    if plot_type in [0]:
        fac = sns.FacetGrid(ppc_summary,
                            col='subject' if level == 'subject' else None,
                            hue='Experiment',
                            col_wrap=col_wrap if level == 'subject' else None,
                            hue_order=['Symbols', 'Coin clouds'],
                            palette=sns.color_palette()[-3:],
                            **kwargs)


    elif plot_type in [1]:
        fac = sns.FacetGrid(ppc_summary,
                            col='subject' if level == 'subject' else None,
                            hue='Experiment',
                            col_wrap=col_wrap if level == 'subject' else None,
                            hue_order=['Symbols', 'Coin clouds'],
                            palette=sns.color_palette()[-3:],
                            **kwargs)

    elif plot_type == 2:
        fac = sns.FacetGrid(ppc_summary,
                            row='subject' if level == 'subject' else None,
                            col='Safe offer',
                            col_wrap=col_wrap,
                            hue='Experiment',
                            col_order=['Coin clouds', 'Symbols'],
                            palette=sns.color_palette()[-3:],
                            **kwargs)
    elif plot_type == 3:
        fac = sns.FacetGrid(ppc_summary,
                            row='subject' if level == 'subject' else None,
                            col='Experiment',
                            hue='Safe offer',
                            col_order=['Coin clouds', 'Symbols'],
                            palette=sns.color_palette('coolwarm', 6),
                            **kwargs)
    elif plot_type == 4:
        fac = sns.FacetGrid(ppc_summary,
                            row='subject' if level == 'subject' else None,
                            col='Experiment',
                            hue='seen_risky_first',
                            col_order=['Coin clouds', 'Symbols'],
                            **kwargs)
    elif plot_type == 5:
        fac = sns.FacetGrid(ppc_summary,
                            row='subject' if level == 'subject' else None,
                            col='Experiment',
                            hue='seen_risky_first',
                            col_order=['Coin clouds', 'Symbols'],
                            **kwargs)
    elif plot_type == 6:
        fac = sns.FacetGrid(ppc_summary,
                            row='subject' if level == 'subject' else None,
                            col='Experiment',
                            hue='risky_first',
                            col_order=['Coin clouds', 'Symbols'],
                            hue_order=['Short', 'Long'],
                            **kwargs)
    elif plot_type in [7, 8]:
        fac = sns.FacetGrid(ppc_summary,
                            row='subject' if level == 'subject' else None,
                            col='Experiment',
                            hue='Risky dwell time',
                            col_order=['Coin clouds', 'Symbols'],
                            hue_order=['Short', 'Long'],
                            **kwargs)
    elif plot_type == 9:
        if level == 'subject':
            raise NotImplementedError

        fac = sns.FacetGrid(ppc_summary,
                            col='n_safe',
                            row='Experiment',
                            hue='Risky dwell time',
                            col_order=['Coin clouds', 'Symbols'],
                            **kwargs)
    if plot_type in [0, 1,2, 3, 4, 5, 6, 7, 8, 9]:
        fac.map_dataframe(plot_prediction, x=x)
        fac.map(plt.scatter, x, 'Prop. chosen risky')
        fac.map(lambda *args, **kwargs: plt.axhline(.5, c='k', ls='--'))

    if plot_type in [0, 2, 3, 4, 8, 9]:
        if level == 'subject':
            fac.map(lambda *args, **kwargs: plt.axvline(np.log(1./.55), c='k', ls='--'))
        else:
            fac.map(lambda *args, **kwargs: plt.axvline(3.5, c='k', ls='--'))

    
    if legend:
        fac.add_legend()

    return fac

def plot_prediction(data, x, color, y='p_predicted', alpha=.5, **kwargs):
    data = data[~data['hdi025'].isnull()]

    plt.fill_between(data[x], data['hdi025'],
                     data['hdi975'], color=color, alpha=alpha)
    plt.plot(data[x], data[y], color=color)
