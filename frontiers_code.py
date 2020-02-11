"""Morphology data analysis pipeline

This code can be used to reproduce the results reported in the paper
"Deciphering the morphology of motor evoked potentials".

Example:
    To run the code using the default parameters, as they were used in
    the paper, run::

        $ python frontiers_code.py
"""

import argparse
import functools
import multiprocessing
import os
import re
import string
import zipfile

import matplotlib.pyplot as plt
params = {
          'font.size' : 12,
          }
plt.rcParams.update(params)

from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, ShuffleSplit
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.stats import iqr
import yaml

CUTOFF = 70


def process_group(group):
    '''
    Process group with grouping based on having the same timeseries.
    This is a helper function to reconstruct the pandas dataframe from the
    zipped csv-files.
    '''
    cur_id = group.iloc[0].timeseries_id
    with zipfile.ZipFile('dataset.zip') as z:
        ts_file = 'dataset/timeseries/%03i.txt' % cur_id
        feat_file = 'dataset/features/%03i.txt' % cur_id
        ts = np.loadtxt(z.open(ts_file, 'r'))
        feat = np.loadtxt(z.open(feat_file, 'r'))

    rows = []
    for _, row in group.iterrows():
        tmp = row.to_dict()
        tmp['timeseries'] = ts
        tmp['features'] = feat
        rows.append(tmp)

    return rows

def create_dataframe():
    '''
    Recreate the dataset from the csv files
    '''

    with zipfile.ZipFile('dataset.zip') as z:
        df = pd.read_csv(z.open('dataset/dataset.csv', 'r'))

    p = multiprocessing.Pool()
    groups = p.map(process_group, [g for _, g in df.groupby('timeseries_id', as_index=False)], chunksize=50)
    p.close()

    # Flatten list
    rows = sum(groups, [])
    dataset = pd.DataFrame(rows)
    dataset = dataset[['name', 'visitid', 'anatomy_side', 'morph', 'clinic_id', 'timeseries_id', 'timeseries', 'features']]

    return dataset

def plot_distribution_with_samples(df_test, feat_vals, classes, neur_thresholds, df, feat_id):
    """
    Plot the distributions of the chosen feature separately for the normal and abnormal class
    Also display 3 samples at various points in the distribution as an illustration.
    Finally, indicate the thresholds of the various neurologists on the distribution
    """
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 3])

    ## Plot the samples
    axs = [fig.add_subplot(gs[0, s]) for s in range(3)]
    ids = string.ascii_uppercase
    for i, (feat_val, cclass) in enumerate(zip(feat_vals, classes)):
        tmp_df = df[df['vote'] == cclass].copy()
        tmp_df['dist'] = np.abs(feat_val - np.vstack(tmp_df.features.values)[:, feat_id])
        ts = tmp_df.sort_values(by='dist').iloc[0].timeseries[CUTOFF:]
        # Minus as this is how the TS is usually shown to neurologists
        axs[i].plot(-ts, color='black', label='%.2f' % feat_val)
        axs[i].set_title('%.2f' % feat_val)
        at = AnchoredText(ids[i],
                  prop=dict(size=10, weight='bold'), frameon=True,
                  loc=1, # Upper right
                  )
        axs[i].add_artist(at)
        axs[i].set_xticks(ticks=[])
        axs[i].set_yticks(ticks=[])

    ## Plot the distributions
    agreement_count = np.sum(df_test[neur].values, axis=-1)
    agreement_mask = np.logical_or(agreement_count == 2, agreement_count == 3)
    neur_errors = df_test.iloc[agreement_mask, :].apen.values
    abnormal = df.query('vote == 1').apen.values
    normal = df.query('vote == 0').apen.values

    width = 0.05
    ax = fig.add_subplot(gs[1, :])
    plot_gauss(neur_errors, 'neurologist disagreement', '-.', bw=width, ax=ax)
    ab_x, ab_y = plot_gauss(abnormal, 'abnormal', '--', bw=width, ax=ax)
    nor_x, nor_y = plot_gauss(normal, 'normal', '-', bw=width, ax=ax)

    for i, (feat_val, cclass) in enumerate(zip(feat_vals, classes)):
        if cclass == 0:
            x, y = min(list(zip(nor_x, nor_y)), key=lambda k: abs(k[0] - feat_val))
        else:
            x, y = min(list(zip(ab_x, ab_y)), key=lambda k: abs(k[0] - feat_val))
        label = string.ascii_uppercase[i]
        ax.annotate(label,
                    horizontalalignment='right',
                    fontsize=10,
                    fontweight='bold',
                    xy=(x, y), xycoords='data',
                    xytext=(x + (0.1 * (feat_val - 0.5)), y + 0.3), textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="-",
                                    color="0.",
                                    ),
                    )
    OFFSET_LINE = -0.2
    HOR_OFFSET = -0.01
    MIN_SEP = 0.01
    x = [0, 1]
    y = [OFFSET_LINE, OFFSET_LINE]
    neur_line = Line2D(x, y, color='black')
    ax.add_line(neur_line)

    # Check if the thresholds are too close to one another
    # If so, merge the labels for visual clarity
    # WARNING: This merging does not work well for separations
    # that are too large!
    thresh_to_plot = {}
    segs = np.digitize(neur_thresholds, np.linspace(0, 1, int(1 / (MIN_SEP / 2.)) + 1))
    for val in np.unique(segs):
        inds = np.where((segs >= val - 1) & (segs <= val))[0]
        label = '\n'.join(['N%i' % (i+1) for i in inds])
        # label = '[' + ', '.join(['N%i' % (i+1) for i in inds]) + ']'

        new_thresh_value = np.average(np.array(neur_thresholds)[inds])
        thresh_to_plot[label] = new_thresh_value

    ax.scatter(thresh_to_plot.values(), [OFFSET_LINE] * len(thresh_to_plot.values()), marker='o', color='black')

    # It is assumed that the neur_thresholds are in the same order as the 'neur'-list
    for label, thresh in thresh_to_plot.items():
        ax.annotate(label, horizontalalignment='center', fontsize=10,
                    xy=(thresh, OFFSET_LINE), xycoords='data',
                    xytext=(thresh, OFFSET_LINE / 2.),
                    textcoords='data',
                    )
    ax.set_ylim(bottom=2*OFFSET_LINE)
    ax.set_xlabel('Approximate entropy')
    ax.set_ylabel('density')
    # plt.tight_layout()
    plt.legend()
    plt.savefig('fig3_distributions.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

def normalize(x):
    med = np.median(x)
    iqr_x = iqr(x)
    return 1. / (1 + np.exp(- (x - med) / (1.35 * iqr_x)))


def find_peaks_ts(df):
    tmp_df = df.copy()

    # Uncomment to visualize peak detection
    # for ts in tmp_df['timeseries'].values:
    #     tmp = find_peaks(ts[100:], prominence=0.02)[0]
    #     plt.plot(ts[100:])
    #     plt.scatter(tmp, ts[100:][tmp])
    #     plt.show()
    #     print(tmp)

    tmp_df['peaks_count'] = tmp_df['timeseries'].apply(lambda k: find_peaks(k[100:], prominence=0.02)[0].shape[0])

    # Normalize
    all_vals = tmp_df['peaks_count'].values
    all_vals = normalize(all_vals)
    tmp_df['peaks'] = all_vals
    return tmp_df


def plot_gauss(X, label, linestyle, bw=0.05, start=0., end=1., ax=None):
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(
        X.reshape((-1, 1)))
    X_plot = np.linspace(start, end, 101)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    if ax is None:
        plt.plot(X_plot[:, 0], np.exp(log_dens), label=label,
             linestyle=linestyle, color='black')
    else:
        ax.plot(X_plot[:, 0], np.exp(log_dens), label=label,
             linestyle=linestyle, color='black')
    return X_plot[:, 0], np.exp(log_dens)




def plot_two_curves(df, neur, thresh):
    """
    Plot the precision-recall curve and the ROC-curve side-by-side
    """
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))
    plot_curve_metric(
        df=df,
        neur=neur,
        metric=roc_curve,
        y_pred=df['logreg_prob'].values,
        thresh=thresh,
        flip=True,
        xlabel='FPR',
        ylabel='TPR',
        ax=axs[0]
    )
    plot_curve_metric(
        df=df,
        neur=neur,
        metric=precision_recall_curve,
        y_pred=df['logreg_prob'].values,
        thresh=thresh,
        flip=False,
        xlabel='precision',
        ylabel='recall',
        legend_loc=3,
        ax=axs[1]
    )
    plt.tight_layout()
    plt.savefig('fig4_curves_metrics.pdf', bbox_inches='tight')
    # plt.savefig('curves_metrics.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


class flip_train_val(GroupKFold):
    """
    This is the same as GroupKFold, though it will return 1 fold as train and
    n - 1 as test set, which is the opposite of the normal case. We can do this
    because we are fitting a very simple model which does not need much
    data to get a decent fit. This then allows us to have a bigger test set.
    """

    # Flips the train and validation indices so the validation set is the larger of the two
    def split(self, X, y=None, groups=None):
        return ((test, train) for train, test in super(flip_train_val, self).split(X, y, groups))


def classifier(*args, **kwargs):
    """
    A convenience function, to allow changing the type of classifier quickly.
    """
    return LogisticRegression(**kwargs)


def curve_metric(y_true, y_pred, metric):
    """
    Somewhat hacky solution to obtain the values of some metric (e.g. ROC)
    for binary labels. This results in a "curve" consisting of three or
    two points. We're interested in the non-trivial point.
    """
    metric_a, metric_b, _ = metric(y_true, y_pred)
    if metric_a.shape[0] == 3:
        metric_a = metric_a[1]
        metric_b = metric_b[1]
    elif metric_a.shape[0] == 2:
        metric_a = metric_a[0]
        metric_b = metric_b[0]
    return metric_a, metric_b


def plot_curve_metric(df, neur, metric, y_pred, thresh, flip=False, xlabel=None, ylabel=None,
                      legend_loc=0, area_metric=None, filename=None, ax=None):
    '''
    Plot the curve of some metric (precision-recall or ROC in our case) along with the
    performances of the neurologists (which are points on the plot)

    :df: The pandas dataframe containing all the data
    :neur: The list of valid neurologist identifier strings
    :metric: The performance metric to use (e.g. precision-recall)
    :y_pred: The predictions (continuous) of our model
    :thresh: The threshold to convert continuous predictions to binary ones
    :flip: Flip x- and y-axis
    :xlabel: The label for the x-axis
    :ylabel: The label for the y-axis
    :legend_loc: Determines where the legend of the plot is placed, see matplotlib docs
    :area_metric: If provided prints the value of this metric as title (e.g. AUROC)
    :filename: If provided, save plot to file with this name
    :ax: If provided, use this matplotlib axis to plot instead of creating a new fig.
    '''

    if ax is None:
        _, ax = plt.subplots(1, 1)
    y_test_vote = df.vote.values

    loo_scores_neur = leave_one_out_score(
        df, neur, functools.partial(curve_metric, metric=metric))
    loo_scores_log = leave_one_out_score(
        df, neur, functools.partial(curve_metric, metric=metric), y_pred_ext=y_pred > thresh)

    if flip:
        metric_a, metric_b, _ = metric(y_test_vote, y_pred)
    else:
        metric_b, metric_a, _ = metric(y_test_vote, y_pred)
    ax.plot(metric_a, metric_b, color='black', label='5-vote based model')

    vals_neur = np.vstack(list(loo_scores_neur.values()))
    vals_log = np.vstack(list(loo_scores_log.values()))

    if flip:
        first, second = (0, 1)
    else:
        first, second = (1, 0)

    ax.plot(vals_neur[:, first], vals_neur[:, second],
            label='neurologists 3-vote', marker='x', linestyle='', color='black')
    ax.plot(vals_log[:, first], vals_log[:, second],
            label='model 3-vote', marker='^', linestyle='', color='black')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if area_metric is not None:
        plt.title('%.2f' % area_metric(y_test_vote, y_pred))
    ax.legend(loc=legend_loc)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')


def fit_and_score_fold(X_fold_train, y_fold_train, X_fold_test, y_fold_test):
    """
    Fit the model using a single feature, for a specific neurologist, and
    return the auc score. This will be run in parallel
    """
    clf = classifier(class_weight='balanced', n_jobs=1, solver='liblinear')
    clf.fit(X_fold_train, y_fold_train)
    y_pred = clf.predict_proba(X_fold_test)
    return roc_auc_score(y_fold_test, y_pred[:, 1])
    # return average_precision_score(y_fold_test, y_pred[:, 1])


def agreement_fraction(labels_a, labels_b):
    """
    Calculate the fraction of labels on which two sets agree
    """
    return np.sum(labels_a == labels_b) / labels_a.shape[0]


def agreement_mat(df, n_neur, func, include_average=False):
    """
    Returns a matrix containing all pairs of neurologists and
    the corresponding score of agreement, based on the passed function
    e.g. the Cohen kappa score
    """
    mat_morph = (df
                 .sort_values(by=['visitid', 'anatomy_side', 'name'])
                 .morph
                 .values
                 .reshape((-1, n_neur)))

    mat = np.zeros((n_neur, n_neur))
    for i in range(n_neur):
        for j in range(n_neur):
            mat[i, j] = func(mat_morph[:, i], mat_morph[:, j])

    if include_average:
        average = (np.average(mat, axis=0).reshape((-1, 1)) * mat.shape[0] - 1.) / (mat.shape[0] - 1)
        mat = np.hstack([mat, average])
    return mat


def get_intrarater(df, func):
    """
    Get the intrarater score, i.e., the agreement of a rater with him/her/itself
    """
    intra_dict = {}
    n_neur = 2
    for neur, group in df.groupby('name'):
        group = group.groupby(['visitid', 'anatomy_side']
                              ).filter(lambda x: len(x) == 2)
        mat = agreement_mat(group, n_neur, func)
        intra_dict[neur] = mat[0, 1]
    return intra_dict


def get_agreement(df, func, intra=None):
    """
    Get the agreement matrix (see agreement_mat) and set the intra-rater
    to the diagonal (as these have the rather uninformative perfect score
    for each of the neurologist). Create a pandas dataframe from it,
    so it may be easily converted to a LaTex table.
    """
    neur = sorted(df.name.unique())
    n_neur = df.name.nunique()

    mat = agreement_mat(df, n_neur, func, include_average=True)

    if intra is not None:
        for key, value in intra.items():
            index = neur.index(key)
            mat[index, index] = value

    res_df = pd.DataFrame([{name: val for name, val in zip(
        neur + ['average'], row)} for row in mat], index=neur)
    return res_df

def get_only_summaries(df_agree, df_cohen, intra_agree, intra_cohen, include_average=True):
    """Returns a dataframe containing only the summaries of the results
    """
    summary_df = (df_agree[['average']]
                  .merge(df_cohen[['average']],
                         left_index=True,
                         right_index=True)
                  .transpose()
                 )
    summary_df = summary_df.rename({'average_x': 'Agreement fraction', 'average_y': 'Cohen\'s kappa'}, axis=0)
    summary_df['Average'] = np.average(summary_df.values, axis=1)

    # Convert to format %.2f (%.2f) for inter- and intrarater
    intra_agree['Average'] = np.average(list(intra_agree.values()))
    intra_cohen['Average'] = np.average(list(intra_cohen.values()))

    summary_dict = summary_df.to_dict()
    for name, row in summary_dict.items():
        for key, vals in row.items():
            summary_dict[name][key] = '%.2f (%.2f)' % (summary_dict[name][key], intra_agree[name]
                                                       if 'agree' in key.lower()
                                                       else intra_cohen[name])
    new_summary_df = pd.DataFrame(summary_dict)

    return new_summary_df.transpose()

    # for index, row in summary_df.iterrows():
    #     summary_df.loc[index, 'str_perf'] = 
    # summary_df['str_perf'] = 


def leave_one_out_score(df, neur, metric, y_pred_ext=None):
    """
    Create a (n-2)-vote (in our case 3) dataset for each neurologist.
    We take n-2 because in our case n-1 does not always have a consensus
    (e.g. 2-2 votes). If y_pred_ext is provided, we evaluate this set
    of labels on each of the 3-vote sets.
    """
    res_dict = {}
    for name in neur:
        subset = [n for n in neur if n != name]
        scores = []
        for subname in subset:
            subsubset = [n for n in subset if n != subname]
            vote = (np.average(df[subsubset].values,
                               axis=-1) > 0.5).astype(np.int8)
            if y_pred_ext is None:
                score = metric(vote, df[name].values)
            else:
                score = metric(vote, y_pred_ext)
            scores.append(score)
        res_dict[name] = np.average(scores, axis=0)
    return res_dict


def print_metric_score(metric, df, neur, y_test, log_pred):
    """
    Debug function to print the performance of the model according to a 
    given metric. Only used for confusion matrix right now.
    """
    print('%s:' % metric.__name__)
    print('-------------------')

    # Score on the 3-vote set
    perf_neur = leave_one_out_score(df, neur, metric)
    for n, val in perf_neur.items():
        print('%s:\t%i\t%i\n\t%i\t%i' % (n, val[0, 0], val[0, 1],
                                            val[1, 0], val[1, 1]))
        print('-------------------')
    # print(perf_neur)
    perf_log = leave_one_out_score(df, neur, metric, y_pred_ext=log_pred)
    # print(np.average([value for value in perf_log.values()]),
    #       np.std([value for value in perf_log.values()]))

    # Score on the 5-vote set
    val = metric(y_test, y_pred > thresh)
    print('%s:\t%i\t%i\n\t%i\t%i' % ('model', val[0, 0], val[0, 1],
                                              val[1, 0], val[1, 1]))
    print('-------------------')


def labeled_confusion_matrix(y_true, y_pred):
    """
    Wrapper around the confusion matrix so it contains labels
    for each of the cells
    """
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm,
                      columns=['neg pred', 'pos pred'],
                      index=['neg real', 'pos real'])
    return df


def generate_performance_table(test_df, neur, thresh, columns, metrics=None):
    """
    Create the main results table for the paper
    """
    fmt_str = '%.2f (%.2f)'

    log_pred = test_df['logreg_prob'].values
    y_true = test_df['vote'].values
    if metrics is None:
        metrics_names = ['AUC', 'AP', 'F1',
                         'Accuracy', 'Precision', 'Recall', 'Cohen']
        metrics = [roc_auc_score, average_precision_score, f1_score,
                   accuracy_score, precision_score, recall_score, cohen_kappa_score]
        binary = [False, False, True, True, True, True, True]
    table_dict = []
    # columns = ['model [5-vote] (std)',
    #            'model [3-vote] (std)', 'neurologists [3-vote] (std)']
    for met, is_bin in zip(metrics, binary):
        results_dict_neur = leave_one_out_score(
            test_df, neur, met) if is_bin else np.nan
        results_dict_log = leave_one_out_score(
            test_df, neur, met, y_pred_ext=log_pred > thresh if is_bin else log_pred)

        average_score_neur = np.average(
            list(results_dict_neur.values())) if is_bin else np.nan
        std_score_neur = np.std(
            list(results_dict_neur.values())) if is_bin else np.nan
        average_score_log = np.average(list(results_dict_log.values()))
        std_score_log = np.std(list(results_dict_log.values()))

        score_5_vote = met(y_true, log_pred > thresh if is_bin else log_pred)

        table_dict.append({columns[2]: fmt_str % (average_score_neur, std_score_neur)
                                       if is_bin
                                       else 'N/A',
                           columns[1]: fmt_str % (average_score_log, std_score_log),
                           columns[0]: score_5_vote})
    perf_df = pd.DataFrame(table_dict, index=metrics_names)
    return perf_df

def get_help(key, config_file):
    '''
    Returns the comment associated with the key in the config file
    '''
    val_line = -1
    found = False
    lines = [l.strip() for l in open(config_file, 'r').readlines()]
    for i, l in enumerate(lines):
        if re.match('^\'%s\':.*' % key, l):
            val_line = i
            found = True
            # Find description lines
            descr_lines = []
            for j in range(val_line - 1, 0, -1):
                if re.match('^#.*', lines[j]):
                    descr_lines.insert(0, lines[j].strip('# '))
                else:
                    break
            description = ' '.join(descr_lines)

    if found:
        return description
    else:
        return ''



if __name__ == "__main__":

    # Load the configuration file
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    # Allow override with command line arguments
    parser = argparse.ArgumentParser()
    for key, val in config.items():
        if isinstance(val, bool):
            parser.add_argument('--%s' % key, action='store_%s' % str(not val).lower(),
                                help=get_help(key, config_file='config.yaml'))
        else:
            parser.add_argument('-%s' % key, type=type(val), default=val,
                                help=get_help(key, config_file='config.yaml'))
    args = vars(parser.parse_args())

    TOP_N_FEAT = args['top_n_feat']

    if not os.path.isfile(args['annot_file']):
        if args['annot_file'] == 'frontiers_dataset.p':
            df_labels = create_dataframe()
            # Cache for subsequent runs
            df_labels.to_pickle('frontiers_dataset.p')
        else:
            raise Exception('Could not locate dataset file: %s' % args['annot_file'])
    else:
        df_labels = pd.read_pickle(args['annot_file'])

    neur = sorted(df_labels.name.unique().tolist())

    # Determine the agreement between the neurologists
    intra_dict_cohen = get_intrarater(df_labels, func=cohen_kappa_score)
    intra_dict_agree = get_intrarater(df_labels, func=agreement_fraction)

    # Remove doubles used for intra-rater
    if args['discard_all_doubles']:
        # Simply remove all visits that are double,
        # a single visit has 4 measurements
        df_labels = df_labels.groupby(
            ['visitid', 'name']).filter(lambda x: len(x) == 4)
    else:
        # Of the doubles used for the intrarater, use only those that have the same label
        # If one of the neurologists was not consistent, we remove the TS from the dataset
        df_labels['consistent_label'] = (df_labels
                                     .groupby(['visitid', 'name', 'anatomy_side'])
                                     .morph
                                     .transform(lambda x: x.nunique() == 1))
        df_labels = (df_labels
                     .groupby(['visitid', 'anatomy_side'])
                     .filter(lambda x: all(x.consistent_label.values)))
        df_labels = (df_labels
                     .groupby(['visitid', 'name', 'anatomy_side'], as_index=False)
                     .nth(0))
    print('# of TS remaining after removing doubles: %i' % (len(df_labels) / len(neur)))

    # Remove TS with value 3 (bad data)
    with_bad_data_count = len(df_labels)
    df_labels = (df_labels
                 .groupby(['visitid', 'anatomy_side'])
                 .filter(lambda x: (x['morph'] != 3).all()))
    print('# Removed due to bad data: %i / %i' %
            (((with_bad_data_count - len(df_labels)) / len(neur)), with_bad_data_count / len(neur)))

    # Generate agreement tables
    df_agree = get_agreement(
        df_labels, func=agreement_fraction, intra=intra_dict_agree)
    df_cohen = get_agreement(
        df_labels, func=cohen_kappa_score, intra=intra_dict_cohen)
    df_summary = get_only_summaries(df_agree, df_cohen, 
            intra_agree=intra_dict_agree, intra_cohen=intra_dict_cohen)

    # Write table 2
    print(df_summary.to_latex(float_format='%.2f'), file=open('table_2.txt', 'w'))

    ## Full agreement matrices
    # print(df_agree.to_latex(float_format='%.2f'))
    # print(df_cohen.to_latex(float_format='%.2f'))

    # For convenience, create a column for each neurologist containing their label
    df_tmp = (df_labels
                 .pivot_table(columns='name',
                              values='morph',
                              index=['visitid', 'anatomy_side'])
                 .reset_index()
                 )
    df_ts_features = df_labels.groupby(['visitid', 'anatomy_side']).nth(0)
    df_labels = pd.merge(df_tmp, df_ts_features, how='inner', on=['visitid', 'anatomy_side'])
    
    # Set labels values from 1 and 2 to 0 and 1
    # Originally, 1 is normal, 2 is abnormal and 3 is bad data
    # We'll use 0 and 1 for normal and abnormal for convenience
    df_labels[neur] = df_labels[neur].apply(lambda x: x - 1)

    # Vote - Create new column with the majority vote
    df_labels['vote'] = (np.average(
        df_labels[neur].values, axis=-1) > 0.5).astype(np.int8)

    # Print the class imbalance
    n_normal = len(df_labels.query('vote == 0'))
    n_abnormal = len(df_labels.query('vote == 1'))
    frac_normal = n_normal / (n_normal + n_abnormal)
    frac_abnormal = n_abnormal / (n_normal + n_abnormal)
    print('Class imbalance: %i / %i (%.2f / %.2f)' % (
                                        n_normal,
                                        n_abnormal,
                                        frac_normal,
                                        frac_abnormal
                                        ))

    df = df_labels

    # Open the description file
    with open('./feature_names.txt', 'r') as f:
        descr = [x.strip() for x in f.readlines()]
    assert len(descr) == df.iloc[0].features.shape[0]

    df.reset_index(inplace=True, drop=True)
    df.sort_values(by=['visitid', 'anatomy_side'], inplace=True)

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%
    In the first stage we will look for the best possible feature,
    based on cross-validation on the training set. This is done
    for each neurologist separately
    %%%%%%%%%%%%%%%%%%%%%%%%
    '''
    # Create train and test set
    gkf = GroupKFold(n_splits=args['n_splits_train_test'])
    
    # Adds a column with peak count (normalized)
    df = find_peaks_ts(df)

    X = np.vstack(df.features.values)
    X = np.hstack([X, df['peaks'].values.reshape((-1, 1))])
    # Add number of peaks to the description
    descr.append('Number of peaks')
    y = df[neur].values

    # Ensure the same patient does not occur in both train and test set
    train_index, test_index = list(gkf.split(df.index, groups=df.clinic_id.values))[0]

    X_train = X[train_index, :]
    y_train = y[train_index, :]
    groups_train = df.clinic_id.values[train_index]
    train_indices = df.index[train_index]

    if not args['skip_first_phase']:
        # Cross-validation
        gkf_fold = flip_train_val(n_splits=args['n_crossval_splits'])
        # Uncomment to use standard GroupKFold
        # gkf_fold = GroupKFold(n_splits=args['n_crossval_splits'])

        cross_results = {}
        best_features = []
        best_features_dict = {}
        best_features_indiv = {}
        for train_fold_index, test_fold_index in gkf_fold.split(X_train, y_train, groups_train):
            X_fold_train = X_train[train_fold_index, :]
            y_fold_train = y_train[train_fold_index, :]
            X_fold_test = X_train[test_fold_index, :]
            y_fold_test = y_train[test_fold_index, :]

            for i, _ in enumerate(neur):
                y_fold_train_neur = y_fold_train[:, i]
                y_fold_test_neur = y_fold_test[:, i]


                sets = ((X_fold_train[:, q].reshape((-1, 1)), y_fold_train[:, i], X_fold_test[:, q].reshape((-1, 1)), y_fold_test[:, i]) for q in range(X_fold_train.shape[-1]))

                p = multiprocessing.Pool()
                results = p.starmap(fit_and_score_fold, sets, chunksize=200)
                p.close()

                cross_results.setdefault(neur[i], []).append(results)

        top_n_table = {}
        # Extract the best features for each neurologist
        neur_anom = {name: 'N%i' % (i + 1) for i, name in enumerate(neur)}

        PEAKS_FEAT = -1
        for name in neur:
            local_best_features = np.argsort(np.average(
                np.array(cross_results[name]), axis=0))[::-1][:TOP_N_FEAT]
            ranks = np.argsort(np.average(
                np.array(cross_results[name]), axis=0))[::-1].argsort()
            best_features_scores = np.sort(np.average(
                np.array(cross_results[name]), axis=0))[::-1][:TOP_N_FEAT]
            unsorted_scores = np.average(np.array(cross_results[name]), axis=0)

            for i in range(TOP_N_FEAT):
                top_n_table.setdefault(i+1, {})['Feature %s' % neur_anom[name]] = descr[local_best_features[i]]
                top_n_table.setdefault(i+1, {})['AUC %s' % neur_anom[name]] = '%.3f' % best_features_scores[i]
                top_n_table.setdefault(i+1, {})['Rank %s' % neur_anom[name]] = i + 1

            # Manually add the peaks feature
            top_n_table.setdefault(TOP_N_FEAT + 1, {})['Feature %s' % neur_anom[name]] = descr[PEAKS_FEAT]
            top_n_table.setdefault(TOP_N_FEAT + 1, {})['AUC %s' % neur_anom[name]] = '%.3f' % unsorted_scores[PEAKS_FEAT]
            top_n_table.setdefault(TOP_N_FEAT + 1, {})['Rank %s' % neur_anom[name]] = ranks[PEAKS_FEAT]

            best_features.append({'name': name, 'features': [descr[feat] for feat in local_best_features]})
            best_features_indiv[name] = [descr[feat] for feat in local_best_features]

            for feat in local_best_features:
                best_features_dict.setdefault(feat, []).append(
                    np.array(cross_results[name])[:, feat])

        # Print Table 4
        # Split into 3 separate tables
        top_ten_df = pd.DataFrame(top_n_table).T
        column_order = sum([
                [x, y, z] for x, y, z in zip(
                    sorted([key for key in top_ten_df.keys() if 'Rank' in key]),
                    sorted([key for key in top_ten_df.keys() if 'AUC' in key]),
                    sorted([key for key in top_ten_df.keys() if 'Feature' in key])
                )
            ],
            [])
        for i in range(3):
            with pd.option_context("max_colwidth", 1000):
                s = top_ten_df[column_order].iloc[:, i * 6: (i*6 + 6)].to_latex(index=False)
            # Bold the Approximate Entropy
            s = re.sub(r'(ApEn[\w\\\_]+)', r'\\textbf{\1}', s)
            s = re.sub(r'(.*%s.*)' % descr[-1], r'\\hline\n\1', s)
            if i == 0:
                print(s, file=open('table_4.txt', 'w'))
            else:
                print(s, file=open('table_4.txt', 'a'))

        # Score the features across the neurologists
        candidates = []
        for feat, values in best_features_dict.items():
            candidates.append({'name': descr[feat], # Descriptive name
                               'id': feat, # Index of the feature
                               'av_perf': np.average(values), # Average performance
                               'std_perf': np.std(values), # Standard deviation of the performance
                               'occurrence': len(values)}) # For how many neurologists does it occur in the top n?
        # Print only features with high occurrence
        max_occurrence = max([x['occurrence'] for x in candidates])
        candidates = filter(lambda x: x['occurrence'] >= max_occurrence - 1,
                            candidates)
        # Convert to pandas df for printing
        print_df = pd.DataFrame(candidates)
        print(print_df)

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%
    From this we find that for the current split of train/test there are 2 features that occur in the
    top 10 of 4 of the neurologists, namely CP_ML_StepDetect_l1pwc_10.rmsoff and ApEn2_02.
    Judging from cross-validation alone, CP_ML_StepDetect_l1pwc_10.rmsoff seems to be the best feature.
    But upon closer inspection we find that ApEn2_02 en ApEn2_01 together occur in the top ten of all
    neurologists. So we opted for this feature instead. The performance difference between CP_ML_StepDetect_l1pwc_10.rmsoff
    and ApEn2_02 is negligible anyway.
    %%%%%%%%%%%%%%%%%%%%%%%%
    '''

    # Based on the analysis above, choose the ApEn2_02 feature.
    FEAT = args['chosen_feature'] # For the final results, we used 847
    print('Chosen feature: %s' % descr[FEAT])


    '''
    %%%%%%%%%%%%%%%%%%%%%%%%
    Now let's have a look at the performance on the held-out test set.
    We'll train on the training set with labels based on the majority
    vote.
    %%%%%%%%%%%%%%%%%%%%%%%%
    '''
    df['apen'] = np.vstack(df.features)[:, FEAT]
    y_train_vote = df.vote.values[train_index]

    clf = classifier(class_weight='balanced', solver='liblinear')
    clf.fit(X_train[:, FEAT].reshape((-1, 1)), y_train_vote)

    y_train_pred = clf.predict_proba(X_train[:, FEAT].reshape((-1, 1)))[:, 1]

    if args['prec_rec_workpoint']:
        # Find precision-recall working point, i.e., the point where precision is approximately
        # the same as the recall
        prec, rec, thresholds = precision_recall_curve(y_train_vote, y_train_pred)
        thresh = thresholds[np.argmin(np.abs(prec - rec))]
    else:
        # Find ROC working point, i.e., the point where tpr is approximately the same as the 1 - fpr
        fpr, tpr, thresholds = roc_curve(y_train_vote, y_train_pred)
        thresh = thresholds[np.argmax(tpr - fpr)]

    print('Chosen threshold: %.3f' % thresh)

    # The FEAT value for which this occurs
    w = clf.coef_[0][0]
    c = clf.intercept_
    # WARNING: This is specific for the logreg model!
    morph_thresh = (np.log(- thresh / (thresh - 1.)) - c) / w

    print('morphology threshold: %.3f' % morph_thresh)

    indiv_thresh = []
    # Individual thresholds for each neurologist
    for name in neur:
        fpr, tpr, thresholds = roc_curve(df.loc[df.index[train_index]][name].values, X_train[:, FEAT].reshape((-1, 1)))
        thresh_loc = thresholds[np.argmax(tpr - fpr)]
        indiv_thresh.append(thresh_loc)

    print('Neurologist\'s thresholds: %s (%.3f)' % (str(indiv_thresh), np.average(indiv_thresh)))

    # Table columns
    columns = ['model [5-vote] (std)',
               'model [3-vote] (std)', 'neurologists [3-vote] (std)']
    label_5vote = [x for x in columns if '5-vote' in x][0]

    if args['subsample']:
        ss = ShuffleSplit(n_splits=args['n_subsample_splits'], test_size=0.8, random_state=321)
        subset_generator = [test_subset
                            for _, test_subset in ss.split(test_index)]
    else:
        subset_generator = [np.arange(test_index.shape[0])]

    votes_5_results = []
    for test_subset_indices in subset_generator:

        subset_indices = test_index[test_subset_indices]
        test_df = df.loc[df.index[subset_indices]].copy()

        X_test = X[subset_indices, :]
        # y_test = y[test_index, :] -> individual labels not used in test set

        y_pred = clf.predict_proba(X_test[:, FEAT].reshape((-1, 1)))[:, 1]

        test_df['logreg_prob'] = y_pred
        test_df['logreg'] = y_pred > thresh

        pt = generate_performance_table(test_df, neur, thresh, columns, metrics=None)
        votes_5_results.append(pt[label_5vote].values)

    test_df = df.loc[df.index[test_index]].copy()
    X_test = X[test_index, :]
    y_pred = clf.predict_proba(X_test[:, FEAT].reshape((-1, 1)))[:, 1]

    # Plot Figure 3
    # For the samples shown in the paper, we'll consider these values for the apen and
    # the corresponding classifications by the neurologists (5-vote)
    feat_vals = [0.2, 0.5, 0.9]
    classes = [0, 1, 1]
    plot_distribution_with_samples(test_df, feat_vals, classes, indiv_thresh, df, FEAT)

    test_df['logreg_prob'] = y_pred
    test_df['logreg'] = y_pred > thresh
    av_std = list(zip(np.average(votes_5_results, axis=0),
                      np.std(votes_5_results, axis=0)))
    # We will use the performance on the full test set for the final results, but one
    # could also use the average across the subsampled test set. Uncomment to print
    # these performances
    # print('\n'.join(['%.2f (%.2f)' % (av, std) for av, std in av_std]))

    # Generate Table 3
    pt = generate_performance_table(test_df, neur, thresh, columns, metrics=None)
    pt[label_5vote] = ['%.2f (%.2f)' % (av, std) for av, std in zip(pt[label_5vote].values, np.array(av_std)[:, 1])]
    print(pt.iloc[:, [2, 1, 0]].to_latex(float_format='%.2f'), file=open('table_3.txt', 'w'))

    # Print the confusion matrices
    # print_metric_score(confusion_matrix, test_df,
    #                          neur, test_df.vote.values, y_pred > thresh)

    # Plot Figure 4
    plot_two_curves(test_df, neur, thresh)
