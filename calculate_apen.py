import pandas as pd
import numpy as np
import tqdm
import yaml
import functools
import multiprocessing
import os
import frontiers_code as fc

class Data:
    def __init__(self, v):
        self.v = v

def approximate_entropy(y, mnom, rth):
    """Calculate the approximate entropy of a 1D timeseries

    Code adapted from HCTSA package (https://github.com/benfulcher/hctsa)
    """
    n = y.shape[-1] # Length of the TS
    r = rth * np.std(y, axis=-1, ddof=1)
    phi = np.zeros((2, 1))

    ds = []
    drs = []
    for k in range(1, 3):
        m = mnom + k - 1 # Pattern length
        C = np.zeros((n - m + 1, 1))
        x = np.zeros((n - m + 1, m))

        # Form vector sequences x from the time series y
        for i in range(n - m + 1):
            x[i, :] = y[i:i + m]

        ax = np.ones((n - m + 1, m));
        for i in range(n - m + 1):
            for j in range(m):
                ax[:, j] = x[i,j]; 
            d = abs(x - ax)
            if m > 1:
                d = np.max(d, axis=-1) 
            dr = d <= r
            ds.append(d)
            drs.append(dr)
            C[i] = np.sum(dr) / (n - m + 1)
        phi[k - 1] = np.mean(np.log(C))

    return phi[0] - phi[1]


def normalize_apen(apen_val, nc): 
    '''Normalize the ApEn values by the values used in the paper
    '''
    apen_norm = - (apen_val - nc['median']) / (nc['iqr'] / 1.35)
    apen_norm = 1. / (1 + np.exp(apen_norm))
    apen_norm = (apen_norm - nc['min']) / (nc['max'] - nc['min'])
    return apen_norm

def normalized_approximate_entropy(x, nc):
    # ts = scipy.stats.zscore(x[nc['cutoff']:])
    ts = x[nc['cutoff']:]
    apen = approximate_entropy(ts, nc['m'], nc['r'])
    norm_apen = normalize_apen(apen, nc)
    return norm_apen

def main():
    '''To illustrate the use of the normalized_approximate_entropy function we recalculate
    the approximate entropy of the dataset and compare it to the original values.
    '''
    nc = yaml.load(open('./normalization_constants.yaml', 'r'), Loader=yaml.FullLoader)
    AP_EN_ID=847

    dataset_file = 'frontiers_dataset.p'
    if not os.path.isfile(dataset_file):
        df_labels = fc.create_dataframe()
        # Cache for subsequent runs
        df_labels.to_pickle(dataset_file)
    df = pd.read_pickle(dataset_file)

    # Calculate only for unique timeseries
    df = df.groupby('timeseries_id').nth(0)

    p = multiprocessing.Pool()
    f = functools.partial(normalized_approximate_entropy, nc=nc)
    results = list(tqdm.tqdm(p.imap(f, df.timeseries.values, chunksize=1), total=len(df)))
    p.close()

    df['apen'] = results

    # Check that they match the features in the dataset
    results = np.hstack(df.apen.values)
    apen_feat = df.features.apply(lambda k: k[AP_EN_ID])
    max_difference = np.max(np.abs(results - apen_feat))
    print('Maximum difference: %.2e' % max_difference)

if __name__ == "__main__":
    main()
