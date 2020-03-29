import sys
import numpy as np
import pandas as pd
import multiprocessing
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from .QCRSC import QCRSC
from .calc_rsd_dratio import calc_rsd_dratio


def batch(BatchTable, PeakTable, gamma='default', transform='log', parametric=True, zeroflag=True, remove_outliers=True, remove_batch=True):

    # Make a copy
    BatchTable = BatchTable.copy(deep=True)
    PeakTable = PeakTable.copy(deep=True)

    batch = BatchTable.Batch
    bnum = np.unique(batch)

    # Default gamma_range
    if gamma is 'default':
        gamma = (0.5, 5, 0.2)

    gamma_range = [x / 100.0 for x in range(int(gamma[0] * 100), int(gamma[1] * 100), int(gamma[2] * 100))]

    if len(bnum) > 1:
        raise ValueError("Samples in this Batch are labeled as multiple batches")

    peak_list = PeakTable.Name

    X = BatchTable[PeakTable.Name]
    t = BatchTable.Order
    qc = BatchTable.QCW
    sam = BatchTable.Sample

    if zeroflag == True:
        X = X.replace(0, np.nan)

    if transform is 'log':
        X = np.log10(X)

    G = np.empty(len(peak_list)) * np.nan
    MPA = np.empty(len(peak_list)) * np.nan
    Z = np.empty(X.shape)
    Z[:] = np.nan

    # try loop in parallel
    time.sleep(0.5)  # Sleep for 0.5 secs to finish printing
    num_cores = multiprocessing.cpu_count()
    try:
        qcrsc_loop = Parallel(n_jobs=num_cores)(delayed(batch_loop_parallel)(i, peak_list, X, t, qc, gamma_range, remove_outliers, remove_batch) for i in tqdm(range(len(peak_list)), desc="Batch {}".format(bnum[0])))

        # Append to list
        for i in range(len(qcrsc_loop)):
            Z[:, i] = qcrsc_loop[i][0]
            G[i] = qcrsc_loop[i][1]
            MPA[i] = qcrsc_loop[i][2]
    except:
        print("Error was raised so parallel won't be used.")
        print("Temporary... printing each peak to figure out issue.")
        for i in tqdm(range(len(peak_list)), desc="Batch {}".format(bnum[0])):
            peak_temp = peak_list[i]
            xx, _, _, _, gamma, mpa = QCRSC(X[peak_temp], t, qc, gamma_range, remove_outliers, remove_batch)
            print("Peak {}".format(peak_temp))
            Z[:, i] = xx
            G[i] = gamma
            MPA[i] = mpa

    # Calculate stats (Pb -> export PeakTable)
    Pb = PeakTable
    qc_options = ['QCW', 'QCB', 'QCT']
    # parametric
    for i in qc_options:
        RSDqc, RSDsam, Dratio = calc_rsd_dratio(Z, BatchTable[i], sam, transform, True)
        Pb['RSD_{}'.format(i)] = RSDqc
        Pb['DRatio_{}'.format(i)] = Dratio

    # nonparametric
    for i in qc_options:
        RSDqc, RSDsam, Dratio = calc_rsd_dratio(Z, BatchTable[i], sam, transform, False)
        Pb['RSD*_{}'.format(i)] = RSDqc
        Pb['DRatio*_{}'.format(i)] = Dratio

    # Db -> export DataTable
    Db = BatchTable
    if transform is 'log':
        Z = np.power(10, Z)
        MPA = np.power(10, MPA)
    for i in range(len(peak_list)):
        peak = peak_list[i]
        Db[peak] = Z[:, i]

    # Additional PeakTable stats
    Pb['MPA'] = MPA
    Pb['Blank%Mean'] = np.nan  # Calc Blank%Mean later
    Pb['Gamma'] = G

    return Db, Pb


def batch_loop_parallel(i, peak_list, X, t, qc, gamma_range, remove_outliers, remove_batch):
    peak_temp = peak_list[i]
    xx, _, _, _, gamma, mpa = QCRSC(X[peak_temp], t, qc, gamma_range, remove_outliers, remove_batch)
    return [xx, gamma, mpa]
