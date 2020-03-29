import numpy as np
import pandas as pd


def control_limits(X, QC, SAM, key, value, transform=False):

    xqc = X[QC == 1]
    xs = X[SAM == 1]

    # RSD = 100 * std_qc / mean_qc -> std_qc = RSD * mean_qc / 100
    if key == 'RSD':
        rsd = value
        mean_qc = np.nanmean(xqc)
        std_qc = rsd * mean_qc / 100
    elif key == 'Dratio':
        std_sam = np.nanstd(xs, ddof=1)
        std_qc = value * std_sam / 100
    else:
        raise ValueError('control_limit can only be RSD or Dratio')

    # conversion to 'log' based on ratio of true vs. log
    if transform == 'log':
        if key == 'RSD':
            true_val = 100 * np.nanstd(np.power(10, xqc), ddof=1, axis=0) / np.nanmean(np.power(10, xqc), axis=0)
            log_val = 100 * np.nanstd(xqc, ddof=1, axis=0) / np.nanmean(xqc, axis=0)
        else:
            true_val = 100 * np.nanstd(np.power(10, xqc), ddof=1, axis=0) / np.nanstd(np.power(10, xs), ddof=1, axis=0)
            log_val = 100 * np.nanstd(xqc, ddof=1, axis=0) / np.nanstd(xs, ddof=1, axis=0)
        std_qc = std_qc * (log_val / true_val)

    # control_limit -> mean +/- 2std
    low = np.nanmean(xqc) - 2 * std_qc
    upp = np.nanmean(xqc) + 2 * std_qc

    return low, upp
