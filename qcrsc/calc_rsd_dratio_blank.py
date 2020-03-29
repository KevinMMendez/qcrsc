import numpy as np
import warnings


def mad(data):
    return np.nanmedian(np.absolute(data - np.nanmedian(data, axis=0)), axis=0)


def calc_rsd_dratio_blank(X, QC, SAM, BLANK, transform, parametric):

    xqc = X[QC == 1]
    xs = X[SAM == 1]
    xb = X[BLANK == 1]

    # Calculate (ignore all nan warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if parametric is True:
            if transform is 'log':
                RSD = 100 * np.nanstd(np.power(10, xqc), ddof=1, axis=0) / np.nanmean(np.power(10, xqc), axis=0)
                Dratio = 100 * np.nanstd(np.power(10, xqc), ddof=1, axis=0) / np.nanstd(np.power(10, xs), ddof=1, axis=0)
                BlankMean = np.nanmean(np.power(10, xb), axis=0) / np.nanmean(np.power(10, xqc), axis=0) * 100
            else:
                RSD = 100 * np.nanstd(xqc, ddof=1, axis=0) / np.nanmean(xqc, axis=0)
                Dratio = 100 * np.nanstd(xqc, ddof=1, axis=0) / np.nanstd(xs, ddof=1, axis=0)
                BlankMean = np.nanmean(xb, axis=0) / np.nanmean(xqc, axis=0) * 100
        else:
            if transform is 'log':
                RSD = 100 * 1.4826 * mad(np.power(10, xqc)) / np.nanmedian(np.power(10, xqc), axis=0)
                Dratio = 100 * mad(np.power(10, xqc)) / mad(np.power(10, xs))
                BlankMean = np.nanmean(np.power(10, xb), axis=0) / np.nanmean(np.power(10, xqc), axis=0) * 100
            else:
                RSD = 100 * 1.4826 * mad(xqc) / np.nanmedian(xqc, axis=0)
                Dratio = 100 * mad(xqc) / mad(xs)
                BlankMean = np.nanmean(xb, axis=0) / np.nanmean(xqc, axis=0) * 100
    return RSD, Dratio, BlankMean
