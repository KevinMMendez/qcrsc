import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from .csaps import CubicSmoothSpline


def QCRSC(x, t, qc, gamma_range, remove_outliers=True, remove_batch=True):

    TTqc = t[qc == 1]
    XXqc = x[qc == 1]

    if remove_outliers == True:
        q75, q25 = np.percentile(XXqc, [75, 25])
        iqr = q75 - q25
        min_outlier = q25 - 1.5 * iqr
        max_outlier = q75 + 1.5 * iqr
        XXqc[XXqc < min_outlier] = np.nan
        XXqc[XXqc > max_outlier] = np.nan

    Xqc = XXqc.dropna()
    Tqc = TTqc[Xqc.index]

    mpa = np.median(Xqc)
    numQC = len(Tqc)
    dist = []
    for i in range(len(TTqc) - 1):
        dist.append(TTqc.iloc[i + 1] - TTqc.iloc[i] - 1)

    h = np.median(dist)
    epsilon = h**3 / 16

    if numQC < 5:
        # QCs < 5 cannot effectively perform QCspline cross-valiadation
        # setting opt_param to effectively a linear correction.
        type_fit = 'linear'

        cvMse = np.empty(len(gamma_range))
        cvMse[:] = np.nan
        gamma = np.max(gamma_range)
    else:
        type_fit = 'cubic'

        loo = LeaveOneOut()
        cvMse = []
        for i in range(len(gamma_range)):
            p = 1 / (1 + epsilon * 10 ** (gamma_range[i]))
            mse = []
            for train_index, test_index in loo.split(Xqc):
                Tqc_train, Tqc_test = Tqc.iloc[train_index], Tqc.iloc[test_index]
                Xqc_train, Xqc_test = Xqc.iloc[train_index], Xqc.iloc[test_index]
                csaps = CubicSmoothSpline(p=p)
                csaps.fit(Tqc_train, Xqc_train)
                Xqc_pred = csaps.predict(Tqc_test.values.tolist())
                mse.append(mean_squared_error(Xqc_test, Xqc_pred))
            cvMse.append(np.mean(mse))

    cvMse = np.array(cvMse)
    min_cvMse = np.argmin(cvMse)

    if type_fit == 'cubic':
        gamma = gamma_range[min_cvMse]

    p = 1 / (1 + epsilon * 10 ** (gamma))

    try:
        csaps = CubicSmoothSpline(p=p)
        csaps.fit(Tqc, Xqc)
        f = csaps.predict(t.values.tolist())
        zz = x - f
        xx = zz + mpa
    except ValueError:
        # Only 1 QC or less
        if remove_batch == True:
            f = [np.nan] * len(x)
            zz = x
            zz[:] = np.nan
            xx = zz
        else:
            f = [np.nan] * len(x)
            zz = x
            xx = zz

    return xx, f, type_fit, cvMse, gamma, mpa
