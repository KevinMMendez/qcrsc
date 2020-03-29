import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
from .calc_rsd_dratio import calc_rsd_dratio


def batch_align(DataTables, PeakTables, transform='log', parametric=True, QC='QC'):

    # Make a copy or it will overwrite these list of tables
    DataTables = deepcopy(DataTables)
    PeakTables = deepcopy(PeakTables)

    num_batches = len(DataTables)
    peak_list = PeakTables[0].Name

    # check the Peak Tables are consistent
    for i in range(1, num_batches):
        temp = PeakTables[i].Name
        if any(temp == peak_list) is False:
            raise ValueError("The peak names in Batch {} PeakTable should exactly match the peak names in Batch 1. They do not!".format(i))

    # check the Data Tables are consistent
    col_list = np.array(list(DataTables[0]))
    for i in range(1, num_batches):
        temp = np.array(list(DataTables[i]))
        if any(temp == col_list) is False:
            raise ValueError("The column headers the in Batch {} DataTable should exactly match the column headers in Batch 1. They do not!".format(i))

    # Use QCW
    QCx = 'QCW'

    PeakTableX = PeakTables[0]
    # Subtract MPA from all Batch X data & at Batch stats to PeakTable

    mpa_array = []
    Dlist = []

    for i in range(num_batches):
        DataTable = DataTables[i].copy(deep=True)
        X = DataTable[peak_list]

        if transform is 'log':
            X = np.log10(X)

        Xqcb = DataTable.QCB
        Xqc_between = X[Xqcb == 1]
        MPA = np.median(Xqc_between)  # Calc MPA using QCB

        X = X - np.array(MPA)

        for j in peak_list:
            DataTable[j] = X[j]

        Dlist.append(DataTable)
        mpa_array.append(MPA)

        qc_options = ['QCW', 'QCB', 'QCT']
        PeakTableX["B{}_Gamma".format(i + 1)] = PeakTables[i]["Gamma"]
        for j in qc_options:
            PeakTableX["B{}_RSD_{}".format(i + 1, j)] = PeakTables[i]["RSD_{}".format(j)]
            PeakTableX["B{}_DRatio_{}".format(i + 1, j)] = PeakTables[i]["DRatio_{}".format(j)]
            PeakTableX["B{}_RSD*_{}".format(i + 1, j)] = PeakTables[i]["RSD*_{}".format(j)]
            PeakTableX["B{}_DRatio*_{}".format(i + 1, j)] = PeakTables[i]["DRatio*_{}".format(j)]

    DataTableX = Dlist[0]
    for i in range(1, num_batches):
        DataTableX = pd.concat([DataTableX, Dlist[i]])

    MPA = np.nanmedian(mpa_array, axis=0)

    X = DataTableX[peak_list]
    X = X + np.array(MPA)

    if transform is 'log':
        X = np.power(10, X)
        MPA = np.power(10, MPA)

    for j in peak_list:
        DataTableX[j] = X[j]

    PeakTableX['MPA'] = MPA
    PeakTableX.drop(columns=['Gamma'], inplace=True)

    for i in qc_options:
        RSDqc, RSDsam, Dratio = calc_rsd_dratio(X, DataTableX[i], DataTableX['Sample'], False, True)
        PeakTableX['RSD_{}'.format(i)] = RSDqc
        PeakTableX['DRatio_{}'.format(i)] = Dratio

    for i in qc_options:
        RSDqc, RSDsam, Dratio = calc_rsd_dratio(X, DataTableX[i], DataTableX['Sample'], False, False)
        PeakTableX['RSD*_{}'.format(i)] = RSDqc
        PeakTableX['DRatio*_{}'.format(i)] = Dratio

    # Blank peak area ratio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blank_bpar = DataTableX[DataTableX.Blank == 1][PeakTableX.Name]
        blank_bpar_param = np.nanmean(blank_bpar, axis=0)
        qc_bpar = DataTableX[DataTableX.QCW == 1][PeakTableX.Name]
        qc_bpar_param = np.nanmean(qc_bpar, axis=0)
        PeakTableX['Blank%Mean'] = blank_bpar_param / qc_bpar_param * 100

    print('{} batches corrected and concatenated'.format(num_batches))
    print('Final data set: {} samples and {} metabolites'.format(len(DataTableX), len(peak_list)))
    return DataTableX, PeakTableX
