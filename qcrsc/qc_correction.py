import numpy as np
import pandas as pd
from .loop import loop
from .sampletype_binary import sampletype_binary
from .batch_align import batch_align
from .table_check import table_check


def qc_correction(DataTable, PeakTable, gamma='default', transform='log', zeroflag=True, remove_outliers=True, impute_missing=True, remove_batch=True):

    # future: remove parametric = True, not used as we calc param and non-param
    parametric = True

    # Table check
    table_check(DataTable, print_statement=False)

    # Create (binary) column in DataTable
    DataTable = sampletype_binary(DataTable)

    # Reset peak index
    PeakTable.reset_index(drop=True, inplace=True)

    # Loop through each batch
    Dlist, Plist = loop(DataTable, PeakTable, gamma=gamma, transform=transform, parametric=parametric, zeroflag=zeroflag, remove_outliers=remove_outliers, remove_batch=remove_batch)

    # Align batches
    DataTableX, PeakTableX = batch_align(Dlist, Plist)

    # Impute missing QCs
    if impute_missing == True:
        DataTableX_QC = DataTableX.loc[DataTableX['QCW'] == 1].copy(deep=True)   # Subset with only QCs
        DataTableX_QC[PeakTableX.Name] = DataTableX_QC[PeakTableX.Name].fillna(DataTableX_QC.mean())  # Inpute missing with column means
        for i in DataTableX_QC.index:
            DataTableX.loc[i, :] = DataTableX_QC.loc[i, :]

    print("Done!")

    return DataTableX, PeakTableX
