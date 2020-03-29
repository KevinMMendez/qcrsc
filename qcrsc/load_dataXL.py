import pandas as pd
import numpy as np
import os.path
from .table_check import table_check


def load_dataXL(filename, DataSheet, PeakSheet):
    """
    This function loads and validates the DataFile and PeakFile DataFile:
    Metabolite IDs must start with 'M' ... best to use M1 M2 M3 M4 etc.
    Remaining columns are assumed to be user specific meta data and are ignored.
    Peak File: The first columns should contain the Peak Label matching the DataFile (M1 M2 .. )
    The remaining columns can contain anything you like. Statistics will be added to this "table"
    """

    if os.path.isfile(filename) is False:
        raise ValueError("{} does not exist.".format(filename))

    if not filename.endswith('.xlsx'):
        raise ValueError("{} should be a .xlsx file.".format(filename))

    ### LOAD PEAK DATA ###
    print("Loadings PeakFile: {}".format(PeakSheet))
    PeakTable = pd.read_excel(filename, sheet_name=PeakSheet)
    peak_list = PeakTable.Name

    peaks = np.unique(peak_list)
    if len(peak_list) != len(peaks):
        raise ValueError("All Peak Names in {} should be unique.".format(PeakSheet))

    ### LOAD DATA TABLE ###
    print("Loadings DataFile: {}".format(DataSheet))
    DataTable = pd.read_excel(filename, sheet_name=DataSheet)

    data_list = DataTable.columns.values
    temp = np.intersect1d(data_list, peak_list)
    if len(temp) != len(peak_list):
        raise ValueError("The Peak Names in {} should exactly match the Peak Names in {}. ( M1, M2 etc. ) Remember that all Peak Names should be unique.".format(PeakSheet, DataSheet))

    # Replace '-99', '.', ' ' with np.nan
    DataTable = DataTable.replace(-99, np.nan)
    DataTable = DataTable.replace('.', np.nan)
    DataTable = DataTable.replace(' ', np.nan)

    ### DO QCRSC TABBLE CHECK ###
    table_check(DataTable, print_statement=True)

    print("TOTAL SAMPLES: {} TOTAL PEAKS: {}".format(len(DataTable), len(peak_list)))
    print("Done!")

    return DataTable, PeakTable
