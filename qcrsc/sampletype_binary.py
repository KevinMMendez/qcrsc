import numpy as np
import pandas as pd


def sampletype_binary(DataTable):
    """
    Create binary columns (['QCW', 'QCB', 'QCT', 'Blank', 'Sample']) from the SampleType column in the DataTable.
    If a sampletype is missing. Create the binary column with only zeros.
    """

    DataTable = DataTable.copy()

    # Create a QCW column based on SampleType (if it doesn't exist)
    try:
        qcw_col = pd.get_dummies(DataTable.SampleType).QCW
    except AttributeError:
        qcw_col = [0] * len(DataTable)
    try:
        DataTable.insert(3, 'QCW', qcw_col)
    except ValueError:
        pass

    # Create a QCW column based on SampleType (if it doesn't exist)
    try:
        qcb_col = pd.get_dummies(DataTable.SampleType).QCB
    except AttributeError:
        qcb_col = [0] * len(DataTable)
    try:
        DataTable.insert(4, 'QCB', qcb_col)
    except ValueError:
        pass

    # Create a QCT column based on SampleType (if it doesn't exist)
    try:
        qct_col = pd.get_dummies(DataTable.SampleType).QCT
    except AttributeError:
        qct_col = [0] * len(DataTable)
    try:
        DataTable.insert(5, 'QCT', qct_col)
    except ValueError:
        pass

    # Create a Sample column based on SampleType (if it doesn't exist)
    sam_col = pd.get_dummies(DataTable.SampleType).Sample
    try:
        DataTable.insert(6, 'Sample', sam_col)
    except ValueError:
        pass

    # Create a Blank column based on SampleType (if it doesn't exist)
    try:
        blank_col = pd.get_dummies(DataTable.SampleType).Blank
    except AttributeError:
        blank_col = [0] * len(DataTable)  # No blanks
    try:
        DataTable.insert(7, 'Blank', blank_col)
    except ValueError:
        pass

    # Create Temporary QC column based on SampleType (if it doesn't exist)
    try:
        qc_col = pd.get_dummies(DataTable.SampleType).QC
    except AttributeError:
        qc_col = [0] * len(DataTable)
    try:
        DataTable.insert(8, 'QC', qc_col)
    except ValueError:
        pass

    # If QC == 1 -> QCW = 1, QCB = 1
    DataTable.loc[DataTable['QC'] == 1, 'QCW'] = 1
    DataTable.loc[DataTable['QC'] == 1, 'QCB'] = 1
    DataTable.drop(columns=['QC'], inplace=True)  # Remove QC Columnn

    return DataTable
