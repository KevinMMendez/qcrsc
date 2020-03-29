import numpy as np


def table_check(DataTable, print_statement=True):
    data_list = DataTable.columns.values

    if 'Order' not in data_list:
        raise ValueError("Data Table does not contain the required 'Order' column")
    if DataTable.Order.isnull().values.any() == True:
        raise ValueError("Order column cannot contain missing values")
    if len(np.unique(DataTable.Order)) != len(DataTable.Order):
        raise ValueError("Order values must be unique")

    #  IMPORTANT ::::: Sorting all data by INJECTION ORDER
    DataTable = DataTable.sort_values('Order')

    if 'Batch' not in data_list:
        raise ValueError("Data Table does not contain the required 'Batch' column")
    if DataTable.Batch.isnull().values.any() == True:
        raise ValueError("Batch column cannot contain missing values")

    if DataTable.SampleType.isnull().values.any() == True:
        raise ValueError("SampleType column cannot contain missing values")

    # Future: Remove this as SampleType is not used
    temp = np.sort(np.unique(DataTable.SampleType))
    options = ['QC', 'Sample', 'Blank', 'QCT', 'QCW', 'QCB']
    checktemp = np.setdiff1d(temp, options)  # Check for values in temp not in options
    if checktemp.size != 0:
        raise ValueError("Possible sample types are {}. Unknown sample type {} was entered.".format(options, checktemp))

    if print_statement is True:
        print("Data Table is suitable for use with QCRSC")
