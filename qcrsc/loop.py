import numpy as np
from .batch import batch


def loop(DataTable, PeakTable, gamma='default', transform='log', parametric=True, zeroflag=True, remove_outliers=True, remove_batch=True):
    batch_list = np.unique(DataTable.Batch)
    print("Number of Batches : {}".format(len(batch_list)))

    Dlist = []
    Plist = []
    for i in batch_list:
        BatchTable = DataTable[DataTable.Batch == i]
        Data, Peak = batch(BatchTable, PeakTable, gamma=gamma, transform=transform, parametric=parametric, zeroflag=zeroflag, remove_outliers=remove_outliers, remove_batch=remove_batch)
        Dlist.append(Data)
        Plist.append(Peak)

    return Dlist, Plist
