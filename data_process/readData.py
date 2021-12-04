"""
Create A Data Dictionary
"""

import os
import natsort

def read_dataset(data_dir,modality):
    Datalist={}
    for modal in modality:#OCT/OCTA/Label
        Datalist.update({modal:{}})
        ctlist=os.listdir(os.path.join(data_dir, modal))
        ctlist=natsort.natsorted(ctlist)
        for ct in ctlist:
            Datalist[modal].update({ct:{}})
            scanlist=os.listdir(os.path.join(data_dir, modal,ct))
            scanlist=natsort.natsorted(scanlist)
            for i in range(0,len(scanlist)):
                scanlist[i]=os.path.join(data_dir, modal,ct,scanlist[i])
            Datalist[modal][ct]=scanlist
    records = Datalist
    return records

