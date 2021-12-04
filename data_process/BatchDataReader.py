import os
import numpy as np
import cv2
import h5py
import random
import torch
import scipy.io as scio
from skimage import transform
import matplotlib.pyplot as plt
class BatchDatset:
    def __init__(self, dataroot,records_list,modality,datasize,blocksize,batch_size,saveroot):
        self.dataroot = dataroot
        self.saveroot = saveroot
        self.filelist = records_list
        self.datasize = datasize
        self.blocksize = blocksize
        self.channels = len(modality)
        self.batch_size = batch_size
        self.modality=modality
        self.resizecube=np.zeros((self.channels ,datasize[0],datasize[1],datasize[2]))
        self.images = np.zeros((batch_size, self.channels , blocksize[0], blocksize[1], blocksize[2]))
        self.annotations=np.zeros((batch_size))
        if datasize!=blocksize:
            self.transformkey = 1
        self.cube_num = len(list(self.filelist[modality[0]]))
        self.data = np.zeros((self.channels, blocksize[0], blocksize[1], blocksize[2], self.cube_num), dtype=np.uint8)
        self.read_images()


    def read_images(self):
        if not os.path.exists(os.path.join(self.saveroot,"data.hdf5")):
            print("picking ...It will take some minutes")
            #read label
            self.label=scio.loadmat(os.path.join(self.dataroot,"label.mat"))['label']
            #read data
            modality_num = -1
            for modality in self.filelist.keys():
                ctlist=list(self.filelist[modality])
                modality_num+=1
                ct_num=-1
                for ct in ctlist:
                    ct_num+=1
                    scanlist=list(self.filelist[modality][ct])
                    scan_num=-1
                    for scan in scanlist:
                        scan_num+=1
                        self.resizecube[modality_num, :, :, scan_num]=np.array(cv2.imread(scan,cv2.IMREAD_GRAYSCALE))
                    self.data[modality_num, :, :, :, ct_num]=transform.resize(self.resizecube[modality_num, :, :,:],self.blocksize)
            f= h5py.File(os.path.join(self.saveroot,"data.hdf5"), "w")
            f.create_dataset('data',data=self.data)
            f.create_dataset('label',data=self.label)
            f.close
        else:
            print("found pickle !!!")
            f = h5py.File(os.path.join(self.saveroot,"data.hdf5"), "r")
            self.data = f['data']
            self.label = f['label']
            f.close


    def readbatch_train(self,trainlist):
        for batch in range(0,self.batch_size):
            cubenum=len(trainlist)
            ctrand = random.randint(0, cubenum - 1)
            ctnum=trainlist[ctrand]
            #self.images[batch,:,:,:,:] = self.data[:,:,:,:, ctnum].astype(np.float32)
            self.images[batch, :, :, :, :] = self.augmentation(self.data[:, :, :, :, ctnum].astype(np.float32))
            self.annotations[batch]=self.label[ctnum].astype(np.float32)
            image = torch.from_numpy(self.images)
            label = torch.from_numpy(self.annotations)
        return image, label

    def readbatch_test(self,testid):
        self.images[0,:,:,:,:]= self.data[:,:,:,:, testid].astype(np.float32)
        self.annotations[0]=self.label[testid].astype(np.float32)
        image = torch.from_numpy(self.images)
        label = torch.from_numpy(self.annotations)
        return image, label

    def augmentation(self,image):
        #horizontal flip
        if torch.randint(0, 4, (1,))==0:
            image=image[:,:,::-1,:]
        #Vertical flip
        if torch.randint(0, 4, (1,))==0:
            image = image[ :, :, :, ::-1]
        #rot90
        if torch.randint(0, 4, (1,))==0:
            k=torch.randint(1, 4, (1,))
            image=np.rot90(image,k,axes=(2,3))
        image=np.ascontiguousarray(image)
        return image