import torch
import torch.nn as nn
import logging
import sys
import os
import model
import numpy as np
import scipy.misc as misc
from options.test_options import TestOptions
import natsort
from model import CNN
import data_process.readData as readData
import data_process.BatchDataReader as BatchDataReader
from gradcam import GradCam
import scipy.io as scio
from skimage import transform
import cv2

def test_net(net,device,foldnum):

    net.eval()
    print(foldnum)
    acc = 0
    Data_records= readData.read_dataset(opt.dataroot,opt.modality_filename)
    dataset_reader = BatchDataReader.BatchDatset(opt.dataroot,Data_records, opt.modality_filename,opt.data_size, opt.block_size,1, saveroot=opt.saveroot)
    grad_cam = GradCam(model=net, feature_module=net.conv2, target_layer_names=["conv"])
    fold=scio.loadmat('fold.mat')
    testlist= np.squeeze(fold['fold'+str(foldnum)])
    for i in testlist:
        #print(i)
        test_images, test_annotations = dataset_reader.readbatch_test(i)
        test_images = test_images.to(device=device, dtype=torch.float32)
        test_annotations = test_annotations.to(device=device, dtype=torch.long)
        Map = grad_cam(test_images)
        Map_write(Map,i)
        pred = net(test_images)
        pred_argmax = torch.argmax(pred, 1)
        if pred_argmax[0] == test_annotations[0]:
            acc = acc + 1
    acc = acc / 20
    print(acc)

def Map_write(map,i):
    os.mkdir('/home/Data/OCT-ASDS/CAM/OCT/'+str(20001+i))
    map_resize=transform.resize(map,(640,304,304))
    map_resize=map_resize*255
    for j in range(304):
        cv2.imwrite('/home/Data/OCT-ASDS/CAM/OCT/'+str(20001+i)+'/'+str(j+1)+'.bmp',map_resize[:,:,j])
    #scio.savemat('/home/Data/OCT-ASDS/CAM/'+str(20001+i)+'.mat', {'CAM': map})

if __name__ == '__main__':
    for i in range(1, 7):
        #setting logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        #loading options
        opt = TestOptions().parse()
        #setting GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        #loading network
        net = CNN()
        restore_path = os.path.join(opt.saveroot, 'best_model',str(i), natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model',str(i))))[-1])
        net.load_state_dict(
            torch.load(restore_path, map_location=device)
        )
        #input the model into GPU
        net.to(device=device)
        try:
            test_net(net=net,device=device,foldnum=i)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
