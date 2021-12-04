import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
from model import CNN
import shutil
import natsort
from options.train_options import TrainOptions
import data_process.readData as readData
import data_process.BatchDataReader as BatchDataReader
import cv2
import scipy.io as scio


def train_net(net,device,foldnum):
    #train setting
    print('fold:',foldnum)
    best_acc=0
    interval=opt.save_interval
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    # Read Data
    print("Start Setup dataset reader")
    Data_records= readData.read_dataset(opt.dataroot,opt.modality_filename)
    datanum=len(list(Data_records[opt.modality_filename[0]]))
    rand_datanum=torch.randperm(datanum)
    fold=scio.loadmat('fold.mat')
    listall=[i for i in range(datanum)]
    testlist= np.squeeze(fold['fold'+str(foldnum)])
    trainlist= list(set(listall).difference(set(testlist)))
    print(trainlist)
    print(testlist)
    print("Setting up dataset reader")
    dataset_reader = BatchDataReader.BatchDatset(opt.dataroot,Data_records, opt.modality_filename,opt.data_size, opt.block_size,opt.batch_size, saveroot=opt.saveroot)
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    #Setting Loss
    criterion = nn.CrossEntropyLoss()
    #Start train
    for itr in range(0, opt.max_iteration):
        net.train()
        train_images, train_annotations = dataset_reader.readbatch_train(trainlist)
        train_images =train_images.to(device=device, dtype=torch.float32)
        train_annotations = train_annotations.to(device=device, dtype=torch.long)
        pred= net(train_images)
        loss = criterion(pred, train_annotations)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if itr % interval == 0:
                torch.save(net.module.state_dict(),
                           os.path.join(model_save_path, f'{itr}.pth'))
                logging.info(f'Checkpoint {itr} saved !')
                net.eval()
                acc=0
                for i,testid in enumerate(testlist):
                    test_images, test_annotations = dataset_reader.readbatch_test(testid)
                    test_images = test_images.to(device=device, dtype=torch.float32)
                    test_annotations = test_annotations.to(device=device, dtype=torch.long)
                    pred = net(test_images)
                    pred_argmax = torch.argmax(pred,1)
                    #print(pred_argmax[0].cpu().detach().numpy())
                    if pred_argmax[0]==test_annotations[0]:
                        acc=acc+1
                acc=acc/(i+1)
                print(itr, loss.item())
                print(acc)
                if acc >= best_acc:
                    best_model_save_path='/home/limingchao/Project/ASDS_CAM/logs/best_model/'+str(foldnum)
                    temp2= f'{itr}.pth'
                    shutil.copy(os.path.join(model_save_path,temp2),os.path.join(best_model_save_path,temp2))
                    model_names = natsort.natsorted(os.listdir(best_model_save_path))
                    if len(model_names) == 5:
                        os.remove(os.path.join(best_model_save_path,model_names[0]))
                    best_acc = acc




if __name__ == '__main__':
    for i in range(1,7):
        #setting logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        #loading options
        opt = TrainOptions().parse()
        #setting GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        #loading network
        net = CNN()
        net=torch.nn.DataParallel(net,[0,1]).cuda()

        #load trained model
        if opt.load:
            net.load_state_dict(
                torch.load(opt.load, map_location=device)
            )
            logging.info(f'Model loaded from {opt.load}')
        #input the model into GPU
        #net.to(device=device)
        try:
            train_net(net=net,device=device,foldnum=i)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)




