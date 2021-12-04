import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_class=2, act='relu'):
        super(CNN, self).__init__()
        self.conv1 = Conv3d(1, 64)
        self.pool1 = nn.MaxPool3d(4)
        self.conv2 = Conv3d(64, 128)
        self.pool2 = nn.MaxPool3d(4)
        self.conv3 = Conv3d(128, 256)
        self.pool3 = nn.MaxPool3d(4)
        self.conv4 = Conv3d(256, 512)
        self.block = nn.Sequential(
            nn.AvgPool3d((4,2,2), 1),
            #torch.nn.Dropout(0.9)
        )
        self.out = Classifier(512, n_class)

    def forward(self, x):  # x:[4,1,256,128,128]
        x1 = self.conv1(x)
        x1p= self.pool1(x1)# x:[4,64,64,32,32]
        x2 = self.conv2(x1p)
        x2p = self.pool1(x2)# x:[4,256,16,8,8]
        x3 = self.conv3(x2p)
        x3p = self.pool1(x3)# x:[4,512,4,2,2]
        x4 = self.conv4(x3p)
        x4p = self.block(x4)
        x4p = x4p.view(x4p.shape[0], -1)
        output = self.out(x4p)  # [4,1]
        return output




class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3d, self).__init__()  # equivalent to nn.Module.__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x=self.conv(x)
        return x



class Classifier(nn.Module):
    def __init__(self, inChans, n_labels):
        super(Classifier, self).__init__()
        self.final_conv = nn.Linear(inChans, n_labels)

    def forward(self, x):
        out = self.final_conv(x)
        return out
