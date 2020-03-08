from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile
import cv2

from .dataset import ImageDataset
from .matlab_cp2tform import get_similarity_transform_for_cv2
from . import net_sphere

from progress.bar import Bar


class SphereFace(nn.Module):
    def __init__(self,classnum=10574,feature=True):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = AngleLinear(512,self.classnum)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        if self.feature:
            print("Feature vector")
            return x

        x = self.fc6(x)
        return x

def alignment(self, src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def get_feature_vector(self, path_to_img, using_cuda):
    with torch.no_grad():
        lm_path = "Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg"
        
        img = self.alignment(cv2.imread(path_to_img,1),landmark[lm_path])
        imglist = [img,cv2.flip(img,1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
            imglist[i] = (imglist[i]-127.5)/128.0

        image = np.vstack(imglist)
        if using_cuda:
            image = Variable(torch.from_numpy(image).float()).cuda()
        else:
            image = Variable(torch.from_numpy(image).float())
        output = net(image)
        f = output.data

        return f[0].cpu().numpy()

    
def load_weights(self):
    # net = getattr(net_sphere, 'sphere20a')()
    net = SphereFace()
    net.load_state_dict(torch.load(args.model))
    using_cuda = torch.cuda.is_available()
    if using_cuda:
        # net.to(torch.device("cuda"))
        net.cuda()
    net.eval()
    net.feature = True
    return net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
    parser.add_argument('--net','-n', default='sphere20a', type=str)
    parser.add_argument('--lfw', default='lfw.zip', type=str)
    parser.add_argument('--model','-m', default='model/sphere20a_20171020.pth', type=str)
    parser.add_argument('--images', '-i', default='../datasets/fiw/tripairs/faces.csv', type=str)
    args = parser.parse_args()

    predicts=[]
    net = getattr(net_sphere,args.net)()
    net.load_state_dict(torch.load(args.model))

    # using_cuda = next(net.parameters()).is_cuda
    using_cuda = torch.cuda.is_available()
    if using_cuda:
        # net.to(torch.device("cuda"))
        net.cuda()
    net.eval()
    net.feature = True

    # Can remove if we are not zipping dataset
    # zfile = zipfile.ZipFile(args.lfw)

    # Need to find a way to get the landmarks of a face 
    landmark = {}
    with open('data/lfw_landmark.txt') as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.replace('\n','').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]

    with open('data/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    sf = run_sphereface()
    f_me = sf.get_feature_vector("", using_cuda)

    print(len(f_me))

    print(f_me)





# receive a file path in arguements
# open up the image 
# check if the model is using cuda here then move image to cuda.
# run through model 
# return feature vector in numpy array if on cuda move to cpu
 


