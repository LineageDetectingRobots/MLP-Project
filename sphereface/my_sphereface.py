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

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere

from progress.bar import Bar


class run_sphereface:

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
        
    def lfw_sphereface(self):
        with torch.no_grad():
            bar = Bar('Processing', max=5)
            for i in range(5):
                p = pairs_lines[i].replace('\n','').split('\t')

                if 3==len(p):
                    sameflag = 1
                    name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                    name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
                if 4==len(p):
                    sameflag = 0
                    name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                    name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
                
                print(name1)
                print(name2)

                img1 = self.alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
                img2 = self.alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])

                print("Landmark")
                print(landmark[name1])

                imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
                for i in range(len(imglist)):
                    imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
                    imglist[i] = (imglist[i]-127.5)/128.0
                
                # Combines the lists in vertically 
                img = np.vstack(imglist)
                # img = Variable(torch.from_numpy(img).float()).cuda()
                # Volatile is depricated
                img = Variable(torch.from_numpy(img).float())
                output = net(img)
                f = output.data
                f1,f2 = f[0],f[2]
                print(f1)
                print(f2)
                # make sure that this is not a tensor make it a numpy array
                bar.next()
                if name2 == "Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg":
                    print("Found")
                    final_f = f2
            bar.finish()
            return final_f

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
 


