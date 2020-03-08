from arcface.data.data_pipe import de_preprocess, get_train_loader, get_val_data
# from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from arcface.my_ArcFace import Backbone, Arcface
# from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# from utils import get_time, gen_plot, hflip_batch
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import sys
from .config import get_config
from .mtcnn import MTCNN

class face_learner(object):
    def __init__(self, conf, inference=False):
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        self.threshold = conf.threshold

    def _device(self):
        return next(self.model.parameters()).device
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        print(save_path)
        try:
            if conf.device.type == 'cpu':
                loaded_model = torch.load(save_path/'model_{}'.format(fixed_str), map_location='cpu')
            else:
                loaded_model = torch.load(save_path/'model_{}'.format(fixed_str))
            
        except IOError:
            print("Arcface: Weights not found. Please download from " +
            "https://onedrive.live.com/?authkey=!AOw5TZL8cWlj10I&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13!835&parId=root&action=locate/" +
            " and add model_ir_se50.pth to arcface/work_space/model/")
            sys.exit(0)
        self.model.load_state_dict(loaded_model)

    def get_features(self, image_inputs):
        with torch.no_grad():
            mtcnn = MTCNN()
            conf = get_config(False)

            # img = Image.open("arcface/Akhmed_Zakayev_0003.jpg")
            

            #====================== Preprocessing the image=================
            # ready_img = mtcnn.align(img)
            #conf.test_transform(ready_img).to(conf.device).unsqueeze(0)

            # Unsqueeze adds an extra dimension to the start of the tensor
            tensor = self.model(image_inputs)
            embedding = tensor[0]
            print(tensor.shape)
            print(len(tensor))
        return embedding

        

