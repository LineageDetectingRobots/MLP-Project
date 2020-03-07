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

class face_learner(object):
    def __init__(self, conf, inference=False):
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        self.threshold = conf.threshold
    

    
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

        

