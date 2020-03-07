from data.data_pipe import de_preprocess, get_train_loader, get_val_data
# from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from my_ArcFace import Backbone, Arcface
# from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# from utils import get_time, gen_plot, hflip_batch
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz

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
        # self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        if torch.cuda.is_available():
            loaded_model = torch.load(save_path/'model_{}'.format(fixed_str))
        else:
            loaded_model = torch.load(save_path/'model_{}'.format(fixed_str), map_location='cpu')
        self.model.load_state_dict(loaded_model)

