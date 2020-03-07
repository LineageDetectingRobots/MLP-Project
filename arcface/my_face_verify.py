import argparse
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
import cv2
from PIL import Image
from torchvision import transforms as trans
import torch



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    # Sets training to False
    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    with torch.no_grad():
        learner = face_learner(conf, True)
        learner.threshold = args.threshold
        if conf.device.type == 'cpu':
            learner.load_state(conf, 'ir_se50.pth', False, True)
        else:
            learner.load_state(conf, 'final.pth', True, True)
        learner.model.eval()
        # print(learner)
        print('learner loaded')


        # if args.update:
        #     targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        #     print('facebank updated')
        # else:
        #     # we should just be using this 
        #     targets, names = load_facebank(conf)
        #     print('facebank loaded')

        # train_transform = trans.Compose([
        #     trans.RandomHorizontalFlip(),
        #     trans.ToTensor(),
        #     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #         ])

        img = Image.open("Akhmed_Zakayev_0003.jpg")
        ready_img = mtcnn.align(img)

        tensor = learner.model(conf.test_transform(ready_img).to(conf.device).unsqueeze(0))
        embedding = tensor[0].cpu().numpy()
        print(len(embedding))
    