# # python align/align_dataset_mtcnn.py \
# # ../data/images/train_raw \
# # ../data/images/train_aligned \
# # --image_size 160 \
# # --gpu_memory_fraction 0



# # from PIL import Image

# # img = Image.open("/afs/inf.ed.ac.uk/user/s16/s1674417/Documents/exp/facenet/data/images")

# # # Get cropped and prewhitened image tensor
# # img_cropped = mtcnn(img, save_path=<optional save path>)

# # # Calculate embedding (unsqueeze to add batch dimension)
# # img_embedding = resnet(img_cropped.unsqueeze(0))

# # # Or, if using for VGGFace2 classification
# # resnet.classify = True
# # img_probs = resnet(img_cropped.unsqueeze(0))


# import os
# import pprint 

# dir1 = "/afs/inf.ed.ac.uk/user/s16/s1674417/Documents/exp/dataset/v0.1.2/FIDs"
# dir2 = "/afs/inf.ed.ac.uk/user/s16/s1674417/Documents/exp/facenet-resources/test_raw"
# dir3 = '/afs/inf.ed.ac.uk/user/s16/s1674417/Documents/exp/face-recognition/friends'
# count = 0
# # for subdir, dirs, files in os.walk(dir3):
# #     # print('Dir: ',dirs)
# #     # print('SubDir: ', subdir)
# #     for filename in files:
        
# #         filepath = subdir + os.sep + filename

# #         if filepath.endswith(".jpg") or filepath.endswith(".png"):
# #             print("FileName: ", filename)
# #             count +=1
# #             # print (filepath)
# #     # break
# #     if count >2:
# #         break
# # print(count)


# pp = pprint.PrettyPrinter()
# def path_to_dict(path): 
#     d = {'dirs':{},'files':[]}
#     name = os.path.basename(path)
#     if os.path.isdir(path):
#         if name not in d['dirs']:
#             d['dirs'][name] = {'dirs':{},'files':[]}
#         for x in os.listdir(path):
#             d['dirs'][name]= path_to_dict(os.path.join(path,x))                                                 
#     else:                  
#         d['files'].append(name)        
#     return d               

# mydict = path_to_dict(dir3)
# pp.pprint(mydict)


from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=32)
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
# Process an image:

imgPath = '/afs/inf.ed.ac.uk/user/s16/s1674417/Documents/exp/facenet/data/images/Anthony_Hopkins_0001.jpg'
savePath = '/afs/inf.ed.ac.uk/user/s16/s1674417/Documents/exp/face-recognition/friends/'
img = Image.open(imgPath, save_path = savePath)
print(img)

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img)
# print(img_cropped)

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

print(img_embedding)


# Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))