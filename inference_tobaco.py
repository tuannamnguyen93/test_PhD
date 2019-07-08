from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import json
import os
from models.all_model import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from datetime import datetime
import numpy as np
import cv2
network="googlenet" #we have 4 options: alexnet,vgg,googlenet,resnet
pretrained=False
dataset="RVL"
n_classes=10
batch_size = 32

if network == "vgg":
    IMG_SIZE = 224
    model = vgg(n_classes,IMG_SIZE,dataset,pretrained)
    
elif network == "googlenet" :
    IMG_SIZE = 224
    model = googlenet(n_classes,IMG_SIZE,dataset,pretrained)
elif network == "resnet" :
    IMG_SIZE = 224
    model = resnet(n_classes,IMG_SIZE,dataset,pretrained)
elif network == "alexnet" : 
    IMG_SIZE = 227
    model = Alexnet(n_classes,IMG_SIZE,dataset,pretrained)
#you must to change checkpoint if you want to change model
model.load_weights("./tobaco_checkpoint/googlenet_tobaco_Document_pretrained.hdf5")

output = []
img_array =[]
list_files= os.listdir("input_tobacco")

with open('labelmap.txt') as json_file:  
    map_2_label = json.load(json_file)

for file in list_files:
    img = cv2.imread("input_tobacco/"+file)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img_array.append(img)

img_array = np.array(img_array)
pre = model.predict(img_array,batch_size=16)
labels=np.argmax(pre,axis=-1)
list_label=os.listdir("Tobacco3482")

with open('./output_tobacco/your_result.txt', 'w') as f:
     for file,label in zip(list_files,labels):
         
         f.write("%s  " % file + "%s\n" % list_label[label])


