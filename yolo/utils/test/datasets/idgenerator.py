import glob
import os
import random

name = []
path = '/media/eric/Daten/KITTI/VOCdevkit/VOC_DADA_SV/JPEGImages'
images = glob.glob(path+'/*.jpg')
for image in images:
    name.append(image[-8:-4])
name.sort(key=int)
train = name[:int(0.9*len(name))]
#print(len(train),train)
f = open('/media/eric/Daten/KITTI/VOCdevkit/VOC_DADA_SV/ImageSets/Main/train.txt',"w")
for one in train[:-1]:
    f.write(one)
    f.write("\n")
f.write(train[-1])
f.close()

test = name[int(0.9*len(name)):]
f = open('/media/eric/Daten/KITTI/VOCdevkit/VOC_DADA_SV/ImageSets/Main/test.txt',"w")
for one in test[:-1]:
    f.write(one)
    f.write("\n")
f.write(test[-1])
f.close()