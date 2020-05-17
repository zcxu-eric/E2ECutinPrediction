import os
import glob
from random import sample

root = '/media/eric/XZC/KITTI/VOCdevkit/VOC2012/ImageSets/Main'
img_dir = '/media/eric/XZC/KITTI/VOCdevkit/VOC2012/JPEGImages'

trainfile = os.path.join(root,'train.txt')
testfile = os.path.join(root,'test.txt')
if os.path.exists(trainfile):
    os.remove(trainfile)
if os.path.exists(testfile):
    os.remove(testfile)

for i in range(1,11):
    ori_img_list = glob.glob(img_dir + '/' +str(i) + '/*.jpg')
    img_list = []
    for id, one in enumerate(ori_img_list):
        base,file = os.path.split(one)
        if int(file[:-4]) >= 40:
            img_list.append(one)
    train = sample(img_list, int(0.7*len(img_list)))
    test = list(set(img_list) - set(train))
    train = [one[len(img_dir) + 1:-4] + '\n' for one in train]
    test = [one[len(img_dir) + 1:-4] + '\n' for one in test]
    with open(trainfile,'a+') as f:
        f.writelines(one for one in train)
    f.close()
    with open(testfile,'a+') as f:
        f.writelines(one for one in test)
    f.close()


