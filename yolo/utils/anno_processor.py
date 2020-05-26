import glob
import os
import re
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from multiprocessing import Pool
import time
import math
from torchvision import models,transforms
import torch.nn as nn
import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
class_names = ['front', 'left', 'right', 'tail']

model_ft = models.shufflenet_v2_x1_0(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('/home/eric/QZYGesture/checkpoints/car_best.pth'))
model_ft.eval()

name_list = ["bus","truck","person","tv"]
cutin_list = ["cutin"]
#name_list = ["car","Van","person","Truck","Cyclist"]
labels = []
filepath = '/home/eric/KITTI/VOCdevkit/VOC2012/Anno/'
imgpath = '/media/eric/Daten/KITTI/VOCdevkit/VOC2012/JPEGImages'
crop_dir = '/media/eric/Daten/KITTI/VOCdevkit/VOC2012/CropImages'
cutin_total = 0
nocutin_total = 0
countcut = 0
countnocut = 0
rois = {'1':[[620,610],[835,610],[525,800],[1045,800]],
        '2':[[424,568],[600,568],[223,780],[803,780]],
        '3':[[492,640],[660,640],[336,770],[880,770]],
        '4':[[310,580],[490,580],[212,770],[680,770]],
        '5':[[506,470],[690,470],[336,770],[1060,770]],
        '6':[[464,413],[757,413],[218,770],[1040,770]],
        '7':[[530,406],[618,406],[214,670],[947,670]],
        '8':[[470,540],[805,540],[221,780],[945,780]],
        '9':[[383,482],[565,482],[110,750],[945,750]],
        '10':[[590,400],[744,400],[436,800],[1056,800]],
        '11':[[363,500],[744,500],[63,710],[1056,700]],
        '12':[[435,555],[594,555],[206,740],[900,740]],
        '13':[[180,400],[330,400],[63,700],[700,776]],}

expand_ratio = {'1':-0.0006,
                '2':-0.002,
                '3':-0.001,
                '4':0.0005,
                '5':-0.0015,
                '6':-0.00037,
                '7':-0.00278,
                '8':-0.0148,
                '9':-0.0009,
                '10':-0.0008,
                '11':0,
                '12':0.000,
                '13':0,}

cutdis = []
nocutdis = []
confusion = np.zeros((2, 2))
####################################
#      P1-------------- P2
#     /                  \
#    /                    \
#   /                      \
#  /                        \
#P3--------------------------P4
# roi order (p1 p2 p3 p4)
#####################################
def xml_files(dir_path):
    xmls = glob.glob(dir_path+'/*.xml')
    xmls.sort()
    return xmls

def prettyXml(element, indent, newline, level=0):
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将elemnt转成list
    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
            # 对子元素进行递归操作
        prettyXml(subelement, indent, newline, level=level + 1)

def isThrough(point1, point2, linep1,linep2):
    pointX1 = point1[0]
    pointY1 = point1[1]
    pointX2 = point2[0]
    pointY2 = point2[1]
    lineX1 = linep1[0]
    lineY1 = linep1[1]
    lineX2 = linep2[0]
    lineY2 = linep2[1]
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    out1 = a*pointX1+b*pointY1+c
    out2 = a * pointX2 + b * pointY2 + c
    return out1*out2<0

def iou(rec1,rec2):
    """
        computing IoU
         rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        rec2: (y0, x0, y1, x1)
        scala value of IoU
        """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0
def inter(rec1,rec2):
    """
        computing IoU
         rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        rec2: (y0, x0, y1, x1)
        scala value of IoU
        """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect/S_rec1) * 1.0

def roi_inter(box, folder):
    pieceroi = []
    count = 0
    roi = rois[folder]
    roi[0][0] = roi[0][0] * (1 - expand_ratio[folder])
    roi[2][0] = roi[2][0] * (1 - expand_ratio[folder])
    roi[1][0] = roi[1][0] * (1 + expand_ratio[folder])
    roi[3][0] = roi[3][0] * (1 + expand_ratio[folder])
    [p1, p2, p3, p4] = roi
    yslide = 5
    kr = (p4[0]-p2[0])/(p4[1]-p2[1])
    kl = (p1[0]-p3[0])/(p3[1]-p1[1])
    xslider = kr*yslide
    xslidel = kl*yslide
    ymin = p1[1] - yslide
    xmin = p1[0]
    ymax = ymin + yslide
    xmax = p2[0]

    _xmin = int(xmin - count * xslidel)
    _xmax = int(xmax + count * xslider)
    _ymin = ymin + count * yslide
    _ymax = ymax + count * yslide

    while _ymin < p3[1]:
        pieceroi.append([_xmin,_ymin,_xmax,_ymax])
        count += 1
        _xmin = int(xmin - count * xslidel)
        _xmax = int(xmax + count * xslider)
        _ymin = ymin + count * yslide
        _ymax = ymax + count * yslide
    intersum = 0
    for one in pieceroi:
        intersum += inter(box,one)
    return intersum

def cutin_predictor(xml_path):
    thres = 0.7
    start = time.time()
    global cutdis
    global nocutdis

    base, fname = os.path.split(xml_path)
    _,folder = os.path.split(base)
    pt = os.path.join(os.path.join(imgpath, folder), fname[:-4] + '.jpg')
    IMG = cv2.imread(os.path.join(os.path.join(imgpath, folder), fname[:-4] + '.jpg'))
    if int(fname[:-4])<2:
        return
    IMGboxes = []
    GT = []
    for i in range(2):
        imgboxes = []
        gt = []
        try:
            Tree = ET.parse(os.path.join(base,str(int(fname[:-4])-i).zfill(4)+'.xml'))
        except:
            return
        root = Tree.getroot()
        ob = root.findall('object')
        for _ob in ob:
            box = _ob.find('bndbox')
            xmin = int(box.find('xmin').text)
            xmax = int(box.find('xmax').text)
            ymin = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)
            imgboxes.append([xmin,ymin,xmax,ymax])
            gt.append(int(_ob.find('cutin').text))
        if len(imgboxes) == 0:
            IMGboxes.append('NULL')
        else:
            IMGboxes.append(imgboxes)
        GT.append(gt)

    imgboxes = IMGboxes[0]
    gt = GT[0]
    if isinstance(imgboxes, str):
        return      #no objects

    INTER = np.zeros(shape=len(imgboxes))  #inter for each obj in one img
    pose = np.zeros(shape=len(imgboxes))   #pose for each obj in one img
    prob = np.zeros(shape=len(imgboxes))   #inter for each obj in one img
    pred = np.zeros(shape=len(imgboxes))   #pred for each obj in one img

    for  obj, one in enumerate(imgboxes):
        if isinstance(IMGboxes[1],str):
            continue
        ious = [iou(one, nextone) for nextone in IMGboxes[1]]
        for _id, io in enumerate(ious):
            if io >= thres:
                ROI_inter_2 = roi_inter(one, folder) #current
                ROI_inter_1 = roi_inter(IMGboxes[1][_id],folder) #pre
                INTER[obj] = (ROI_inter_2)  # record inter for obj in current img
                if ROI_inter_1 > 0.00 or ROI_inter_2 > 0.00:
                    cropped = IMG[one[1]:one[3], one[0]:one[2]]
                    try:
                        img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))  # opencv to PIL
                    except:
                        a = 1
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img = transform(img)
                    img = img.unsqueeze(0)
                    img = img.to(device)
                    with torch.no_grad():
                        output = model_ft(img)
                        _, pose[obj] = torch.max(output, 1)
                        # print(class_names[preds])
                        # out = torchvision.utils.make_grid(img)
                        # imshow(out.cpu().data)
                break
    Tree = ET.parse(os.path.join(base, str(int(fname[:-4])).zfill(4) + '.xml'))
    root = Tree.getroot()
    ob = root.findall('object')
    for i in range(len(imgboxes)):
        if pose[i] == 3 or pose[i] == 0:
            INTER[i] = 0
            #print(xml_path)
    max = INTER.argsort()[::-1][0:2]
    max = np.max(INTER)
    for i in range(len(imgboxes)):
        if INTER[i] >0:
            prob[i] = INTER[i]
            _prob = SubElement(ob[i],'prob')
            _prob.text = str(prob[i])
            if INTER[i]/max >= 0.5:
                pred[i] = 1

    for i in range(len(imgboxes)):

        if ob[i].find('prob') == None:
            _prob = SubElement(ob[i], 'prob')
            _prob.text = str(0)

        try:
            if (int(gt[i])==0) and int(pred[i])==1:
                #print(xml_path)
                pass
            confusion[int(gt[i]), int(pred[i])] += 1
        except:
            pass
    prettyXml(root, '\t', '\n')
    Tree.write(xml_path)
    end = time.time()
    print('\rProcessing ' + folder + '/' + fname + ' time consuming: %s s' % str(end - start), end=" ")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def roi_checker(xml_path):
    start = time.time()
    global cutdis
    global nocutdis
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    base, folder = os.path.split(base)
    root = Tree.getroot()

    roi = rois[folder]

    roi[0][0] = roi[0][0]*(1-expand_ratio[folder])
    roi[2][0] = roi[2][0]*(1-expand_ratio[folder])
    roi[1][0] = roi[1][0]*(1+expand_ratio[folder])
    roi[3][0] = roi[3][0]*(1+expand_ratio[folder])
    ob = root.findall('object')
    pt = os.path.join(os.path.join(imgpath, folder), fname[:-4] + '.jpg')
    IMG = cv2.imread(os.path.join(os.path.join(imgpath, folder), fname[:-4] + '.jpg'))
    for _ob in ob:

        box = _ob.find('bndbox')
        xmin = int(box.find('xmin').text)
        xmax = int(box.find('xmax').text)
        ymin = int(box.find('ymin').text)
        ymax = int(box.find('ymax').text)
        cropped = IMG[ymin:ymax,xmin:xmax]
        try:
            img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))  # opencv to PIL
        except:
            a = 1
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = model_ft(img)
            _, pose = torch.max(output, 1)
            #print(class_names[preds])
            #out = torchvision.utils.make_grid(img)
            #imshow(out.cpu().data)
        if ymax <= roi[0][1]:
            pred = 0
        elif ((xmin >= roi[2][0] and xmin <= roi[3][0]) or (xmax >= roi[2][0] and xmax <= roi[3][0])) and (pose == 2 or pose == 1):
            pred = 1
        elif isThrough((xmin,ymin),(xmax,ymax),roi[0],roi[2]) or isThrough((xmin,ymax),(xmax,ymin),roi[1],roi[3]) and (pose == 2 or pose == 1):
            pred = 1
        else:
            pred = 0
        if not pred:
            root.remove(_ob)
        try:
            confusion[int(_ob.find('cutin').text),pred] += 1
        except:
            print(xml_path)

            #dis = min(getDis(center[0],center[1],roi[0][0],roi[0][1],roi[2][0],roi[2][1]),getDis(center[0],center[1],roi[1][0],roi[1][1],roi[3][0],roi[3][1]))


    a = 1

    prettyXml(root, '\t', '\n')
    #Tree.write(xml_path)

    end = time.time()
    print('\rProcessing ' + folder + '/' + fname + ' time consuming: %s s' % str(end - start), end=" ")

def xml_projector(xml_path):
    start = time.time()
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    base, folder = os.path.split(base)
    root = Tree.getroot()
    root.find('folder').text = folder
    ob = root.findall('object')
    for _ob in ob:
        '''
        box = _ob.find('bndbox')
        if box.find('xmin').text == '493'and box.find('ymin').text== '410':
            print(xml_path)
        '''
        #if _ob.text == 'VOC2012':
        #    _ob.text = folder
        #_ob.text = fname[:-4]+'.jpg'
        '''
        ob = root.findall('folder')
        for _ob in ob:
            if _ob.text == "VOC2012":
                _ob.text = "10"
        '''

        #if _ob.text == str(one[1]):

        name = _ob.find('name').text
        #if name in name_list:
        #    _ob.find('name').text = "car"

        if name not in cutin_list and _ob.find('cutin') == None:
            cutin = SubElement(_ob,"cutin")
            cutin.text = "0"
        elif name in cutin_list and _ob.find('cutin') == None:
            _ob.find('name').text = "car"
            cutin = SubElement(_ob,"cutin")
            cutin.text = "1"

    prettyXml(root, '\t', '\n')
    Tree.write(xml_path)
    
    end = time.time()
    print('\rProcessing '+folder + '/' + fname+' time consuming: %s s'%str(end-start), end= " ")


def cutin_balancer(xml_path):
    cutin = []
    nocutin = []
    reserve = []
    global cutin_total
    start = time.time()
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    base, folder = os.path.split(base)
    root = Tree.getroot()
    ob = root.findall('object')
    for _ob in ob:
        try:
            label = _ob.find('cutin').text
        except:
            print(xml_path)
        label = _ob.find('cutin').text
        if label == '1':
            cutin.append(_ob)
        else:
            nocutin.append(_ob)
    if len(cutin)!=0:
        cutin_total += 1
        try:
            reserve = sample(nocutin,1)
        except:
            pass
    rm = list(set(nocutin)-set(reserve))
    for one in rm:
        root.remove(one)

    prettyXml(root, '\t', '\n')
    Tree.write(xml_path)

    end = time.time()
    print('\rProcessing ' + folder + '/' + fname + ' time consuming: %s s' % str(end - start), end=" ")


def xml_reverser(xml_path):
    start = time.time()
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    base, folder = os.path.split(base)
    root = Tree.getroot()
    ob = root.findall('object')
    for _ob in ob:


        # if _ob.text == str(one[1]):
        cl = _ob.find('cutin').text

        # if name in name_list:
        #    _ob.find('name').text = "car"

        if cl == '1':
            _ob.find('name').text = 'cutin'

    prettyXml(root, '\t', '\n')
    Tree.write(xml_path)

    end = time.time()
    print('\rProcessing ' + folder + '/' + fname + ' time consuming: %s s' % str(end - start), end=" ")


def crop_imgs(xml_path):
    start = time.time()
    global countcut
    global countnocut
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    base, folder = os.path.split(base)
    root = Tree.getroot()
    folder = root.find('folder').text
    roi = rois[folder]

    roi[0][0] = roi[0][0]*(1-expand_ratio[folder])
    roi[2][0] = roi[2][0]*(1-expand_ratio[folder])
    roi[1][0] = roi[1][0]*(1+expand_ratio[folder])
    roi[3][0] = roi[3][0]*(1+expand_ratio[folder])
    ob = root.findall('object')
    pt = os.path.join(os.path.join(imgpath,folder),fname[:-4]+'.jpg')
    img = cv2.imread(os.path.join(os.path.join(imgpath,folder),fname[:-4]+'.jpg'))

    for _ob in ob:

        box = _ob.find('bndbox')
        xmin = int(box.find('xmin').text)
        xmax = int(box.find('xmax').text)
        ymin = int(box.find('ymin').text)
        ymax = int(box.find('ymax').text)
        cropped = img[ymin:ymax,xmin:xmax]
        #cv2.imshow('img', cropped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if _ob.find('cutin').text == '1':
            cv2.imwrite(crop_dir+'/1/'+str(countcut).zfill(4)+'.jpg',cropped)
            countcut += 1
        else:
            cv2.imwrite(crop_dir + '/2/' + str(countnocut).zfill(4) + '.jpg', cropped)
            countnocut += 1

    #Tree.write(xml_path)

    end = time.time()
    print('\rProcessing ' + folder + '/' + fname + ' time consuming: %s s' % str(end - start), end=" ")



if __name__ == '__main__':
    for i in range(4,5):
        xmls = xml_files(filepath+str(i))
        #cutin_predictor(xmls[227])
        list(map(cutin_predictor,xmls))
    print('\n',confusion,"recall:",confusion[1,1]/(confusion[1,0]+confusion[1,1]))
    #pool = Pool(4)
    #pool.map(roi_checker,xmls)
    #pool.close()
    #pool.join()
