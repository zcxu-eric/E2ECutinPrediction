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

name_list = ["bus","truck","person","tv"]
cutin_list = ["cutin"]
#name_list = ["car","Van","person","Truck","Cyclist"]
labels = []
filepath = '/home/eric/KITTI/VOCdevkit/VOC2012/C_Annotations/'

rois = {'1':[[620,610],[835,610],[525,800],[1045,700]],
        '2':[[424,568],[600,562],[223,778],[803,793]],
        '3':[[492,640],[660,640],[336,770],[880,787]],
        '4':[[310,580],[490,580],[212,740],[680,770]],
        '5':[[506,470],[690,470],[336,770],[1060,690]],
        '6':[[464,413],[757,413],[218,740],[1040,700]],
        '7':[[530,406],[618,406],[214,676],[947,674]],
        '8':[[470,544],[805,540],[221,780],[945,755]],
        '9':[[383,493],[565,482],[110,733],[945,757]],
        '10':[[590,400],[744,400],[436,800],[1056,700]],}

expand_ratio = {'1':-0.0006,
                '2':-0.002,
                '3':-0.001,
                '4':0.0005,
                '5':-0.0015,
                '6':-0.00037,
                '7':-0.00278,
                '8':-0.0148,
                '9':-0.0009,
                '10':-0.0008}

cutdis = []
nocutdis = []
confusion = np.zeros((2, 2))
count = 0
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

def roi_checker(xml_path):
    start = time.time()
    global cutdis
    global nocutdis
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

    for _ob in ob:

        box = _ob.find('bndbox')
        xmin = int(box.find('xmin').text)
        xmax = int(box.find('xmax').text)
        ymin = int(box.find('ymin').text)
        ymax = int(box.find('ymax').text)

        if ymax <= roi[0][1]:
            pred = 0
        elif ((xmin >= roi[2][0] and xmin <= roi[3][0]) or (xmax >= roi[2][0] and xmax <= roi[3][0])):
            pred = 1
        elif isThrough((xmin,ymin),(xmax,ymax),roi[0],roi[2]) or isThrough((xmin,ymax),(xmax,ymin),roi[1],roi[3]):
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
    Tree.write(xml_path)

    end = time.time()
    print('\rProcessing ' + folder + '/' + fname + ' time consuming: %s s' % str(end - start), end=" ")

def xml_projector(xml_path):
    global count
    start = time.time()
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    base, folder = os.path.split(base)
    root = Tree.getroot()
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
        count += 1
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

    #prettyXml(root, '\t', '\n')
    #Tree.write(xml_path)
    
    end = time.time()
    print('\rProcessing '+folder + '/' + fname+' time consuming: %s s'%str(end-start), end= " ")
       
            
        

if __name__ == '__main__':
    for i in range(1,11):
        xmls = xml_files(filepath+str(i))
        list(map(xml_projector,xmls))
    print('\n',count)
    #pool = Pool(4)
    #pool.map(roi_checker,xmls)
    #pool.close()
    #pool.join()
