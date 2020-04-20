import glob
import os
import re
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from multiprocessing import Pool
import time

name_list = ["bus","truck","person","tv"]
cutin_list = ["cutin"]
#name_list = ["car","Van","person","Truck","Cyclist"]
labels = []
filepath = '/media/eric/Daten/KITTI/VOCdevkit/VOC2012/Annotations'
#filepath = './'

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

def xml_projector(xml_path):
    start = time.time()
    Tree = ET.parse(xml_path)
    base, fname = os.path.split(xml_path)
    root = Tree.getroot()
    ob = root.findall('object')
    for _ob in ob:
        #if _ob.text == str(one[1]):
        name = _ob.find('name').text
        
        if name in name_list:
            _ob.find('name').text = "car"

        if name not in cutin_list and _ob.find('cutin') == None:
            cutin = SubElement(_ob,"cutin")
            cutin.text = "0"
        elif name in cutin_list and _ob.find('cutin') == None:
            _ob.find('name').text = "car"
            cutin = SubElement(_ob,"cutin")
            cutin.text = "1"
    prettyXml(root, '\t', '\n')
    Tree.write(os.path.join(filepath,fname))
    
    end = time.time()
    print('\rProcessing '+fname+' time consuming: %s s'%str(end-start), end= " ")
       
            
        

if __name__ == '__main__':
    #xmls = xml_files('./')
    #xml_projector(xmls[0])

    pool = Pool(2)
    xmls = xml_files(filepath)
    pool.map(xml_projector,xmls)
    pool.close()
    pool.join()
