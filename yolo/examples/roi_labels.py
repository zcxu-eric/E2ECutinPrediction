#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Transform annotations for VOCdevkit to the brambox pickle format
#

# modified by mileistone

import os
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, '.')
import brambox.boxes as bbb

DEBUG = True        # Enable some debug prints with extra information
ROOT = '/media/eric/Daten/KITTI/VOCdevkit'       # Root folder where the VOCdevkit is located
'''
TRAINSET = [
    ('2012', 'train'),
    ('2012', 'val'),
    ('2007', 'train'),
    ('2007', 'val'),
    ]
    '''
TRAINSET = [
    ('2012', 'train'),
    ]
TESTSET = [
    ('2012', 'test'),
    ]

def identify(xml_file):
    root_dir = ROOT
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = root.find('filename').text
    path = f'{root_dir}/VOC2012/JPEGImages/{folder}/{filename}'
    return path

if __name__ == '__main__':

    print('Getting testing annotation filenames')
    test = []
    for (year, img_set) in TESTSET:
        with open(f'{ROOT}/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
            ids = f.read().strip().split()
        test += [f'{ROOT}/VOC{year}/ROI_Annotations/{xml_id}.xml' for xml_id in ids]

    if DEBUG:
        print(f'\t{len(test)} xml files')

    print('Parsing testing annotation files')
    test_annos = bbb.parse('anno_pascalvoc', test, identify)

    print('Generating testing annotation file')
    bbb.generate('anno_pickle', test_annos, f'{ROOT}/onedet_cache/test.pkl')
