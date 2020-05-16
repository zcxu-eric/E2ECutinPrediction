import logging as log
import torch
from torchvision import transforms as tf
import torch.nn.functional as F
from statistics import mean
import os
import glob
import re
import numpy as np

from .. import data as vn_data
from .. import models
from . import engine
from utils.test import voc_wrapper
from utils.test.datasets import voc_eval
from ..data.crop_tools import cropped_img_generatir
__all__ = ['VOCTest']

class CustomDataset(vn_data.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.testfile
        root = hyper_params.data_root
        network_size = hyper_params.network_size
        labels = hyper_params.labels


        lb  = vn_data.transform.Letterbox(network_size)
        it  = tf.ToTensor()
        img_tf = vn_data.transform.Compose([lb])
        anno_tf = vn_data.transform.Compose([lb])

        def identify(img_id):
            return f'{img_id}'

        super(CustomDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        imgcur, imgpre, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return imgcur, imgpre, anno


def VOCTest(hyper_params):

    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    nms_thresh = hyper_params.nms_thresh
    #prefix = hyper_params.prefix
    results = hyper_params.results

    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size, 'labels': labels}
    net = models.__dict__[model_name](weights, train_flag=2, test_args=test_args)
    net.eval()
    log.info('Net structure\n%s' % net)
    #import pdb
    #pdb.set_trace()
    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(hyper_params),
        batch_size = batch,
        shuffle = False,
        drop_last = False,
        num_workers = nworkers if use_cuda else 0,
        pin_memory = pin_mem if use_cuda else False,
        collate_fn = vn_data.list_collate,
    )

    log.debug('Running network')
    total = 0
    correct = 0
    cutin_correct = 0
    cutin_total = 0
    confusion_matrix = np.zeros(shape=(2,2))
    for idx, data in enumerate(loader):
        if (idx + 1) % 20 == 0: 
            log.info('image batches: %d/%d' % (idx + 1, len(loader)))
        cropped_imgs, labels = cropped_img_generatir(data)
        try:
            total += len(labels)
        except:
            continue
        for id, pair in enumerate(cropped_imgs):
            # to(device)
            if use_cuda:
                data1 = pair[0].cuda()
                data2 = pair[1].cuda()
                with torch.no_grad():
                    output = net([data1,data2], labels[id])
                score = F.softmax(output,dim=1).cpu().numpy()
                pred = np.argmax(score)
                gt = labels[id]
                gt = int(gt.cpu().numpy())
                confusion_matrix[gt,pred] += 1
                if gt:
                    cutin_total += 1
                    if pred:
                        cutin_correct += 1
                if (int(pred) == gt):
                    correct += 1
    cmd = 'cp ' + hyper_params.weights + ' ' + 'weights/' + 'cutinprednet_%.3f_%.3f.pth' % (correct/total,cutin_correct/cutin_total)
    os.system(cmd)
    print(f'top1-accuracy:{correct/total},cutin accuracy: {cutin_correct/cutin_total}')
    print(confusion_matrix)

        #key_val = len(anno)
        #anno.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(box)})
        #det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})



def calculate_map(hyper_params):
    ###################################
    #calculate mAP  2020.5.9  by xzc  #
    ###################################
    data_root_dir,_ = os.path.split(hyper_params.data_root)
    detfiles = glob.glob('results/*.txt')
    sum_ap = 0
    for one_det in detfiles:
        classname = re.match(r'results/(.*?)_det_test_(.*?).txt', one_det).group(2)

        _,_, ap = voc_eval.voc_eval(detpath = one_det,
                                 annopath = data_root_dir + '/VOC2012/Annotations/{}.xml',
                                 imagesetfile= data_root_dir + '/VOC2012/ImageSets/Main/test.txt',
                                 classname = classname,
                                 cachedir = data_root_dir)
        sum_ap += ap

    mAP = sum_ap/len(detfiles)
    cmd = 'cp '+ hyper_params.weights + ' ' + 'weights/' + 'regionshufflenetv2_%f.pth'%mAP
    os.system(cmd)
    print('mAP:',mAP,'model saved.')



