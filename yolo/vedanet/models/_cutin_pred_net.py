import os
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
from .. import loss
from .yolo_abc import YoloABC
from ..network import backbone
from ..network import head
from ..network.backbone._cutin_net import CutinNet

__all__ = ['CutinPredNet']


class CutinPredNet(CutinNet):
    def __init__(self, weights_file=None, train_flag=1, clear=False, test_args=None):
        super().__init__(cfg = {
        'out_channels': (116, 232, 464),
        'num_blocks': (3, 7, 3, 3),
        'groups': 2
    }, train_flag=train_flag)

        self.train_flag = train_flag
        self.test_args = test_args
        self.loss = None

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights(slope=0.1)
    '''
    def _forward(self, x, anno):
        
        features = self.backbone(x, anno)
        loss_fn = loss.RegionLoss
        self.compose(x, features, loss_fn)

        return features
    '''
