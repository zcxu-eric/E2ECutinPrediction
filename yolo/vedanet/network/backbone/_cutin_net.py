import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import logging as log
from .. import layer as vn_layer
from .brick import shufflenetv2 as bsnv2
from brambox.boxes.annotations.pickle import PickleParser

__all__ = ['cutinnet']

# default shufflenetv2 1x
class CutinNet(nn.Module):
    def __init__(self, cfg, train_flag = 1):
        super().__init__()
        self.seen = 0
        self.train_flag = train_flag
        out_planes = cfg['out_channels']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        # Network
        layers_list_pre = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('stage3/convbatchrelu', vn_layer.Conv2dBatchReLU(3, 24, 3, 2)),
                ('stage3/max', nn.MaxPool2d(3, 2, 1)),
            ]),

            OrderedDict([
                ('Stage4', bsnv2.Stage(24, out_planes[0], groups, num_blocks[0])),
            ]),
        ]
        layers_list_suc = [
            OrderedDict([
                ('Stage5', bsnv2.Stage(2*out_planes[0], out_planes[1], groups, num_blocks[1])),
            ]),

            OrderedDict([
                ('Stage6', bsnv2.Stage(out_planes[1], out_planes[2], groups, num_blocks[2])),
                # the following is extra
            ]),
        ]

        self.pre_layers = nn.ModuleList([nn.Sequential(layer_dict_pre) for layer_dict_pre in layers_list_pre])
        self.suc_layers = nn.ModuleList([nn.Sequential(layer_dict_suc) for layer_dict_suc in layers_list_suc])

        self.dense_1 = nn.Linear(1*464*5*5,200)
        self.dense_2 = nn.Linear(200, 2)


    def forward(self, x, target):
        self.seen += x[0].size(0)
        # x conaints consecutive frames
        # divide x into x1 and x2
        x1 = x[0].unsqueeze(dim = 0)
        x2 = x[1].unsqueeze(dim = 0)
        target = target.unsqueeze(dim = 0)

        stem_1 = self.pre_layers[0](x1) #bs,24,128,128
        stage4_1 = self.pre_layers[1](stem_1)
        stem_2 = self.pre_layers[0](x2)
        stage4_2 = self.pre_layers[1](stem_2)
        stage4 = torch.cat((stage4_1, stage4_2), dim=1)
        stage5 = self.suc_layers[0](stage4)
        stage6 = self.suc_layers[1](stage5) #1,464,5,5
        d1 = self.dense_1(torch.flatten(stage6))
        d2 = self.dense_2(d1)
        output = d2.unsqueeze(dim = 0)
        # ROI Align
        loss = nn.CrossEntropyLoss()(output, target.long())
        if self.train_flag == 2:
            return output
        else:
            return loss



    def init_weights(self, mode='fan_in', slope=0):
        info_list = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                info_list.append(str(m))
                nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                info_list.append(str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                info_list.append(str(m))
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        log.info('Init weights\n\n%s\n' % '\n'.join(info_list))

    def load_weights(self, weights_file, clear=False):
        """ This function will load the weights from a file.
        It also allows to load in weights file with only a part of the weights in.

        Args:
            weights_file (str): path to file
        """
        old_state = self.state_dict()
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = 0 if clear else state['seen']

        self.load_state_dict(state['weights'])

        if hasattr(self.loss, 'seen'):
            self.loss.seen = self.seen

        log.info(f'Loaded weights from {weights_file}')

    def save_weights(self, weights_file, seen=None):
        """ This function will save the weights to a file.

        Args:
            weights_file (str): path to file
            seen (int, optional): Number of images trained on; Default **self.seen**
        """
        if seen is None:
            seen = self.seen

        state = {
            'seen': seen,
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)

        log.info(f'Saved weights as {weights_file}')



def cutinnet():
    cfg = {
        'out_channels': (116, 232, 464),
        'num_blocks': (3, 7, 3),
        'groups': 2
    }
    return CutinNet(cfg)