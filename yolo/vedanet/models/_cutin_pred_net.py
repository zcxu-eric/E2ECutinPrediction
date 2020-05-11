import os
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import torchvisioin
from .. import loss
from .yolo_abc import YoloABC
from ..network import backbone
from ..network import head

__all__ = ['CutinPredNet']


class CutinPredNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        out_planes = cfg['out_channels']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        # Network
        layers_list_pre = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('stage3/convbatchrelu', vn_layer.Conv2dBatchReLU(3, 24, 3, 2)),
                ('stage3/max',           nn.MaxPool2d(3, 2, 1)),
                ]),

            OrderedDict([
                ('Stage4', bsnv2.Stage(24, out_planes[0], groups, num_blocks[0])),
                ]),
            ]
        layers_list_suc = [
            OrderedDict([
                ('Stage5', bsnv2.Stage(out_planes[0], out_planes[1], groups, num_blocks[1])),
                ]),

            OrderedDict([
                ('Stage6', bsnv2.Stage(out_planes[1], out_planes[2], groups, num_blocks[2])),
                # the following is extra
                ]),
            ]

        self.pre_layers = nn.ModuleList([nn.Sequential(layer_dict_pre) for layer_dict_pre in layers_list_pre])
        self.suc_layers = nn.ModuleList([nn.Sequential(layer_dict_suc) for layer_dict_suc in layers_list_suc])

    def forward(self, x, boxs):
        #x conaints consecutive frames
        #divide x into x1 and x2
        x1 = torch.zeros(size = (504,504,3))
        x2 = torch.zeros(size = (504,504,3))
        stem_1 = self.pre_layers[0](x1)
        stage4_1 = self.pre_layers[1](stem_1)
        stem_2 = self.pre_layers[0](x2)
        stage4_2 = self.pre_layers[1](stem_2)

        #ROI Align
        torchvision.ops.roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)
        stage4 = torch.cat((stage4_1,stage4_2),dim = 0)

        stage5 = self.suc_layers[2](stage4)
        stage6 = self.suc_layers[3](stage5)
        features = stage6

        return features
    '''
    def __init__(self, num_classes=1, weights_file=None, input_channels=3, train_flag=1, clear=False, test_args=None):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.cutin = 0
        self.train_flag = train_flag
        self.test_args = test_args

        self.loss = None
        self.postprocess = None
        self.backbone = backbone.shufflenetv2()

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights(slope=0.1)

    def _forward(self, x):
        middle_feats = self.backbone(x)
        features = self.head(middle_feats)
        loss_fn = loss.RegionLoss

        self.compose(x, features, loss_fn)

        return features

    def modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module
    '''
