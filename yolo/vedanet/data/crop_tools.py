
import torch
from torchvision import transforms as tf
import numpy as np
from PIL import Image

#__all__ = ['CropTools']

def __build_targets_brambox(ground_truth, expand_ratio=0.0):
    """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
    # Parameters
    nB = len(ground_truth)
    reduction = 1
    # Tensors
    GT = []
    L = []
    for b in range(nB):
        if len(ground_truth[b]) == 0:  # No gt for this image
            GT.append('NULL')  # hold the position for image without annotations
            L.append('NULL')  # hold the position for image without annotations
            continue
        # Build up tensors

        gt = np.zeros((len(ground_truth[b]), 4))
        label = np.zeros(len(ground_truth[b]))
        # one img
        for i, anno in enumerate(ground_truth[b]):
            gt[i, 0] = (anno.x_top_left) / reduction * (1.0 - expand_ratio)
            gt[i, 1] = (anno.y_top_left) / reduction * (1.0 - expand_ratio)
            gt[i, 2] = (anno.x_top_left + anno.width) / reduction * (1.0 + expand_ratio)
            gt[i, 3] = (anno.y_top_left + anno.height) / reduction * (1.0 + expand_ratio)
            if anno.cutin == 1.0:
                label[i] = 1
        GT.append(gt)
        L.append(label)
    return GT, L



def cropped_img_generatir(data):
    img1, img2, target = data

    # visual
    t1 = tf.ToPILImage()(img1[0, :, :, :])
    t2 = tf.ToPILImage()(img2[0, :, :, :])
    # t1.show()
    # t2.show()

    boxes, labels = __build_targets_brambox(target)
    if len(boxes) == 0:
        return None, None
    boxseq = []
    labelseq = []
    for id, one in enumerate(boxes):
        if not isinstance(one, str):
            bndboxes = one.tolist()
            imglabels = labels[id].tolist()
            '''
            t1 = img1[id,:,:,:]
            t2 = img2[id,:,:,:]

            '''

            for ii, box in enumerate(bndboxes):
                boxseq.append(torch.tensor(box).cuda())
                labelseq.append(imglabels[ii])
                a = 1
    if len(boxseq) == 0:
        return img1, img2, None, None

    return img1, img2, boxseq, torch.tensor(labelseq).cuda()