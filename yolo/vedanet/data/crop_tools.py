
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
    IMGS, target = data

    boxes, labels = __build_targets_brambox(target)
    if len(boxes) == 0:
        return None, None
    imgs = []
    tmp = []
    labelseq = []
    for id, one in enumerate(boxes):
        if not isinstance(one, str):
            bndboxes = one.tolist()
            imglabels = labels[id].tolist()
            for i in range(8):
                tmp.append(IMGS[0, i, :, :])
            tmp = [tf.ToPILImage()(one) for one in tmp]

            for ii, box in enumerate(bndboxes):
                tmp = [one.crop((box[0], box[1], box[2], box[3])).resize((160, 160), Image.BILINEAR) for one in tmp]
                # for one in tmp:
                # one.show()
                imgs.append(tmp)
                labelseq.append(imglabels[ii])
    if len(imgs) == 0:
        return None, None
    cropped_imgs = [[tf.ToTensor()(one) for one in oneset] for oneset in imgs]  # cropped imgs from one image for cutin
    # cropped_imgs, labels = self.cutin_balance(cropped_imgs, labelseq)
    return cropped_imgs, torch.tensor(labelseq).cuda()