
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

    boxes, labels = __build_targets_brambox(target)
    if len(boxes) == 0:
        return None, None
    imgs = []
    labelseq = []
    for id, one in enumerate(boxes):
        if not isinstance(one, str):
            bndboxes = one.tolist()
            imglabels = labels[id].tolist()

            t1 = img1[id, :, :, :]
            t2 = img2[id, :, :, :]
            t1 = tf.ToPILImage()(t1)
            t2 = tf.ToPILImage()(t2)

            for ii, box in enumerate(bndboxes):
                tmp1 = t1.crop((box[0], box[1], box[2], box[3]))
                tmp1 = tmp1.resize((160, 160), Image.BILINEAR)
                tmp2 = t2.crop((box[0], box[1], box[2], box[3]))
                tmp2 = tmp2.resize((160, 160), Image.BILINEAR)
                imgs.append([tmp1, tmp2])
                labelseq.append(imglabels[ii])
    if len(imgs) == 0:
        return None, None
    cropped_imgs = [[tf.ToTensor()(one[0]), tf.ToTensor()(one[1])] for one in
                    imgs]  # cropped imgs from one image for cutin

    l = torch.tensor(labelseq).cuda()

    return cropped_imgs, l