
import torch
from torchvision import transforms as tf

from PIL import Image

#__all__ = ['CropTools']

def __build_targets_brambox(ground_truth, expand_ratio=0.1):
    """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
    # Parameters
    nB = len(ground_truth)
    reduction = 1
    # Tensors
    GT = []
    L = []
    for b in range(nB):
        if len(ground_truth[b]) == 0:  # No gt for this image
            GT.append(0)
            continue
        # Build up tensors

        gt = torch.zeros(len(ground_truth[b]), 4, device='cuda')
        label = torch.zeros(len(ground_truth[b]), device='cuda')
        for i, anno in enumerate(ground_truth[b]):
            gt[i, 0] = (anno.x_top_left) / reduction * (1.0 - expand_ratio)
            gt[i, 1] = (anno.y_top_left) / reduction * (1.0 - expand_ratio)
            gt[i, 2] = (anno.x_top_left + anno.width) / reduction * (1.0 + expand_ratio)
            gt[i, 3] = (anno.y_top_left + anno.height) / reduction * (1.0 + expand_ratio)
            if anno.cutin == 1.0:
                label[i] = 1
        gt.cuda()
        GT.append(gt)

        label.cuda()
        if len(L) != 0:
            L = [torch.cat((L[0], label), 0)]
        else:
            L.append(label)
    if len(L) == 0:
        return GT, None
    else:
        return GT, L[0]


def cropped_img_generatir(data):
    img1, img2, target = data

    boxes, labels = __build_targets_brambox(target)

    imgs = []
    for id, one in enumerate(boxes):

        t1 = img1[id, :, :, :]
        t2 = img2[id, :, :, :]

        t1 = tf.ToPILImage()(t1)
        t2 = tf.ToPILImage()(t2)

        if isinstance(one, torch.Tensor):
            bndboxes = one.cpu().numpy().tolist()
            for box in bndboxes:
                tmp1 = t1.crop((box[0], box[1], box[2], box[3]))
                tmp1 = tmp1.resize((160, 160), Image.BILINEAR)
                tmp2 = t2.crop((box[0], box[1], box[2], box[3]))
                tmp2 = tmp2.resize((160, 160), Image.BILINEAR)
                imgs.append([tmp1, tmp2])

    cropped_imgs = [[tf.ToTensor()(one[0]), tf.ToTensor()(one[1])] for one in imgs]  # cropped imgs from one image for cutin

    return cropped_imgs, labels