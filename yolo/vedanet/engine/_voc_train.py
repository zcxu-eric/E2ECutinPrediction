import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import os
import numpy as np
from random import sample,shuffle
from PIL import Image
from .. import data
from .. import models
from . import engine

__all__ = ['VOCTrainingEngine']

class VOCDataset(data.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.trainfile
        root = hyper_params.data_root
        flip = hyper_params.flip
        jitter = hyper_params.jitter
        hue, sat, val = hyper_params.hue, hyper_params.sat ,hyper_params.val
        network_size = hyper_params.network_size
        labels = hyper_params.labels
        rf  = data.transform.RandomFlip(flip)
        rc  = data.transform.RandomCropLetterbox(self, jitter)
        hsv = data.transform.HSVShift(hue, sat, val)
        it  = tf.ToTensor()

        #img_tf = data.transform.Compose([rc, rf, hsv, it])
        img_tf = data.transform.Compose([rc, rf, hsv])
        anno_tf = data.transform.Compose([rc, rf])

        def identify(img_id):
            #return f'{root}/VOCdevkit/{img_id}.jpg'
            return f'{img_id}'

        super(VOCDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)



class VOCTrainingEngine(engine.Engine):
    """ This is a custom engine for this training cycle """

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        # all in args
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        self.max_batches = hyper_params.max_batches

        self.classes = hyper_params.classes

        self.cuda = hyper_params.cuda
        self.backup_dir = hyper_params.backup_dir
        self.cutin_pool = []
        log.debug('Creating network')
        model_name = hyper_params.model_name
        net = models.__dict__[model_name](hyper_params.weights, train_flag=1, clear=hyper_params.clear)
        log.info('Net structure\n\n%s\n' % net)
        if self.cuda:
            net.cuda()

        log.debug('Creating optimizer')
        learning_rate = hyper_params.learning_rate
        momentum = hyper_params.momentum
        decay = hyper_params.decay
        batch = hyper_params.batch
        log.info(f'Adjusting learning rate to [{learning_rate}]')
        optim = torch.optim.SGD(net.parameters(), lr=learning_rate/batch, momentum=momentum, dampening=0, weight_decay=decay*batch)

        log.debug('Creating dataloader')
        dataset = VOCDataset(hyper_params)
        dataloader = data.DataLoader(
            dataset,
            batch_size = self.mini_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = hyper_params.nworkers if self.cuda else 0,
            pin_memory = hyper_params.pin_mem if self.cuda else False,
            collate_fn = data.list_collate,
        )

        super(VOCTrainingEngine, self).__init__(net, optim, dataloader)

        #self.nloss = self.network.nloss

        #self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]

    def start(self):
        log.debug('Creating additional logging objects')
        hyper_params = self.hyper_params

        lr_steps = hyper_params.lr_steps
        lr_rates = hyper_params.lr_rates

        bp_steps = hyper_params.bp_steps
        bp_rates = hyper_params.bp_rates
        backup = hyper_params.backup

        rs_steps = hyper_params.rs_steps
        rs_rates = hyper_params.rs_rates
        resize = hyper_params.resize

        self.add_rate('learning_rate', lr_steps, [lr/self.batch_size for lr in lr_rates])
        self.add_rate('backup_rate', bp_steps, bp_rates, backup)
        self.add_rate('resize_rate', rs_steps, rs_rates, resize)

        self.dataloader.change_input_dim()

    def process_batch(self, data):
        loss = 0
        cropped_imgs, labels = self.cropped_img_generatir(data)
        if cropped_imgs == None:
            return
        for id, pair in enumerate(cropped_imgs):
            # to(device)
            if self.cuda:
                data1 = pair[0].cuda()
                data2 = pair[1].cuda()
            loss = self.network([data1,data2], labels[id])
            loss.backward()
        self.train_loss = float(loss.item())


    
    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        log.info(f'{self.batch}/{self.max_batches} CUTIN Loss: {self.train_loss}')



        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'weights_{self.batch}.pt'))

        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))

        if self.batch % self.resize_rate == 0:
            if self.batch + 200 >= self.max_batches:
                finish_flag = True
            else:
                finish_flag = False
            self.dataloader.change_input_dim(finish=finish_flag)

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_dir, f'final.dw'))
            return True
        else:
            return False

    def __build_targets_brambox(self, ground_truth, expand_ratio = 0.1):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = len(ground_truth)
        self.reduction = 1
        # Tensors
        GT = []
        L = []
        for b in range(nB):
            if len(ground_truth[b]) == 0:  # No gt for this image
                continue
            # Build up tensors

            gt = np.zeros((len(ground_truth[b]), 4))
            label = np.zeros(len(ground_truth[b]))
            #one img
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno.x_top_left) / self.reduction * (1.0 - expand_ratio)
                gt[i, 1] = (anno.y_top_left) / self.reduction * (1.0 - expand_ratio)
                gt[i, 2] = (anno.x_top_left + anno.width) / self.reduction * (1.0 + expand_ratio)
                gt[i, 3] = (anno.y_top_left + anno.height) / self.reduction * (1.0 + expand_ratio)
                if anno.cutin == 1.0:
                    label[i] = 1
            GT.append(gt)
            if len(L)!=0:
                L = [np.hstack((L[0],label))]
            else:
                L.append(label)
        if len(L) == 0 or not L:
            return GT, None
        else:
            return GT, L[0]

    def cutin_balance(self, cropped_imgs, labels):
        if len(cropped_imgs) == 0 or labels.shape==0:
            return
        else:
            nocut = []
            cut = []
            for id, one in enumerate(list(labels)):
                if one:
                    cut.extend([cropped_imgs[id]])
                    if len(self.cutin_pool>=1000):
                        self.cutin_pool = sample(self.cutin_pool, 500)
                    self.cutin_pool.extend([cropped_imgs[id]])
                else:
                    nocut.extend([cropped_imgs[id]])
            nocut = sample(nocut, int(0.5*len(nocut)))
            if len(nocut) > len(cut):
                if len(self.cutin_pool) > len(nocut)-len(cut):
                    cut.extend(sample(self.cutin_pool,len(nocut)-len(cut)))
                else:
                    cut.extend(sample(self.cutin_pool, len(self.cutin_pool)))
            imgs = nocut+cut
            lnocut = np.zeros(len(nocut))
            lcut = np.ones(len(cut))
            label = np.hstack((lnocut,lcut))
            ind = list(range(len(imgs)))
            com = list(zip(ind,imgs))
            shuffle(com)
            try:
                ind, imgs = zip(*com)
            except:
                print('ERROR!')
                return None, None
            imgs = list(imgs)
            label_new = label[list(ind)]
            label_t = torch.from_numpy(label_new).cuda()


        return imgs, label_t


    def cropped_img_generatir(self, data):

        img1, img2, target = data

        boxes, labels = self.__build_targets_brambox(target)

        imgs = []
        for id, one in enumerate(boxes):

            t1 = img1[id,:,:,:]
            t2 = img2[id,:,:,:]

            t1 = tf.ToPILImage()(t1)
            t2 = tf.ToPILImage()(t2)

            if not isinstance(one,int):
                bndboxes = one.tolist()
                for box in bndboxes:
                    tmp1 = t1.crop((box[0],box[1],box[2],box[3]))
                    tmp1 = tmp1.resize((160,160),Image.BILINEAR)
                    tmp2 = t2.crop((box[0], box[1], box[2], box[3]))
                    tmp2 = tmp2.resize((160, 160), Image.BILINEAR)
                    imgs.append([tmp1,tmp2])

        cropped_imgs = [[tf.ToTensor()(one[0]),tf.ToTensor()(one[1])] for one in imgs] #cropped imgs from one image for cutin
        try:
            cropped_imgs, labels = self.cutin_balance(cropped_imgs, labels)
            return cropped_imgs, labels
        except:
            return None, None