import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers

from matplotlib import pyplot as plt

import numpy as np



def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler

def save_imgs2(img, msk, msk_pred, x , i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    msk = msk.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    msk = msk / 255. if msk.max() > 1.1 else msk

    msk_pred = msk_pred.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    msk_pred = msk_pred / 255. if msk_pred.max() > 1.1 else msk_pred

    x = x.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    x = x / 255. if x.max() > 1.1 else x


    plt.figure(figsize=(7,15))

    plt.subplot(4,1,1)
    plt.imshow(img[:, :, 0], extent=[0, 16, 0, 16])
    plt.axis('off')

    plt.subplot(4,1,2)
    plt.imshow(msk[:, :, 0], extent=[0, 16, 0, 16])
    plt.axis('off')

    plt.subplot(4,1,3)
    plt.imshow(msk_pred[:, :, 0], extent=[0, 16, 0, 16])
    plt.axis('off')

    plt.subplot(4,1,4)
    plt.imshow(x[:, :, 0], extent=[0, 16, 0, 16])
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()


def save_imgs_explainable(ig,img,ig2,ig3,ig4,ig5, i,a, save_path, datasets, threshold=0.5, test_data_name=None):
    ig = ig.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    ig = ig / 255. if ig.max() > 1.1 else ig

    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    ig2 = ig2.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    ig2 = ig2 / 255. if ig2.max() > 1.1 else ig2

    ig3 = ig3.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    ig3 = ig3 / 255. if ig3.max() > 1.1 else ig3

    ig4 = ig4.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    ig4 = ig4 / 255. if ig4.max() > 1.1 else ig4

    ig5 = ig5.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    ig5 = ig5 / 255. if ig5.max() > 1.1 else ig5

    images=[]

    plt.figure(figsize=(7,15))

    plt.subplot(6,1,1)
    plt.imshow(ig)
    plt.axis('off')

    plt.subplot(6,1,2)
    plt.imshow(img[:, :, 0], extent=[0, 32, 0, 32])
    plt.axis('off')
    images.append(img)

    plt.subplot(6,1,3)
    plt.imshow(ig2[:, :, 0], extent=[0, 32, 0, 32])
    plt.axis('off')
    images.append(ig2)

    plt.subplot(6,1,4)
    plt.imshow(ig3[:, :, 0], extent=[0, 32, 0, 32])
    plt.axis('off')
    images.append(ig3)

    plt.subplot(6,1,5)
    plt.imshow(ig4[:, :, 0], extent=[0, 32, 0, 32])
    plt.axis('off')
    images.append(ig4)

    plt.subplot(6,1,6)
    plt.imshow(ig5[:, :, 0], extent=[0, 32, 0, 32])
    plt.axis('off')
    images.append(ig5)

    # EICA
    """
    Explainable Inference Classification Algorithm
    Created on Wed Nov 15 14:23:43 2023
    @author: Renkai Wu
    """
    img_gray = (img[:, :, 0] * 255).astype(np.uint8)
    ig2_gray = (ig2[:, :, 0] * 255).astype(np.uint8)
    ig3_gray = (ig2[:, :, 0] * 255).astype(np.uint8)
    ig4_gray = (ig4[:, :, 0] * 255).astype(np.uint8)
    ig5_gray = (ig5[:, :, 0] * 255).astype(np.uint8)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    # img
    quadrant_1 = img_gray[:center_y, center_x:]
    quadrant_2 = img_gray[:center_y, :center_x]
    quadrant_3 = img_gray[center_y:, :center_x]
    quadrant_4 = img_gray[center_y:, center_x:]

    # ig2
    ig2quadrant_1 = ig2_gray[:center_y, center_x:]
    ig2quadrant_2 = ig2_gray[:center_y, :center_x]
    ig2quadrant_3 = ig2_gray[center_y:, :center_x]
    ig2quadrant_4 = ig2_gray[center_y:, center_x:]

    # ig3
    ig3quadrant_1 = ig4_gray[:center_y, center_x:]
    ig3quadrant_2 = ig4_gray[:center_y, :center_x]
    ig3quadrant_3 = ig4_gray[center_y:, :center_x]
    ig3quadrant_4 = ig4_gray[center_y:, center_x:]

    # ig4
    ig4quadrant_1 = ig5_gray[:center_y, center_x:]
    ig4quadrant_2 = ig5_gray[:center_y, :center_x]
    ig4quadrant_3 = ig5_gray[center_y:, :center_x]
    ig4quadrant_4 = ig5_gray[center_y:, center_x:]

    # ig5
    ig6quadrant_1 = ig3_gray[:center_y, center_x:]
    ig6quadrant_2 = ig3_gray[:center_y, :center_x]
    ig6quadrant_3 = ig3_gray[center_y:, :center_x]
    ig6quadrant_4 = ig3_gray[center_y:, center_x:]

    threshold = 225

    condition_1 = np.max(quadrant_1) > threshold or np.max(quadrant_2) > threshold
    condition_2 = np.max(ig2quadrant_4) > threshold
    condition_3 = np.max(ig3quadrant_1) > threshold
    condition_4 = np.max(ig4quadrant_2) > threshold or np.max(ig4quadrant_3) > threshold

    conditions_met = sum([condition_1, condition_2, condition_3, condition_4])

    output = 1 if conditions_met >= 4 else 0
    print(output)
    s = a + output


    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()

    return s


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()
    


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss



class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


class myNormalize:
    def __init__(self, data_name, train=True,test=False):
        if data_name == 'isic17':
            if train:
                self.mean = 156.9704
                self.std = 27.8712
            elif test:
                self.mean = 156.3398
                self.std = 28.2681
            else:
                self.mean = 159.0763
                self.std = 29.6363
        elif data_name == 'isic18':
            if train:
                self.mean = 154.7468
                self.std = 28.6875
            elif test:
                self.mean = 156.6547
                self.std = 28.5943
            else:
                self.mean = 156.0735
                self.std = 28.9695
        elif data_name == 'PH2':
            if train:
                self.mean = 154.8111
                self.std = 41.3169
            elif test:
                self.mean = 154.3235
                self.std = 40.8471
            else:
                self.mean = 154.3235
                self.std = 40.8471
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk

        
        
        
