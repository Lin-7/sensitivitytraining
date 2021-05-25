import argparse
import logging
import sys
import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.wideresnet import WideResNet
from model.preactresnet import PreActResNet18
from utils import *

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit, lower_limit = 1,0

def normalize(X):
    return (X - mu)/std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        # 随机初始化
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        # 计算扰动
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def get_args():
    parser = argparse.ArgumentParser()
    # 待评估模型参数
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--width-factor', default=10, type=int)    # wideresnet的参数
    parser.add_argument('--data-dir', default='../Data/CIFAR10', type=str)
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='cosine', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam'])
    parser.add_argument('--lr-max', default=0.01, type=float)    # 最初的学习率
    parser.add_argument('--attack', default='noise', type=str, choices=['pgd', 'fgsm', 'free', 'none', 'noise'])            # attack=free的情况不理解????
    parser.add_argument('--weight-per', default='cWei', type=str, choices=['cWei', 'noiWei', 'advWei'])
    parser.add_argument('--weight-per-attack', default='pgd', type=str, choices=['pgd', 'fgsm'])   # 如果attack==none，weight-per==advWei，则由此参数决定产生参数对抗扰动的方法
    parser.add_argument('--Qneighbors', default=5, type=int)
    parser.add_argument('--sen-lambda', default=0.05, type=float)
    parser.add_argument('--sen-radius', default=7.65, type=float)

    # 评估参数
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    
    return parser.parse_args()
# 0.03-7.65   0.05-12.75  0.07-17.85  0.1-25.5  0.3-76.5

def main():
    args = get_args()

    # 评估模型和评估结果存放文件夹
    if args.attack == 'noise':
        if args.optimizer == 'SGD':
            args.fname = '{}/{}(Q{}-L{}-R{})-{}-{}({})-lr{}-epoch{}'.format(args.fname, args.attack, args.Qneighbors, \
                args.sen_lambda, args.sen_radius, args.weight_per, args.optimizer, args.lr_schedule, args.lr_max, args.epochs)
        elif args.optimizer == 'Adam':
            args.fname = '{}/{}(Q{}-L{}-R{})-{}-{}-lr{}'.format(args.fname, args.attack, args.Qneighbors, args.sen_lambda, args.sen_radius, args.weight_per, args.optimizer, args.lr_max)
    else:
        args.fname = '{}/{}-{}-epoch{}'.format(args.fname, args.attack, args.weight_per, args.epochs)
    if not os.path.exists(args.fname):
        raise ValueError("file does not exit")

    # 设置日志输出
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'evaluate_noise.log'))
        ])

    logger.info(args)

    # 设置随机种子   
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 获取用于测试的数据集
    dataset = cifar10(args.data_dir)
    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    # 加载模型并载入参数
    if args.model == 'PreActResNet18':
        model = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(args.fname, f'model_best.pth'))['state_dict'])
    criterion = nn.CrossEntropyLoss()
    
    # 噪声强度
    noise_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]    

    test_acc = 0
    test_loss = 0
    # 输出测试时的原始损失和准确率，以及鲁棒损失和准确率
    logger.info('Noise Strength \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    model.eval()
    for noise_strength in noise_list:
        test_robust_acc = 0
        test_robust_loss = 0
        test_n = 0
        for _, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']
            # 生成对应强度的噪声样本
            delta = torch.zeros_like(X)
            if args.norm == "l_inf":
                torch.nn.init.normal_(delta, mean=0, std=noise_strength)
            if args.norm == 'l_2':
                raise ValueError("Unimplemented noise perturbation generation method")
            # 使用生成的噪声样本对模型进行评估
            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            # 只需要评估一次模型的原始准确率和损失
            if noise_strength == noise_list[0]:
                output = model(normalize(X))
                loss = criterion(output, y)
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
        # 输出评估结果
        logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                noise_strength, test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

if __name__ == "__main__":
    main()
