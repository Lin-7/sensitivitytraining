import argparse
import logging
import sys
import time
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.wideresnet import WideResNet
from model.preactresnet import PreActResNet18
from utils import *
from utils_wp import AdvWeightPerturb, NoiseWeightPerturb
from sample_method import *


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def eval_interval_schedule(t, epochs):
    if t / epochs <= 0.5:
        return 5
    elif t / epochs <= 0.75:
        return 10
    elif t / epochs <= 0.8:
        return 2
    else:
        return 1


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
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../Data/CIFAR10', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='cosine', choices=['cosineWR', 'superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam'])
    parser.add_argument('--lr-max', default=0.01, type=float)    # 最初的学习率
    parser.add_argument('--lr-one-drop', default=0.01, type=float)    # one-drop学习率
    parser.add_argument('--lr-drop-epoch', default=100, type=int)     # 变为one-drop学习率的epoch 
    parser.add_argument('--attack', default='noise', type=str, choices=['pgd', 'fgsm', 'free', 'none', 'noise'])            # attack=free的情况不理解????
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)        # pgd训练时换回10，训练时评估用20
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)    # 单次攻击步长
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)    # 单次攻击步长
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])    # 
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--chkpt-iters', default=100, type=int)
    # parser.add_argument('--sample-per', default='cSam', type=str, choices=['cSam', 'noiSam', 'advSam'])
    parser.add_argument('--weight-per', default='cWei', type=str, choices=['cWei', 'noiWei', 'advWei'])
    parser.add_argument('--weight-per-attack', default='pgd', type=str, choices=['pgd', 'fgsm'])   # 如果attack==none，weight-per==advWei，则由此参数决定产生参数对抗扰动的方法
    parser.add_argument('--Qneighbors', default=5, type=int)
    parser.add_argument('--sen-lambda', default=0.05, type=float)
    parser.add_argument('--sen-radius', default=7.65, type=float)
    parser.add_argument('--sample-method', default='lossvalue', type=str, choices=['random', 'confid', 'sens', 'grad', 'lossvalue'])
    parser.add_argument('--sample-criterion', default='Nearest', type=str, choices=['Furthest', 'Nearest'])
    parser.add_argument('--sample-rate', default=0.8, type=float)
    
    parser.add_argument('--l2', default=0, type=float)       # about weight decay，默认false
    parser.add_argument('--l1', default=0, type=float)       # about loss value 是否加上非bias非norm的权重值，默认false
    parser.add_argument('--wp-gamma', default=0.01, type=float)    # 控制权重噪声的缩放系数（需为正数
    parser.add_argument('--wp-warmup', default=0, type=int)        # 开始加入权重噪声的epoch
    parser.add_argument('--cutout', action='store_true')      # 图像预处理，默认false
    parser.add_argument('--cutout-len', type=int)             # 图像预处理
    parser.add_argument('--width-factor', default=10, type=int)    # wideresnet的参数
    return parser.parse_args()
# 0.03-7.65   0.05-12.75  0.07-17.85  0.1-25.5  0.3-76.5

def main():
    args = get_args()
    # wp_gamma需大于0，否则不会进行权重扰动
    if args.wp_gamma <= 0.0:
        args.wp_warmup = np.infty

    # 训练信息输出文件夹
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
    if args.attack == 'noise':
        if args.optimizer == 'SGD':
            args.fname = '{}/{}(Q{}-L{}-R{})-{}-{}({})-lr{}-epoch{}-{}({})-seed257'.format(args.fname, args.attack, args.Qneighbors, \
                args.sen_lambda, args.sen_radius, args.weight_per, args.optimizer, args.lr_schedule, args.lr_max, args.epochs, args.sample_method, args.sample_rate)
        else:
            raise ValueError('check your optimizer..')
    else:
        args.fname = '{}/{}-{}-{}({})'.format(args.fname, args.attack, args.weight_per, args.sample_method, args.sample_rate)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    # 设置日志输出
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'output.log')),
            # logging.StreamHandler()      # 只输出到上面的日志中，不输出到输出流中
        ])

    logger.info(args)

    # 设置随机种子   
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 获取训练数据以及预处理
    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    # 调整扰动步距和扰动预算
    epsilon = (args.epsilon / 255.)
    radius = (args.sen_radius / 255)
    pgd_alpha = (args.pgd_alpha / 255.)         # 没有fgsm_alpha=(args.fgsm_alpha / 255.)?
    fgsm_alpha=(args.fgsm_alpha / 255.)       # 源代码没有，后面补充的

    # 加载模型
    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        if args.weight_per == 'advWei':
            proxy = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
        if args.weight_per == 'advWei':
            proxy = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    if args.weight_per == 'advWei':
        proxy = nn.DataParallel(proxy).cuda()

    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    # 优化函数,扰动对象
    if args.optimizer == 'SGD':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if args.weight_per == 'advWei':
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
        wp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.wp_gamma)
    if args.weight_per == 'noiWei':
        wp_adversary = NoiseWeightPerturb(model=model, gamma=args.wp_gamma)
    criterion = nn.CrossEntropyLoss()
    
    # # 设置学习策略
    # if args.lr_schedule == 'cosineWR':
    #     lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=0)
    # elif args.lr_schedule == 'cosine':
    #     pass
    # elif args.lr_schedule == 'piecewise':
    #     pass
    # elif args.lr_schedule == 'linear':
    #     pass    
    # elif args.lr_schedule == 'onedrop':
    #     pass
    # elif args.lr_schedule == 'multipledecay':
    #     pass
    # elif args.lr_schedule == 'cyclic':
    #     pass

    # 初始化扰动?
    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    # 设置学习策略
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))    # args.epochs为200时，把这里的args.epochs改成50.lr就会有两个余弦周期的变化
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]

    best_test_robust_acc = 0
    best_test_robust_acc_epoch = 0
    best_val_robust_acc = 0
    # 是否重新启动
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        lr_schedule.load_state_dict(torch.load(os.path.join(args.fname, f'lr_schedule_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 0

    if args.optimizer == 'Adam':
        lr = args.lr_max

    # 输出训练准确率和损失，以及测试时的原始和鲁棒准确率
    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Training Loss \t Training Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        training_loss = 0
        training_acc = 0
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0

        s_s_time = time.time()
        # 采样
        if args.sample_rate == 1.0 or args.sample_method == 'random':
            lenx = np.arange(0, len(train_set), 1)
            sample_index = random.sample(list(lenx), int(args.sample_rate * len(train_set)))
        else:
            if args.sample_method == 'sens':
                sample_index = select_sample_CY(model, train_batches, 3, args.sample_rate, args.sample_criterion, radius)
            if args.sample_method == 'confid':
                sample_index = select_sample_CP(model, train_batches, args.sample_rate, args.sample_criterion)
            if args.sample_method == 'grad':
                sample_index = select_sample_DY(model, train_batches, args.sample_rate, args.sample_criterion)
            if args.sample_method == 'lossvalue':
                CEloss = nn.CrossEntropyLoss(reduction='none')
                sample_index = select_sample_LV(model, train_batches, args.sample_rate, args.sample_criterion, CEloss)
        sample_index.sort()
        e_s_time = time.time()
        logger.info('%.10f', e_s_time-s_s_time)

        model.train()
        for batch_idx, batch in enumerate(train_batches):
            # logger.info('%d', batch_idx)
            # 获取并处理这一批次的样本
            X, y = batch['input'], batch['target']
            if args.optimizer == 'SGD':
                lr = lr_schedule(epoch + (batch_idx + 1) / len(train_batches))
                opt.param_groups[0].update(lr=lr)
            
            # 确定这一个batch中需要被采样的样本下标
            mark_index = []
            for index in range(len(X)):
                if sample_index != [] and index + train_batches.batch_size * batch_idx == sample_index[0]:
                    mark_index.append(index)
                    sample_index.pop(0)

            # 攻击样本
            if args.attack == 'pgd':
                delta = attack_pgd(model, X[mark_index], y[mark_index], epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta.detach()
                X_adv = normalize(torch.clamp(X[mark_index] + delta, min=lower_limit, max=upper_limit))
                # X_adv_part = normalize(torch.clamp(X[mark_index] + delta, min=lower_limit, max=upper_limit))
                # X_adv = X.clone()
                # X_adv[mark_index] = X_adv_part
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X[mark_index], y[mark_index], epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
                delta.detach()
                X_adv = normalize(torch.clamp(X[mark_index] + delta, min=lower_limit, max=upper_limit))
                # X_adv_part = normalize(torch.clamp(X[mark_index] + delta, min=lower_limit, max=upper_limit))
                # X_adv = X.clone()
                # X_adv[mark_index] = X_adv_part
            elif args.attack == 'none':
                # Standard training
                X_adv = X
            elif args.attack == 'noise':
                X_adv_list = []
                for i in range(args.Qneighbors):
                    delta = torch.zeros_like(X[mark_index])
                    if args.norm == "l_inf":
                        torch.nn.init.normal_(delta, mean=0, std=radius)
                    if args.norm == 'l_2':
                        raise ValueError("Unimplemented perturbation generation method")
                    X_adv_list.append(normalize(torch.clamp(X[mark_index] + delta[:len(mark_index)], min=lower_limit, max=upper_limit)))

            # 攻击权重
            if epoch >= args.wp_warmup:
                if args.weight_per == 'advWei':
                    if args.attack == 'pgd' or args.attack == 'fgsm':
                        per_weight_X = X_adv
                    else:
                        if args.weight_per_attack == 'pgd':
                            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                        elif args.weight_per_attack == 'fgsm':
                            delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
                        delta = delta.detach()
                        per_weight_X = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
                    # calculate adversarial weight perturbation and perturb it
                    awp = wp_adversary.calc_awp(inputs_adv=per_weight_X,
                                                 targets=y)
                    wp_adversary.perturb(awp)
                if args.weight_per == 'noiWei':
                    nwp = wp_adversary.calc_nwp()
                    wp_adversary.perturb(nwp)

            # 计算鲁棒性损失
            if args.attack == 'noise':
                train_output = model(normalize(X))
                sen = 0
                if mark_index != []:
                    for i in range(args.Qneighbors):
                        sen_output = model(X_adv_list[i])
                        sen += torch.dist(train_output[mark_index], sen_output, 2)
                loss = criterion(train_output, y)
                loss = (1-args.sen_lambda) * loss + args.sen_lambda * sen / args.Qneighbors
            else:
                train_output_adv = model(X_adv)   # 只使用了对抗样本
                train_output = model(X)
                loss = criterion(train_output_adv, y[mark_index]) + criterion(train_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        loss += args.l1*param.abs().sum()      # 权重正则

            # 利用鲁棒性损失更新权重
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 清除鲁棒性扰动
            if epoch >= args.wp_warmup:
                if args.weight_per == 'advWei':
                    wp_adversary.restore(awp)
                if args.weight_per == 'noiWei':
                    wp_adversary.restore(nwp)
            
            # 记录训练准确率和训练损失
            training_loss += loss.item() * y.size(0)
            training_acc += (train_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        # lr = opt.param_groups[0]['lr']
        # lr_schedule.step()

        train_time = time.time()

        eval_interval = eval_interval_schedule(epoch, args.epochs)
        if epoch % eval_interval == 0:
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            for batch_idx, batch in enumerate(test_batches):
                X, y = batch['input'], batch['target']

                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=False)
                delta = delta.detach()
                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            test_time = time.time()

            # 输出训练准确率和损失，以及测试时的原始和鲁棒准确率
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                training_loss/train_n, training_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
                best_test_robust_acc_epoch = epoch
        else:
            # 输出训练准确率和损失，以及测试时的原始和鲁棒准确率
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t \t %.4f', epoch, train_time - start_time, 0.0, lr, training_loss/train_n, training_acc/train_n)

        # save checkpoint
        if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
            torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))
            # torch.save(lr_schedule.state_dict(), os.path.join(args.fname, f'lr_schedule_{epoch}.pth'))
    logger.info('best test robust epoch:%d \t test robust acc:%.4f', best_test_robust_acc_epoch, best_test_robust_acc)


if __name__ == "__main__":
    main()
