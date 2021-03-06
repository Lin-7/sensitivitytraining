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
from torch.autograd import Variable

# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method as fgsm
# from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd

from model.wideresnet import WideResNet
from model.preactresnet import PreActResNet18
from utils import *
from utils_wp import AdvWeightPerturb, NoiseWeightPerturb

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu)/std


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def eval_interval_schedule(t, epochs):
    if t / epochs <= 0.5:
        return 30
    elif t / epochs <= 0.75:
        return 1
    elif t / epochs <= 0.8:
        return 1
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
        # ???????????????
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
        # ????????????
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
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam'])
    parser.add_argument('--lr-max', default=0.1, type=float)    # ??????????????????
    parser.add_argument('--lr-one-drop', default=0.01, type=float)    # one-drop?????????
    parser.add_argument('--lr-drop-epoch', default=100, type=int)     # ??????one-drop????????????epoch 
    parser.add_argument('--attack', default='noise', type=str, choices=['pgd', 'fgsm', 'free', 'none', 'noise'])            # attack=free??????????????????????
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)        # pgd???????????????10?????????????????????20
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)    # ??????????????????
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)    # ??????????????????
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])    # 
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--chkpt-iters', default=100, type=int)
    # parser.add_argument('--sample-per', default='cSam', type=str, choices=['cSam', 'noiSam', 'advSam'])
    parser.add_argument('--weight-per', default='cWei', type=str, choices=['cWei', 'noiWei', 'advWei'])
    parser.add_argument('--weight-per-attack', default='pgd', type=str, choices=['pgd', 'fgsm'])   # ??????attack==none???weight-per==advWei?????????????????????????????????????????????????????????
    parser.add_argument('--Qneighbors', default=5, type=int)
    parser.add_argument('--sen-lambda', default=0.1, type=float)
    parser.add_argument('--sen-radius', default=25.5, type=float)
    
    parser.add_argument('--l2', default=0, type=float)       # about weight decay?????????false
    parser.add_argument('--l1', default=0, type=float)       # about loss value ???????????????bias???norm?????????????????????false
    parser.add_argument('--wp-gamma', default=0.01, type=float)    # ????????????????????????????????????????????????
    parser.add_argument('--wp-warmup', default=0, type=int)        # ???????????????????????????epoch
    parser.add_argument('--cutout', action='store_true')      # ????????????????????????false
    parser.add_argument('--cutout-len', type=int)             # ???????????????
    parser.add_argument('--width-factor', default=10, type=int)    # wideresnet?????????
    return parser.parse_args()
# 0.03-7.65   0.05-12.75  0.07-17.85  0.1-25.5  0.3-76.5

def main():
    args = get_args()
    # wp_gamma?????????0?????????????????????????????????
    if args.wp_gamma <= 0.0:
        args.wp_warmup = np.infty

    # ???????????????????????????
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
    if args.attack == 'noise':
        if args.optimizer == 'SGD':
            args.fname = '{}/{}(Q{}-L{}-R{})-{}-{}({})-lr{}-epoch{}'.format(args.fname, args.attack, args.Qneighbors, \
                args.sen_lambda, args.sen_radius, args.weight_per, args.optimizer, args.lr_schedule, args.lr_max, args.epochs)
        elif args.optimizer == 'Adam':
            args.fname = '{}/{}(Q{}-L{}-R{})-{}-{}-lr{}'.format(args.fname, args.attack, args.Qneighbors, args.sen_lambda, args.sen_radius, args.weight_per, args.optimizer, args.lr_max)
    else:
        args.fname = '{}/{}-{}-epoch{}'.format(args.fname, args.attack, args.weight_per, args.epochs)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    # ??????????????????
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'output.log')),
            # logging.StreamHandler()      # ?????????????????????????????????????????????????????????
        ])

    logger.info(args)

    # ??????????????????   
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ?????????????????????????????????
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

    # ?????????????????????????????????
    epsilon = (args.epsilon / 255.)
    radius = (args.sen_radius / 255)
    pgd_alpha = (args.pgd_alpha / 255.)         # ??????fgsm_alpha=(args.fgsm_alpha / 255.)?
    fgsm_alpha=(args.fgsm_alpha / 255.)       # ?????????????????????????????????

    # ????????????
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

    # ????????????,????????????
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

    # ????????????????
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

    # ??????????????????
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
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))    # args.epochs???200??????????????????args.epochs??????50.lr????????????????????????????????????
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]

    best_test_robust_acc = 0
    best_test_robust_acc_epoch = 0
    best_val_robust_acc = 0
    # ??????????????????
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 0

    # ???????????????????????????????????????????????????
    # logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    # ???????????????????????????????????????????????????????????????????????????
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
        model.train()
        for i, batch in enumerate(train_batches):
            # ????????????????????????????????????
            X, y = batch['input'], batch['target']
            if args.optimizer == 'SGD':
                lr = lr_schedule(epoch + (i + 1) / len(train_batches))
                opt.param_groups[0].update(lr=lr)
            if args.optimizer == 'Adam':
                lr = args.lr_max

            # ????????????
            # ?????????????????????????????????
            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta.detach()
                X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
                # # cleverhans
                # X_adv = pgd(model, X, epsilon, pgd_alpha, args.attack_iters, np.inf if args.norm=='l_inf' else 2, clip_min=lower_limit, clip_max=upper_limit, rand_init=True)
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
                delta.detach()
                X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
                # # cleverhans
                # X_adv = fgsm(model, X, epsilon, np.inf if args.norm=='l_inf' else 2, clip_min=lower_limit, clip_max=upper_limit)
            elif args.attack == 'none':
                # Standard training
                X_adv = X
            elif args.attack == 'noise':
                X_adv_list = []
                for i in range(args.Qneighbors):
                    delta = torch.zeros_like(X)
                    if args.norm == "l_inf":
                        torch.nn.init.normal_(delta, mean=0, std=radius)
                    if args.norm == 'l_2':
                        raise ValueError("Unimplemented perturbation generation method")
                    X_adv_list.append(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))

            # ????????????
            if epoch >= args.wp_warmup:
                if args.weight_per == 'advWei':
                    if args.attack == 'pgd' or args.attack == 'fgsm':
                        per_weight_X = X_adv
                    else:
                        if args.weight_per_attack == 'pgd':
                            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                            # # cleverhans
                            # X_adv = pgd(model, X, epsilon, pgd_alpha, args.attack_iters, np.inf if args.norm=='l_inf' else 2, clip_min=lower_limit, clip_max=upper_limit, rand_init=True)
                        elif args.weight_per_attack == 'fgsm':
                            delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
                            # # cleverhans
                            # X_adv = fgsm(model, X, epsilon, np.inf if args.norm=='l_inf' else 2, clip_min=lower_limit, clip_max=upper_limit)
                        delta = delta.detach()
                        per_weight_X = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
                    # calculate adversarial weight perturbation and perturb it
                    awp = wp_adversary.calc_awp(inputs_adv=per_weight_X,
                                                 targets=y)
                    wp_adversary.perturb(awp)
                if args.weight_per == 'noiWei':
                    nwp = wp_adversary.calc_nwp()
                    wp_adversary.perturb(nwp)

            # ?????????????????????
            if args.attack == 'noise':
                train_output = model(normalize(X))
                sen = 0
                for i in range(args.Qneighbors):
                    sen_output = model(X_adv_list[i])
                    sen += torch.dist(train_output, sen_output, 2)
                loss = criterion(train_output, y)
                loss = (1-args.sen_lambda) * loss + args.sen_lambda * sen / args.Qneighbors
            else:
                train_output = model(X_adv)   # ????????????????????????
                loss = criterion(train_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        loss += args.l1*param.abs().sum()      # ????????????

            # ?????????????????????????????????
            opt.zero_grad()
            loss.backward()
            opt.step()

            # ?????????????????????
            if epoch >= args.wp_warmup:
                if args.weight_per == 'advWei':
                    wp_adversary.restore(awp)
                if args.weight_per == 'noiWei':
                    wp_adversary.restore(nwp)
            
            # # ?????????????????????????????????
            # model.eval()
            # origin_output = model(normalize(X))
            # orgin_loss = criterion(origin_output, y)
            # # ??????chkpt_iters?????????????????????????????????
            # if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
            #     X_adv = pgd(model, X, epsilon, pgd_alpha, args.attack_iters_test, np.inf if args.norm=='l_inf' else 2, clip_min=lower_limit, clip_max=upper_limit, rand_init=True)
            #     robust_output = model(normalize(X_adv))
            #     robust_loss = criterion(robust_output, y)

            #     train_robust_loss += robust_loss.item() * y.size(0)
            #     train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            # train_loss += origin_loss.item() * y.size(0)
            # train_acc += (origin_output.max(1)[1] == y).sum().item()
            # train_n += y.size(0)
            # ????????????????????????????????????
            training_loss += loss.item() * y.size(0)
            training_acc += (train_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        eval_interval = eval_interval_schedule(epoch, args.epochs)
        if epoch % eval_interval == 0:
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            for i, batch in enumerate(test_batches):
                X, y = batch['input'], batch['target']

                # # cleverhans
                # X_adv = pgd(model, X, epsilon, pgd_alpha, args.attack_iters_test, np.inf if args.norm=='l_inf' else 2, clip_min=lower_limit, clip_max=upper_limit, y=y, rand_init=True)
                # robust_output = model(normalize(X_adv))
                # robust_loss = criterion(robust_output, y)

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

            # ???????????????????????????????????????????????????
            # logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
            #     epoch, train_time - start_time, test_time - train_time, lr,
            #     train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
            #     test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            # ???????????????????????????????????????????????????????????????????????????
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
            # ???????????????????????????????????????????????????
            # logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f',
            #     epoch, train_time - start_time, test_time - train_time, lr,
            #     train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n)
            # ???????????????????????????????????????????????????????????????????????????
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t \t %.4f', epoch, train_time - start_time, 0.0, lr, training_loss/train_n, training_acc/train_n)

        # save checkpoint
        if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
            torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))
    logger.info('best test robust epoch:%d \t test robust acc:%.4f', best_test_robust_acc_epoch, best_test_robust_acc)


if __name__ == "__main__":
    main()
