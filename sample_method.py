import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CLIP_MIN = 0
CLIP_MAX = 1

def generate_noise_samples_and_output(model, train_batches, noise_samples_per_sample, radius):
    model.eval()
    flag = 1
    length = 0
    for _, batch in enumerate(train_batches):
        x, y = batch['input'], batch['target']
        # 构造noise_samples_per_sample批噪声样本
        noise_sample_list = []
        for i in range(noise_samples_per_sample):
            one_batch_noise = torch.nn.init.normal_(
                torch.Tensor(len(x), x.size(1), x.size(2), x.size(3)), mean=0,
                std=radius)
            # 将噪声信号加到样本上
            one_batch_noise_sample = torch.add(x, 1, one_batch_noise, out=None)
            one_batch_noise_sample = torch.clamp(one_batch_noise_sample, min=CLIP_MIN, max=CLIP_MAX,
                                                 out=None)
            noise_sample_list.append(one_batch_noise_sample)
        # 得到噪声样本和原始样本的输出
        noise_sample_out = []
        out = model(x)
        for i in range(noise_samples_per_sample):
            noise_sample_out.append(model(noise_sample_list[i]).detach().cpu())
        length += len(x)
        # 拼接结果
        if flag:
            Y, Output = y, out.detach().cpu()
            Noise_sample_out = noise_sample_out
            flag = 0
        else:
            Y = torch.cat((Y, y), 0)
            Output = torch.cat((Output, out.detach().cpu()), 0)
            for i in range(noise_samples_per_sample):
                Noise_sample_out[i] = torch.cat((Noise_sample_out[i], noise_sample_out[i]), 0)
    return Y, Output, Noise_sample_out, length


def generate_all_ouput(model, train_batches):
    model.eval()
    flag = 1
    length = 0
    for _, batch in enumerate(train_batches):
        x, y = batch['input'], batch['target']
        out = model(x)
        length += len(x)
        if flag:
            Y, Output = y.cpu(), out.detach().cpu()
            flag = 0
        else:
            Y = torch.cat((Y, y.cpu()), 0)
            Output = torch.cat((Output, out.detach().cpu()), 0)
    return Y, Output, length


# type1 根据灵敏度抽样
def select_sample_CY(model, train_batches, noise_samples_count, sample_rate, sample_criterion, radius):
    Y, Output, Noise_out, _ = generate_noise_samples_and_output(model, train_batches, noise_samples_count, radius)

    # 确定抽样评估值(ed)
    dy = torch.norm(Noise_out[0] - Output, 2, 1).view(-1, 1)
    for i in range(1, noise_samples_count):
        dy = torch.cat((dy, torch.norm(Noise_out[i] - Output, 2, 1).view(-1, 1)), 1)
    dy = torch.mean(dy, 1)

    # 确定分类错误的样本下标
    wrong_index = np.arange(0, len(dy))[Output.max(1)[1] != Y].tolist()
    if sample_criterion == 'Furthest':
        # 将分类错的样本的抽样概率设置为最低值1000
        dy[wrong_index] = 1000
        # 选Furthest的样本，即随deta x的改变变化最小的dy
        sample_value, sample_index = torch.topk(dy, int(abs(sample_rate) * len(out_put)), 0, largest=False)
    if sample_criterion == 'Nearest':
        # 将分类错的样本的抽样概率设置为最低值0
        dy[wrong_index] = 0
        # 选Nearest的样本
        sample_value, sample_index = torch.topk(dy, int(abs(sample_rate) * len(out_put)), 0, largest=True)

    sample_index.sort()
    return sample_index.tolist()   # 转换为list后面才可以使用pop属性


# type2 根据输出对输入的梯度抽样
def select_sample_DY(model, train_batches, sample_rate, sample_criterion):
    model.eval()
    flag = 1
    for _, batch in enumerate(train_batches):
        x, y = batch['input'], batch['target']
        x.requires_grad_()
        out = model(x)
        # optimizer.zero_grad()
        out.backward(torch.ones(out.size()))
        if flag:
            x_grad = x.grad.cpu()
            Y, Output = y, out.detach().cpu()
            flag = 0
        else:
            x_grad = torch.cat((x_grad, x.grad.cpu()), 0)
            Y = torch.cat((Y, y), 0)
            Output = torch.cat((Output, out.detach().cpu()), 0)
    x_grad = torch.sum(torch.abs(x_grad), dim=[1, 2, 3])
    # 确定分类错的样本下标
    wrong_index = np.arange(0, len(x_grad))[Output.max(1)[1]!=Y].tolist()
    if sample_criterion == 'Nearest':
        x_grad[wrong_index] = 0
        detay, sample_index = torch.topk(x_grad, int(sample_rate * len(x_grad)), dim=0, largest=True)
    if sample_criterion == 'Furthest':
        x_grad[wrong_index] = 1000
        detay, sample_index = torch.topk(x_grad, int(sample_rate * len(x_grad)), dim=0, largest=False)
    sample_index.sort()
    return sample_index.tolist()   # 转换为list后面才可以使用pop属性


# type3 根据类别概率抽样
def select_sample_CP(model, train_batches, sample_rate, sample_criterion):
    # 获得所有输出
    softmax = nn.Softmax(dim=1)
    Y, Output, length = generate_all_ouput(model, train_batches)

    # 计算出采样平均值
    out_sorted, _ = torch.sort(softmax(Output), 1, True)  # sorted by row  decede
    out_sorted_sliped = torch.chunk(out_sorted, chunks=10, dim=1)  # slip the tensor to 10 item by  column
    Probability_ration = torch.add(out_sorted_sliped[0], -1, out_sorted_sliped[1])                            # a-b / a

    # 确定分类错的样本下标
    wrong_index = np.arange(0, len(Probability_ration))[Output.max(1)[1]!=Y].tolist()
    # 抽样
    if sample_criterion == 'Nearest':
        Probability_ration[wrong_index] = 1
        sample_value, sample_index = torch.topk(Probability_ration, int(abs(sample_rate) * len(Probability_ration)), 0, False)
    if sample_criterion == 'Furthest':
        Probability_ration[wrong_index] = 0
        sample_value, sample_index = torch.topk(Probability_ration, int(abs(sample_rate) * len(Probability_ration)), 0, True)
    
    sample_index.sort()
    return sample_index.tolist()   # 转换为list后面才可以使用pop属性


# type4 根据损失值抽样
def select_sample_LV(model, train_batches, sample_rate, sample_criterion, CEloss):
    # 获得所有输出
    Y, Output, length = generate_all_ouput(model, train_batches)

    # 计算出抽样评估值
    loss_value = CEloss(Output, Y)

    # 确定分类错的样本下标
    wrong_index = np.arange(0, len(loss_value))[Output.max(1)[1]!=Y].tolist()
    # 抽样
    if sample_criterion == 'Nearest':
        loss_value[wrong_index] = 0
        sample_value, sample_index = torch.topk(loss_value, int(abs(sample_rate) * len(loss_value)), 0, True)
    if sample_criterion == 'Furthest':
        loss_value[wrong_index] = 1000
        sample_value, sample_index = torch.topk(loss_value, int(abs(sample_rate) * len(loss_value)), 0, False)
    
    sample_index.sort()
    return sample_index.tolist()   # 转换为list后面才可以使用pop属性

