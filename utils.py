import copy

import numpy as np
import torch
from PIL import Image
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss

class DiceLoss(nn.Module):#骰子损失，通常用于评估图像分割任务中的预测结果和真实标签之间的相似度。该损失函数的计算基于预测结果和真实标签的交集和并集，通常用于解决图像分割任务中类别不平衡的问题。
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()#调用了父类的初始化方法
        self.n_classes = n_classes#将传入的n_classes赋值给类的成员变量self.n_classes。

    def _one_hot_encoder(self, input_tensor):#对输入的张量进行读入编码
        tensor_list = []
        for i in range(self.n_classes):#在n_classes-1内循环
            temp_prob = input_tensor == i  # 将输入张量中的每个值转换为布尔类型的向量。
            tensor_list.append(temp_prob.unsqueeze(1))#使用 PyTorch 的 unsqueeze() 函数将 temp_prob 张量的维度扩展为 (batch_size, 1, n_classes)。将 temp_prob 张量添加到 tensor_list 列表中。
        output_tensor = torch.cat(tensor_list, dim=1)#沿着第二个维度进行cat
        return output_tensor.float()

    def _dice_loss(self, score, target):#计算骰子损失，分别计算了交集、目标和、预测值的平方和三个量。其中，交集的计算方式为 torch.sum(score * target)，表示预测值和目标值相乘后求和；目标和和预测值的平方和的计算方式类似。
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):#计算模型预测结果和真实标签之间评价指标的
    pred[pred > 0] = 1#先将groudtruth和预测的图像进行二值化
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0: #如果预测结果和真实标签中都存在正样本（即像素值为1的像素点），则计算dice系数和hd95指标，并返回计算结果
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.cpu().detach().numpy()
    _, x, y = image.shape
    # 缩放图像符合网络输入大小224x224
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
        # label = zoom(label, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()  # 将numpy数组转换为pytorch的张量，并将其移动到GPU上；
    net.eval()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(classes)
    with torch.no_grad():  # 将神经网络设置为评估模式，不计算梯度；输入数据进行前向传播并返回预测的类别标签，该标签以NumPy数组的形式返回。
        p1 = net(input)
        label = torch.from_numpy(label).cuda()
        loss_dice = dice_loss(p1, label, softmax=True)
        loss_ce = ce_loss(p1, label.long())
        loss = 0.6 * loss_dice + 0.4 * loss_ce
        out = torch.argmax(torch.softmax(p1, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()  # 代码将out转移到CPU上，并使用detach()方法从计算图中分离出来，再使用numpy()方法将其转化为NumPy数组，最终将结果保存在out变量中。
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out
        label = label.squeeze(0).cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list, prediction, loss


