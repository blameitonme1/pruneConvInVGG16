import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# 定义prune的函数，之前还不知道pytorch可以直接调用知名模型并且还可以选择是否pretrained
def prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=False):
    # 获取想要prune的那一层conv，不关心名字
    _, conv = list(model.features._modules.items())[layer_index]
    next_conv = None
    offset = 1
    # 查找下一个卷积层
    while layer_index + offset < len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1
    # 基本就是复制了一遍conv的参数，但是因为prune，所以要减去out_channels
    new_conv = nn.Conv2d(in_channels=conv.in_channels,
                      out_channels=conv.out_channels - 1, # 因为要prune掉一个filter，对应的featuremap也就没有了呢
                      kernel_size=conv.kernel_size,
                      stride=conv.stride,
                      padding=conv.padding,
                      # 这两个参数之后学习，现在还没有讲到那里去
                      dilation=conv.dilation,
                      groups=conv.group,
                      bias=(conv.bias is not None))
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    # 复制数据，因为减少了一个filter，自然第一维out_channels要少1
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        # 新的权重转到GPU进行训练，因为cpu是方便numpy计算才转移的
        new_conv.weight.data = new_conv.weight.data.cuda()
    # 开始处理bias
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(bias_numpy.shape[0] - 1, dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias)
    if use_cuda:
        new_conv.bias.data = new_conv.bias.data.cuda()
    
    if not next_conv is None:
        # 自从layer_index的下一个卷积层, 因为上一层修改了out_channels,所以这一层也要修改in_channels,这也就是为什么要find next
        # conv2d layer
        next_new_conv = torch.nn.Conv2d(
            in_channels=next_conv.in_channels,
            out_channels=next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride = next_conv.stride,
            padding=next_conv.padding,
            dilation=next_conv.dilation,
            groups=next_conv.groups,
            bias=(next_conv.bias is not None)
        )
