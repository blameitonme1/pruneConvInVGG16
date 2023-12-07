import torch
from torch import nn
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
    

