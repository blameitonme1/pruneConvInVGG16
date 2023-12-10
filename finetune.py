import torch
from torchvision import models
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
import time
from heapq import nsmallest
from operator import itemgetter
class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = models.vgg16(pretrained=True)
        self.features = model.features
        # 冻结这些参数，使他们不会更新
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            # 基本上和VGG16最后的全连接层一样，就是修改了最后一层的参数
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )
    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        return X
    
class FilterPrunner:
    def __init__(self, model) -> None:
        self.model = model
        self.reset()
    def reset(self):
        # 初始化一个对于filter排名的dict
        self.filter_ranks = {}
    def forward(self, X):
        # 其实activations储存的是特征图啊，这个名字有点莫名其妙
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            X = module(X)
            if isinstance(module, nn.modules.conv.Conv2d):
                # 计算梯度的时候顺带执行用户自定义的函数（这里是compute_rank）
                X.register_hook(self.compute_rank)
                self.activations.append(X)
                # layer是原来的模型里面的下标
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
        
        return self.model.classifier(X)
    def compute_rank(self, grad):
        # 梯度从而往前计算，所以activation_index也是从后往前
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        # 不是内积，是逐元素做乘法，所以维度不会改变（这里有点疑问）
        taylor = activation * grad
        # 求平均值, 求每一个filter的平均taylor expassion值，注意taylor的维度和特征图是一样的
        taylor = taylor.mean(dim = (0, 2, 3)).data
        # 初始化
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
            torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
        
        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

def lowest_ranking_filters(self, num):
    data = []
    for i in sorted(self.filter_ranks.keys()):
        for j in range(self.filter_ranks[i].size(0)):
            # 每一个filter_rank[i]就是i层的所有filter的importance
            data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
    # itemgetter表示按照第三个元素进行排序，在整个模型范围内寻找
    return nsmallest(num, data, itemgetter(2))

def normalize_ranks_per_layer(self):
    # 使用L2范数
    for i in self.filter_ranks:
        # ranks可以理解为importance，对于loss函数的贡献
        v = torch.abs(self.filter_ranks[i])
        v = v / np.sqrt(torch.sum(v * v))
        self.filter_ranks[i] = v.cpu()

def get_prunning_plan(self, num_filters_to_prune):
    # 取得需要prune的最低的那几个filters的列表
    filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
    filters_to_prune_per_layer = {} # 储存每一层需要prune的filters
    for (l, f, _) in filters_to_prune:
        if l not in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = []
        filters_to_prune_per_layer[l].append(f)
    # 这里处理
    for l in filters_to_prune_per_layer:
        filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
        for i in range(len(filters_to_prune_per_layer[l])):
            filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i
    
    filters_to_prune = []
    for l in filters_to_prune_per_layer:
        for i in filters_to_prune_per_layer[l]:
            filters_to_prune.append((l, i))
    
    return filters_to_prune