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
    def __init__(self) -> None:
        super().__init__()
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
                X = X.cuda()
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
            v = v / np.sqrt((torch.sum(v * v)).cpu())
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
class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model) -> None:
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)
        self.model = model
        # 损失函数使用交叉熵
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()
    
    def test(self):
        # return  # 这里加了一个return，不知道是干嘛的
        self.model.eval()
        correct = 0
        total = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            # 注意是批量训练，所以第一维是batch_size
            if args.use_cuda:
                batch = batch.cuda()
            output = model(batch)
            # 找到每一个样本的最大值的索引 因为max(1)返回的是(maxvalue, indexOfMaxvalue)
            pred = output.data.max(1)[1]
            # 预测正确的个数
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        print("Accuracy: ", float(correct) / total)
        # 为什么要训练？有点莫名其妙
        # self.model.train()

    def train(self, optimizer=None, epoches=10):
        if next(self.model.parameters()).device != torch.device('cuda'):
            self.model = self.model.cuda()
        if optimizer is None:
            # 注意优化器的问题，找时间复习一下
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum = 0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finised fine tuning")
        
    def train_batch(self, optimizer, batch, label, rank_filters):
        t0 = time.time()
        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()
        # 清空梯度，防止梯度累积
        self.model.zero_grad()
        input = batch
        input.requires_grad = True
        if rank_filters:
            output = self.prunner.forward(input)
            # 计算梯度的时候，会顺带计算filter的rank
            self.criterion(output, label).backward()
        else:
            self.criterion(self.model(input), label).backward()
            optimizer.step()
        print(f" time for this training batch is {time.time() - t0} ")
    
    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            print(f" batch {i} trained")
            self.train_batch(optimizer, batch, label, rank_filters)
    
    def get_candidate_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        # 这里存疑
        return self.prunner.get_prunning_plan(num_filters_to_prune)
    
    def total_num_filters(self):
        # 这里存疑 (修改：filter个数就是out_channel的个数！！！只不过维度包含了in_channels而已)
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        
        return filters
    def prune(self):
        self.test()
        # 保证所有的layers都可训练，因为之前可能冻结了
        for param in self.model.features.parameters():
            param.requires_grad = True
        
        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        # 因为要prune掉67%的filter
        iterations = int(iterations * 2.0 / 3)

        print("Number of iterations to prune 67% filters", iterations)

        for _ in range(iterations):
            # 开始prune, finetune, prune的loop，持续iterations次
            print("Ranking filters..")
            prune_targets = self.get_candidate_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                # 统计该层prune的filter的数量
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1
            
            print("Layers tha will be pruned", layers_prunned)
            print("Prunning filters.. ")
            # 先转到cpu方便进行numpy的计算
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)
            # 更新model
            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()
            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            # 基本上，finetune就是多训练几次，重新拟合一下罢了
            print("Fine tuning to rcover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            self.train(optimizer, epoches = 1) # 我人为减少了epoch，加大了lr，到时候看看效果

        print("Finised. Going to fine tune the model a bit more.")
        self.train(optimizer, epoches = 10)
        self.test()
        torch.save(self.model.state_dict(), "model_prunned")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='use nvidia gpu')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.train:
        model = ModifiedVGG16Model()
    elif args.prune:
        # load pretrained VGG ,如果需要prune的话
        model = torch.load("model", map_location=lambda storage, loc : storage)

    if args.use_cuda:
        model = model.cuda()
    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches=10)
        torch.save(model, "model")
    
    elif args.prune:
        fine_tuner.prune()

    



    