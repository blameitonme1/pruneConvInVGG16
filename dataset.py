import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# 里面部分数据增强技术还没有学，就当作抽象黑盒看待了
# 加载训练数据集
def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path, 
            transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), # 将图像转化为tensor
                    normalize,])),
        batch_size=batch_size,
        shuffle=True, # 默认会随机打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory
    )
def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path, 
            transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224), # 测试数据集就不需要随机裁剪了，直接中心裁剪
                    # 简而言之不需要那些数据增强的步骤了，因为是拿来测试的
                    transforms.ToTensor(),
                    normalize,])),
        batch_size=batch_size,
        shuffle=False, # 默认会随机打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory
    )
