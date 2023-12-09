import torch
from torch import nn
import numpy as np
import time
from torch.nn import functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# 新层换旧层的操作,因为要形成sequential，所以还是要返回没有被替换的层
def replace_layers(model, i, indexs, layers):
    if i in indexs:
        return layers[indexs.index(i)]
    return model[i]         
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
                      groups=conv.groups,
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

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()
        # 注意这里是第二个维度了，因为对于new_conv来说in_channels减少了
        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda
        next_new_conv.bias.data = next_conv.bias.data
    if not next_conv is None:
        # 删除两个layer并生成新的model.feature
        features = torch.nn.Sequential(
            *(replace_layers(
                model.features, i , [layer_index, layer_index + offset], 
                [new_conv, next_new_conv]
            ) for i, _ in enumerate(model.features))
        )
        del model.features
        del conv
        model.features = features
    
    else:
        # prune的是最后一层卷积层，会影响接下来的全连接层
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], [new_conv])
            for i, _ in enumerate(model.features))
        )

        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, nn.Linear):
                old_linear_layer = module
                break
        layer_index += 1
        if old_linear_layer is None:
            # 类似之前学的panic的语法
            raise BaseException("No linear layer has been found in classifier")
        # 计算每一个channel的参数数量
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels
        new_linear_layer = nn.Linear(
            # 因为会减少一个filter，等于减少一个channel
            old_linear_layer.in_features - params_per_input_channel, 
            old_linear_layer.out_features
        )
        # 操作同上
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        new_weights[:, : params_per_input_channel * filter_index] = old_weights[:, : params_per_input_channel * filter_index]
        new_weights[:, params_per_input_channel * filter_index :] = old_weights[:, params_per_input_channel * (filter_index + 1) : ]
        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda
        # bias不需要修改，因为bias维度是out_channels，直接复制,为什么要使用data，我猜测可能和deep copy有关
        new_linear_layer.bias.data = old_linear_layer.bias.data
        classifier = nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index],
                             [new_linear_layer]) for i, _ in enumerate(model.classifier))
                )
        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier
    return model

if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    model.train()
    t0 = time.time()
    model = prune_vgg16_conv_layer(model, 28, 10)
    print("The prunning took", time.time() - t0)