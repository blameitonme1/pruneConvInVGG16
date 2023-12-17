# Using Pytorch to prune a VGG16 cat-dog classifier

My thoughts about the pytorch implementation of this article [[1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference]](https://arxiv.org/abs/1611.06440)

The origin implementation is in [[jacobgil/pytorch-pruning]](https://github.com/jacobgil/pytorch-pruning)

# **Pruning and finetuning**

## 1 Training Speed

- After the first iteration of pruning, the training speed for each epoch (or batch of cource) will **decrease dramatically!** When using pretrained model from pytorch to train on my toy dataset, the average training time for one batch is **0.3s** . However, when I first pruned  13%  its filters, the training time for one batch during finetuning is **1.17s!** That's almost x4 slower!
  
- As pruning goes on, the problem seems to mitigate due to fewer filters (**smaller model will train faster than big one ofc**). Now the avgSpeed per batch is 0.24s~0.25s
  
- So it's safe to conclude prune will increase performance when talking about training or prediction speed due to smaller model size.(**So less calculation needed**).
  
- Training speed after 4 iterations of pruning:
  
  ![[speed after 4 iterations of prunning.png]](https://raw.githubusercontent.com/blameitonme1/pics/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-12-17%20174933.png)
## 2 Accuracy
- As for accuracy, the original pretrained model's accuracy is roughly 0.98
- After every pruning iteration, the accuracy will drop a little bit (some time a lot more), but with finetuning, the accuracy will go up.
- Finally, the accuracy stays about 0.97. Though it's dropped a little bit, it works just fine on a small dataset such as the **cat-dog calssifier** since it's much smaller than original VGG16.
## 3 Model Size

- original model's size is 512M
- model's size after being pruned 67% : 148M
- A much smaller model with similar accuracy on this specific task!

# Conclusion

As shown above, after pruning a pretrained model  (VGG16 trained on ImageNet), and train it on a toy dataset (cat-dog classifier, much smaller dataset), we get a much smaller model with faster traing speed and good enough accuracy.