# CIFAR10-Distributed-Training-BaiduNet9P
DAWN CIFAR10 distributed training results by BAIDU USA GAIT LEOPARD team

CIFAR10-Distributed-Training-BaiduNet9P

codes for DAWN training on CIFAR10 using eight v100 on Baidu Cloud


Training
----------
Training a small network(BaiduNet9P) to reach 94.0% test accuracy on CIFAR10 data using Baidu Cloud Tesla 8*V100 GPU with 16 GB memory.

Please setup fastai environment in https://github.com/fastai/fastai

To reproduce our results, after logging into Baidu Cloud, just type:

python training_cifar10.py

The detailed traning process will be demonstrated in table format on the screen.

In our tests, we can reach 94.0% test accuracy in about 45s.

