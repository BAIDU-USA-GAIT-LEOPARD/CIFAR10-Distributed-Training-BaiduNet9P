# CIFAR10-Distributed-Training-BaiduNet9P
DAWN CIFAR10 distributed training results by BAIDU USA GAIT LEOPARD team

CIFAR10-Distributed-Training-BaiduNet9P

Codes for DAWN training on CIFAR10 using 8xV100 on Baidu Cloud


Training
----------
Training a small network (BaiduNet9P) to reach 94.0% test accuracy on CIFAR10 data using Baidu Cloud Tesla 8*V100 GPU with 16 GB memory.

To reproduce our results: 

1. Log into Baidu Cloud, install Pytorch, setup fastai environment: https://github.com/fastai/fastai 
2. Add BaiduNet.py to fastai/fastai/vision/models
3. Add training_cifar10_BaiduNet9P.py to fastai/examples
4. Replace basic_train.py and data.py in corresponding fastai library repositories 
5. Issue the following command:

python -m fastai.launch training_cifar10_BaiduNet9P.py

The detailed traning process will be demonstrated in table format on the screen, including data preprocessing time (transformation and normalization). The performance including training time for each epoch will be saved in Perf_BaiduNet9P.tsv

In our tests, we can reach 94.0% test accuracy in about 45s (including data preprocessing time of 0.5s roughly).

