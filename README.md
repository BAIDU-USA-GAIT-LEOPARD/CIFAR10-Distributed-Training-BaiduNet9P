# CIFAR10-Distributed-Training-BaiduNet9P
DAWN CIFAR10 distributed training results by BAIDU USA GAIT LEOPARD team

CIFAR10-Distributed-Training-BaiduNet9P

Codes for DAWN training on CIFAR10 using 8xV100 on Baidu Cloud


Training
----------
Training a small network (BaiduNet9P) to reach 94.0% test accuracy on CIFAR10 data using Baidu Cloud Tesla 8*V100 GPU with 16 GB memory.

To reproduce our results, after setting up fastai environment following https://github.com/fastai/fastai and logging into Baidu Cloud, using the following script:

python -m fastai.launch training_cifar10_BaiduNet9P.py

The detailed traning process will be demonstrated in table format on the screen, including data preprocessing time. The performance including training time for each epoch will be saved as Perf_BaiduNet9P.tsv

In our tests, we can reach 94.0% test accuracy in about 45s (including data preprocessing time).

