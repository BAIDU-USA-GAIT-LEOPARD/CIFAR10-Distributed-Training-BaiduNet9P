from fastai.script import *
from fastai.vision import *
from fastai.distributed import *
from fastai.vision.models.BaiduNet import BaiduNet9P
torch.backends.cudnn.benchmark = True

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distrubuted training of CIFAR-10.
        python -m fastai.launch train_cifar10_BaiduNet9P.py"""

    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    path = url2path(URLs.CIFAR)
    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    workers = min(16, num_cpus()//n_gpus)
    
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=1024//n_gpus,
                                      num_workers=workers)
    data = data.normalize(cifar_stats)
 
    with open('Perf_BaiduNet9P.tsv', 'w') as tsvfile:
        tsvfile.write("epoch	hours	top1Accuracy\n")

    learn = Learner(data, BaiduNet9P(), metrics=accuracy)

    learn.to_distributed(gpu)
    learn.to_fp16().mixup()

    learn.fit_one_cycle(24, 0.0052, pct_start=0.41, wd=0.42)
