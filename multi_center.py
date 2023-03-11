import os
import argparse
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from one_center import *

def category_mean(train_loader, arg):
    import time
    start = time.time()
    im_test, target_test = [], []
    for im, targ, idx in train_loader:
        im_test.append(im.detach())
        target_test.append(targ.detach())
    im_test, target_test = torch.cat(im_test), torch.cat(target_test)
    print(im_test.shape)

    im_sz = ds_size(arg)
    size = [3, im_sz, im_sz]
    centers = torch.zeros([args.n_classes, int(np.prod(size))])
    covs = torch.zeros([args.n_classes, int(np.prod(size)), int(np.prod(size))])

    imc = im_test
    im_test = downsample(imc, im_sz).view(len(imc), -1)
    for i in range(args.n_classes):
        imc = im_test[target_test == i]
        imc = imc.view(len(imc), -1)
        mean = imc.mean(dim=0)
        sub = imc - mean.unsqueeze(dim=0)
        cov = sub.t() @ sub / len(imc)
        centers[i] = mean
        covs[i] = cov
    print(time.time() - start)
    print(centers.shape, covs.shape)

    v = torch.__version__.split('.')[1]
    Path('v%s' % v).mkdir(parents=True, exist_ok=True)
    torch.save(centers, 'v%s/%s_mean.pt' % (v, arg.dataset))
    torch.save(covs, 'v%s/%s_cov.pt' % (v, arg.dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "svhn", "cifar100", 'tinyimagenet', 'img32', 'img128', 'img256', 'celeba128'])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--gpu-id", type=str, default="0")
    args = parser.parse_args()
    args.seed = 1
    args.batch_size = 100
    args.debug = False
    args.n_valid = 0
    if 'celeba' in args.dataset:
        data_loader = get_train(args)
    else:
        data_loader, _, test_loader = get_train_test(args)

    if 'img' in args.dataset:
        data_loader = test_loader
    category_mean(data_loader, args)
