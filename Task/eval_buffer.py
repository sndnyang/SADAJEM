import os
import time
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
import numpy as np
from torch.utils.data import DataLoader


def norm_ip(img, min, max):
    temp = t.clamp(img, min=min, max=max)
    temp = (temp + -min) / (max - min + 1e-5)
    return temp


def cond_is_fid(f, new_buffer, args, device, ratio=0.1, eval='all'):
    n_it = new_buffer.size(0) // 100
    all_y = []
    probs = []
    with t.no_grad():
        for i in range(n_it):
            x = new_buffer[i * 100: (i + 1) * 100].to(device)
            logits = f.classify(x)
            y = logits.max(1)[1]
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)

    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    each_class = [new_buffer[all_y == l] for l in range(args.n_classes)]
    each_class_probs = [probs[all_y == l] for l in range(args.n_classes)]
    print([len(c) for c in each_class])

    new_buffer = []
    for c in range(args.n_classes):
        each_probs = each_class_probs[c]
        # print("%d" % len(each_probs))
        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        topks = t.topk(each_probs, topk)
        index_list = topks[1]
        images = each_class[c][index_list]
        new_buffer.append(images)

    new_buffer = t.cat(new_buffer, 0)
    print(new_buffer.shape)
    from Task.eval_buffer import eval_is_fid
    inc_score, std, fid = eval_is_fid(new_buffer, args, eval=eval)
    if eval in ['is', 'all']:
        print("Inception score of {} with std of {}".format(inc_score, std))
    if eval in ['fid', 'all']:
        print("FID of score {}".format(fid))
    return inc_score, std, fid


def eval_is_fid(replay_buffer, args, eval='all'):
    from Task.inception import get_inception_score
    from Task.fid import get_fid_score
    if isinstance(replay_buffer, list):
        images = replay_buffer[0]
    elif isinstance(replay_buffer, tuple):
        images = replay_buffer[0]
    else:
        images = replay_buffer

    feed_imgs = []
    for i, img in enumerate(images):
        n_img = norm_ip(img, -1, 1)
        new_img = n_img.cpu().numpy().transpose(1, 2, 0) * 255
        feed_imgs.append(new_img)

    feed_imgs = np.stack(feed_imgs)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_px = tr.Compose(
        [
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ]
    )

    if 'cifar100' in args.dataset:
        from Task.data import Cifar100
        test_dataset = Cifar100(args, augment=False)
    elif 'cifar' in args.dataset:
        from Task.data import Cifar10
        test_dataset = Cifar10(args, full=True, noise=False)
    elif 'svhn' in args.dataset:
        from Task.data import Svhn
        test_dataset = Svhn(args, augment=False)
    elif args.dataset in ["imagenet", "img128", 'img256']:
        from Task.data import Imagenet
        test_dataset = Imagenet(train=False)
    elif args.dataset == 'stl10':
        test_dataset = tv.datasets.STL10(root='./data', transform=transform_px, download=True, split="train")
    elif args.dataset in ['celeba128', 'img32']:
        from utils import dataset_with_indices
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        # no test set for celeba128, I save all images in args.data_root/train/subdir
        set_name = 'train' if args.dataset in ['celeba128'] else 'val'
        test_dataset = cls(root=os.path.join(args.data_root, set_name), transform=transform_px)
    else:
        assert False, 'dataset %s' % args.dataset

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=False)

    test_ims = []

    def rescale_im(im):
        return np.clip(im * 256, 0, 255).astype(np.uint8)

    for d in test_dataloader:

        if args.dataset == 'stl10':
            data = d[0].numpy().transpose(0, 2, 3, 1)
        else:
            data_corrupt, data, label_gt = d
            if args.dataset in ['celeba128', 'img32']:
                data = data_corrupt.numpy().transpose(0, 2, 3, 1)
            else:
                data = data.numpy()
        test_ims.extend(list(rescale_im(data)))
        if (args.dataset == "imagenet" or 'img' in args.dataset) and len(test_ims) > 60000:
            test_ims = test_ims[:60000]
            break

    # FID score
    # n = min(len(images), len(test_ims))
    fid = -1
    print(feed_imgs.shape, len(test_ims), test_ims[0].shape)
    if eval in ['fid', 'all']:
        try:
            start = time.time()
            fid = get_fid_score(feed_imgs, test_ims)
            print("FID of score {} takes {}s".format(fid, time.time() - start))
        except:
            print("FID failed")
            fid = -1
    score, std = 0, 0
    if eval in ['is', 'all']:
        splits = max(1, len(feed_imgs) // 5000)
        start = time.time()
        score, std = get_inception_score(feed_imgs, splits=splits)
        print("Inception score of {} with std of {} takes {}s".format(score, std, time.time() - start))
    return score, std, fid
