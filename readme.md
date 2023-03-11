# SADA-JEM

Official code for the paper [Towards Bridging the Performance Gaps of Joint Energy-based Models](https://arxiv.org/abs/2209.07959)



## Environment Initialization

Default version: Python 3.6

1. pip install -r requirements.txt           # For IS/FID, it's better to use conda environment.
2. python multi_center.py --dataset cifar10  # generate Gaussian Mixtures' mus and sigmas

The trained SADA-JEM models can be found in https://1drv.ms/u/s!AgCFFlwzHuH8nS7Ewaupps3hhqtl?e=GD4cuM

## Training

To train a SADA-JEM model on CIFAR10 as in the paper, please refer to scripts/sadajem_cifar10.sh

```
python train_sadajem.py --dataset cifar10 \
     --lr .1 --optimizer sgd \
     --px 1.0 --pyx 1.0 \
     --sigma .0 --width 10 --depth 28 \
     --plot_uncond --warmup_iters 1000 \
     --model wrn \
     --norm batch \
     --print_every 100 \
     --n_epochs 200 --decay_epochs 60 120 180 \
     --n_steps 5      \
     --sgld_lr 1   \
     --sgld_std 0.0  \
     --gpu-id 0
```


## Evaluation

To evaluate the model (on CIFAR10), please refer to scripts/eval_sadajem.sh, all_auroc.sh, all_ood.sh,  bpda_attack.sh


### test accuracy

```
python eval_sadajem.py --eval test_clf --load_path $1
```

### generate from scratch

```
python eval_sadajem.py --eval uncond_samples \
            --buffer_size 100 \
            --batch_size 100  \
            --n_sample_steps 100 \
            --n_steps 10 \
            --print_every 1  \
            --gpu-id  0  \
            --load_path ~/sadajem10_948_withbuffer.pt
```


### evaluate IS/FID in the replay buffer

Note: sometimes, the evaluation of FID may fail and you can rerun the evaluation.
```
python eval_sadajem.py --eval fid \
            --ratio 0.9 \
            --gpu-id  3  \
            --load_path ~/sadajem10_948_withbuffer.pt
```

### ECE calibration

```
python eval_sadajem.py --eval cali \
            --gpu-id  3  \
            --load_path ~/sadajem10_948_withbuffer.pt
```

### robustness using BPDA_EOT bpda_attack

1. check the bpda_eot_attack_jem.json,  "adv_norm": "l_inf",   and other configurations
2. run `CUDA_VISIBLE_DEVICES=4 python bpda_eot_attack.py checkpoint_pt_file_path 8`,  8 is the attack strength.


## Citation

If you found this work useful and used it in your research, please consider citing this paper.
```
@article{yang2023sadajem,
    title={Towards Bridging the Performance Gaps of Joint Energy-based Models},
    author={Xiulong Yang, Qing Su and Shihao Ji},
    journal={IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
```