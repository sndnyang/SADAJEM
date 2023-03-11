
python train_sadajem.py --dataset cifar10 \
 --lr .1 --optimizer sgd \
 --px 1.0 --pyx 1.0 \
 --sigma .0 --width 10 --depth 28 \
 --plot_uncond --warmup_iters 1000 \
 --model wrn \
 --norm batch \
 --print_every 100 \
 --n_epochs 200 --decay_epochs 60 120 180 \
 --n_steps 5   \
 --sgld_lr 1   \
 --sgld_std 0.0 \
 --gpu-id 0
