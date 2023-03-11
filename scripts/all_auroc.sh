

echo svhn
python eval_sadajem.py  --norm batch --model wrn --load_path $1 --sigma 0 --eval logp_hist --datasets cifar10 svhn
echo cifar100
python eval_sadajem.py  --norm batch --model wrn --load_path $1 --sigma 0 --eval logp_hist --datasets cifar10 cifar100
echo celeba
python eval_sadajem.py  --norm batch --model wrn --load_path $1 --sigma 0 --eval logp_hist --datasets cifar10 celeba

