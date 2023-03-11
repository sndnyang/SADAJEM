

# test accuracy
python eval_sadajem.py --eval test_clf --load_path $1


# generate from scratch

python eval_sadajem.py --eval uncond_samples \
            --buffer_size 100 \
            --batch_size 100  \
            --n_sample_steps 100 \
            --n_steps 10 \
            --print_every 1  \
            --gpu-id  0  \
            --load_path  $1


# evaluate IS/FID in the replay buffer

python eval_sadajem.py --eval fid \
            --ratio 0.9 \
            --gpu-id  3  \
            --load_path $1


# evaluate ECE(calibration)

python eval_sadajem.py --eval cali \
            --gpu-id  3  \
            --load_path $1

