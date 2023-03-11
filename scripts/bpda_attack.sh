
for i in {1,2,4,8,12,16,22,30}
do
  echo $i
  CUDA_VISIBLE_DEVICES=4 python bpda_eot_attack.py $1 $i
done