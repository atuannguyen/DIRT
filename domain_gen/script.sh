#!/bin/bash 

dataset='OfficeHome'
data_dir='/mnt/vinai/vinai/data/'
#data_dir='/vinai/tuanna105/vinai/data/'
model='ours_gan'
alpha=0.0
gpu=0
echo $dataset
for target_domain in 0
  do
  echo $target_domain
  for i in 1 2 3 4 5
    do
      CUDA_VISIBLE_DEVICES=$gpu python -u main.py --alpha=$alpha --data_dir=$data_dir --dataset=$dataset --model=$model --seed=$i --epochs=100 --weight_decay=0.001 --lr=0.001 --target_domain=$target_domain > results/${model}_alpha${alpha}_${dataset}_domain${target_domain}_seed${i}.txt
    done
  done
