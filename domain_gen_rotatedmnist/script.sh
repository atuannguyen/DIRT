#!/bin/bash 

model=ours_gan
gpu=0
for target_domain in 0 15 30 45 60 75
  do
  echo $target_domain
  for i in 1 2 3 4 5
    do
      CUDA_VISIBLE_DEVICES=$gpu python -u train.py --model=$model --seed=$i --epochs=500 --target_domain=$target_domain > results/${model}_domain${target_domain}_seed${i}.txt
    done
  done
