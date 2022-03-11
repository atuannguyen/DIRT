source /homes/55/tuan/torch/bin/activate

target_domain=0
seed=5
dataset='VLCS'
hidden_dim=64
gpu=3

mkdir /scratch/local/ssd/tuan/data/
mkdir /scratch/local/ssd/tuan/gan_path/

tar -xf /datasets/tuan/VLCS.tar -C /scratch/local/ssd/tuan/data/
tar -xf /storage/tuan/dg/stargan_cpkt/${dataset}_domain${target_domain}_last-G.ckpt.tar -C /scratch/local/ssd/tuan/gan_path/

data_dir='/scratch/local/ssd/tuan/data/'
gan_path='/scratch/local/ssd/tuan/gan_path/'

CUDA_VISIBLE_DEVICES=$gpu python main.py --data_dir=${data_dir} --gan_path=${gan_path} --target_domain=${target_domain} --dataset=$dataset --seed=$seed --hidden_dim=${hidden_dim}

#rm -rf /scratch/local/ssd/tuan/*
