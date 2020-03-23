#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1

python ../codes/train_with_bert.py \
  --times 100 \
  --epoch 2 \
  --batch_size 1 \
  --maxlen 512 \
  --do_train  \
  --do_predict \
  --nclass 3 \
