#!/bin/bash
save_dir="/data/experiments/DA_Faster_ICR_CCR/clipart/model"
dataset="clipart"
net="res101"

python da_train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --max_epochs 12
