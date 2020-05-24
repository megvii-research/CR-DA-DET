#!/bin/bash
save_dir="/data/experiments/SW_Faster_ICR_CCR/clipart/model"
dataset="clipart"
pretrained_path="/data/pretrained_model/resnet101_caffe.pth"
net="res101"

python da_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex