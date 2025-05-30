#!/bin/bash

#SBATCH -N 1 ### 使用的节点数目
#SBATCH -n 10 ### 一个节点内部的20个核
#SBATCH --gres=gpu:1 ### 使用 2 张 gpu 卡
#SBATCH --nodelist=gpu005 ### 使用 gpu001 节点
#SBATCH --partition=amdgpu40g ### PARTITION 名称，可通过 sinfo 查询
#SBATCH --job-name=de5 ### 提交的任务名称
#SBATCH --output=./log/log5.txt ### 输出文件
#SBATCH --error=./log/err5.txt ### 错误日志文件
#SBATCH -A hmt03
ulimit -s unlimited
ulimit -v unlimited
ulimit -c unlimited

ex=1.0
python ./train.py --batch-size 100 \
                --lr 0.005 \
                --epochs 1000 \
                --kc 48 \
                --inch 6 \
                --ntrain 600 \
                --ntest 20 \
                --gpu 0 \
                --ex $ex 
mv ./ex${ex}_bsz100_Ir0.005_Unet_kc48_inch6 ./ex${ex}_de
cp ./dev_train_5.py ./ex${ex}_de/dev_train.py
cp ./Unet_5.py ./ex${ex}_de/Unet.py
