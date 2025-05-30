#!/usr/bin/sh
#SBATCH -N 1 ### 使用的节点数目
#SBATCH -n 7 ### 一个节点内部的20个核
#SBATCH --gres=gpu:1 ### 使用 2 张 gpu 卡
#SBATCH --nodelist=gpu40904 ### 使用 gpu001 节点
#SBATCH --partition=gpu4090_128 ### PARTITION 名称，可通过 sinfo 查询
#SBATCH --job-name=stp4 ### 提交的任务名称
#SBATCH --output=./logstp4.txt ### 输出文件
#SBATCH --error=./errstp4.txt ### 错误日志文件
#SBATCH -A hmt03
ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited
# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

for size in 32 64 96 128 256 512
do
    python ./prb4_compare.py --w $size
done
python ./prb4_compare.py --w 512 --converge True 

# plot figure
python ./plot_data.py
