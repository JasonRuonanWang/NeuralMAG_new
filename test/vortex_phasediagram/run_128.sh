#!/usr/bin/sh

#SBATCH -N 1 ### 使用的节点数目
#SBATCH -n 10 ### 一个节点内部的20个核
#SBATCH --gres=gpu:1 ### 使用 2 张 gpu 卡
#SBATCH --nodelist=gpu40903 ### 使用 gpu001 节点
#SBATCH --partition=gpu4090_128 ### PARTITION 名称，可通过 sinfo 查询
#SBATCH --job-name=sz128 ### 提交的任务名称
#SBATCH --output=./log/log_128.txt ### 输出文件
#SBATCH --error=./log/err_128.txt ### 错误日志文件
#SBATCH -A hmt03
# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

for size in 32 64 128 256;
do
    for nstart in 0 100 200 300 400 500 600 700
    do
    python ./compute_vortex_fft.py --w $size \
                                   --layers 2 \
                                   --krn 48 \
                                   --split $(($size/2)) \
                                   --error_min 1.0e-5 \
                                   --dtime 5.0e-13 \
                                   --max_iter 150000 \
                                   --nsamples 100 

    python ./compute_vortex_unet.py --w $size \
                                    --layers 2 \
                                    --krn 48 \
                                    --split $(($size/2)) \
                                    --error_min 1.0e-5 \
                                    --dtime 5.0e-13 \
                                    --max_iter 150000 \
                                    --nsamples 100
    done
    python ./analyze_vortex.py --w $size \
                            --split $(($size/2)) \
                            --method fft \
                            --errorfilter 1e-4
    python ./analyze_vortex.py --w $size \
                                --split $(($size/2)) \
                                --method unet \
                                --errorfilter 1e-4 \
    python ./plot_phase_diagram.py --method fft
    python ./plot_phase_diagram.py --method unet
done

