#!/bin/bash

#SBATCH -N 1 ### 使用的节点数目
#SBATCH -n 10 ### 一个节点内部的20个核
#SBATCH --gres=gpu:1 ### 使用 2 张 gpu 卡
#SBATCH --nodelist=gpu005 ### 使用 gpu001 节点
#SBATCH --partition=amdgpu40g ### PARTITION 名称，可通过 sinfo 查询
#SBATCH --job-name=tt ### 提交的任务名称
#SBATCH --output=./log/sp_1024_log.txt ### 输出文件
#SBATCH --error=./log/sp_1024_err.txt ### 错误日志文件
#SBATCH -A hmt03
ulimit -s unlimited
ulimit -v unlimited
ulimit -c unlimited
# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

for width in 32 64 128 256 512 1024 2048;do

    # Define the log file path

    LOG_FILE="speed_test_w$width.log"

    # Clear the log file if it exists
    > "$LOG_FILE"


    # Run the Python scripts and append output to the log file
    {
        python unet_speed.py --gpu 0 --w $width --layers 2 --trt False
        # python unet_speed.py --gpu 0 --w $width --layers 2 --trt True
        python mm_speed.py   --gpu 0 --w $width --layers 2
    } &>> "$LOG_FILE"

    # Print the contents of the LOG_FILE
    cat "$LOG_FILE"

    # Extract and print the specific lines from the log file, then append them to the end
    UNet_line=$(grep "Unt_size:" "$LOG_FILE")
    # UNet_linetrt=$(grep "Unt_trt:" "$LOG_FILE")
    MAG_line=$(grep "MAG_size:" "$LOG_FILE")
    echo -e "\n\n +---------------------------Results Summary-----------------------------+"
    echo -e "$UNet_line"
    # echo -e "$UNet_linetrt"
    echo -e "$MAG_line"

    # Append the captured lines to the log file
    {
        echo -e "\n\n +---------------------------Results Summary-----------------------------+"
        echo -e "$UNet_line"
        # echo -e "$UNet_linetrt"
        echo -e "$MAG_line"
    } >> "$LOG_FILE"
done