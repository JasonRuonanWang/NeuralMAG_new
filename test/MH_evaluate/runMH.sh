#!/bin/bash


# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH

gpu=0
# 1
for width in 64 128 256 512
do
    for mask in False True triangle hole
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax 0.5e-6 --Ku 0.0 --dtime 2.0e-13 --mask $mask
    done
done

# 2
for width in 64 128 256 512
do
    for Ms in 1200 800
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms $Ms --Ax 0.5e-6 --Ku 0.0 --dtime 2.0e-13
    done
done

# 3
for width in 64 128 256 512
do
    for Ax in 0.7e-6 0.3e-6
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax $Ax --Ku 0.0 --dtime 2.0e-13
    done
done

# 4
for width in 64 128 256 512
do
    for Ku in 1e5 2e5 3e5 4e5
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax 0.5e-6 --Ku $Ku --Kvec 1,0,0 --dtime 2.0e-13
    done
done

python ./plot.py
python ./plot_point.py

