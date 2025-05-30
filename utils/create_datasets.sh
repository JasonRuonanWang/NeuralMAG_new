#!/bin/bash

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# Define common parameters
gpu=0
max_iter=50000

# Run the Python script for different combinations of width, Hext_val, and mask
for w in 32 64; do
    for h in 0 100 1000; do
        if [ "$h" -eq 0 ]; then
            mask='False'
        else
            mask='True'
        fi
        python -m utils.gen_data_s \
            --w $w \
            --Hext_val $h \
            --nseeds 100 \
            --mask $mask \
            --gpu $gpu \
            --max_iter $max_iter \
            --nstart 0 
        for split in 1 2 4 8 16 32;do
            python -m utils.gen_data_with_split_s \
                --w $w \
                --Hext_val $h \
                --nseeds 20 \
                --mask $mask \
                --split $split \
                --gpu $gpu \
                --max_iter $max_iter \
                --nstart 0 
        done
    done
done

for w in 128 256; do
    for h in 0 100 1000; do
        if [ "$h" -eq 0 ]; then
            mask='False'
        else
            mask='True'
        fi
        python -m utils.gen_data_l \
            --w $w \
            --Hext_val $h \
            --nseeds 100 \
            --mask $mask \
            --gpu $gpu \
            --max_iter $max_iter \
            --nstart 0 
        for split in 1 2 4 8 16 32;do
            python -m utils.gen_data_with_split_l \
                --w $w \
                --Hext_val $h \
                --nseeds 20 \
                --mask $mask \
                --split $split \
                --gpu $gpu \
                --max_iter $max_iter \
                --nstart 0 
        done
    done
done