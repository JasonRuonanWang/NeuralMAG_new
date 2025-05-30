#!/bin/bash

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

for size in 32 64 128 256;do
    for experiment in "square" "random_material" "random_shape"; do
        python ./compute_seed.py --experiment $experiment \
                                    --w $size \
                                    --layers 2 \
                                    --split 4 \
                                    --error_min 1.0e-5 \
                                    --dtime 5.0e-13 \
                                    --max_iter 150000 \
                                    --nsamples 400 \
                                    --deg 0.1
        python ./evaluate_seed.py --experiment $experiment \
                                    --w $size \
                                    --layers 2 \
                                    --split 4 \
                                    --deg 0.1
        for model in "ori" "new"; do
            python ./compute_vortex.py --experiment $experiment \
                                    --model $model \
                                    --w $size \
                                    --deg 0.1 \
                                    --layers 2 \
                                    --split 4 \
                                    --error_min 1.0e-5 \
                                    --dtime 5.0e-13 \
                                    --max_iter 150000 \
                                    --nsamples 100
                                    
            python ./analyze_vortex.py --experiment $experiment \
                                    --model $model \
                                    --w $size \
                                    --deg 0.1 \
                                    --split 4 \
                                    --errorfilter 1.0e-4 
        done
    done
done

        