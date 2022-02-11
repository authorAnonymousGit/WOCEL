#!/bin/bash

CONFIGURATIONS_NUM=$1;

export NVIDIA_VISIBLE_DEVICES=$2
export CUDA_VISIBLE_DEVICES=$2

for (( iter=1; iter<=CONFIGURATIONS_NUM; iter++ ))
do
  echo $iter;

  python -W ignore main.py \
            --iter_num $iter \
            --task_type "train_primary" \
            --cuda_id $2
done
