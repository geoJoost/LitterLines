#!/bin/bash

# Number of runs for each configuration
N_RUNS=10

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 1, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS none \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING True \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 2, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS low \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING True \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 3, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS normal \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING True \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 4, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS high \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING True \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 5, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS vhigh \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING True \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 6, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS none \
     --START_FROM_CHECKPOINT False \
     --BAND_MAPPING False \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 48 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 7, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS high \
     --START_FROM_CHECKPOINT False \
     --BAND_MAPPING False \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 48 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 8, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS none \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING False \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 9, iteration =$i"
    python train.py --model_num $i \
     --AUGMENTATIONS high \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING False \
     --FIRST_CONV_WARMUP_EPOCHS 0 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done

for (( i=1; i<$N_RUNS+1; i++ ))
do
    echo "Running training with config 10, iteration $i"
    python train.py --model_num $i \
     --AUGMENTATIONS high \
     --START_FROM_CHECKPOINT True \
     --BAND_MAPPING False \
     --FIRST_CONV_WARMUP_EPOCHS 5 \
     --BATCH_SIZE 64 \
     --early_stopping_patience 20
done