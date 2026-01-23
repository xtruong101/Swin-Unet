#!/bin/bash
# Train Swin-Unet on ACDC dataset

python train_swin_acdc.py \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path ../data/ACDC \
    --output_dir ../model/swin_acdc_224 \
    --dataset ACDC \
    --num_classes 4 \
    --max_epochs 150 \
    --batch_size 24 \
    --base_lr 0.05 \
    --img_size 224 \
    --seed 1234 \
    --num_workers 4
