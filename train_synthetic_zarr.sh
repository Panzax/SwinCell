#!/bin/bash
# Training script for synthetic Zarr cube data

python ./swincell/train_main.py \
    --data_dir=/data/ \
    --metadata_root=/clusterfs/nvme/segment_3d/databases/synthetic_data/training_metadata.csv \
    --use_zarr_metadata \
    --use_synthetic \
    --dataset=zebrafish_memhistone \
    --model=swin \
    --logdir=./results/synthetic_zarr \
    --max_epochs=3000 \
    --batch_size=1 \
    --sw_batch_size=2 \
    --optim_lr=1e-4 \
    --optim_name=adamw \
    --val_every=100 \
    --roi_x=96 \
    --roi_y=96 \
    --roi_z=32 \
    --feature_size=48 \
    --in_channels=1 \
    --out_channels=4 \
    --a_min=0 \
    --a_max=255 \
    --b_min=0.0 \
    --b_max=1.0 \
    --downsample_factor=1 \
    --use_flows \
    --save_checkpoint \
    --workers=8 \
    --max_rois=1000 \
    --wandb_project=swincell_zebrafish \
    --wandb_run_name=synthetic_zarr_experiment \
    --wandb_mode=online

