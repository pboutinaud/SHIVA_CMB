#!/bin/bash
# Example script to run segmentation prediction for CMB (Cerebral Microbleeds)
# using the new K3 SavedModel models (v2) with SWI modality.
#
# Adjust MODEL_DIR, IMAGE_DIR and output path to your setup.
# --batch_size: number of slices per batch (increase for GPU, default 1 for CPU)
# --gpu: GPU index to use (-1 for CPU)
#
# @author : Philippe Boutinaud - Fealinx

MODEL_DIR=./SWI-CMB
IMAGE_DIR=./images

python ./predict_one_file.py \
    --verbose --gpu 0 --batch_size 1 \
    -m $MODEL_DIR/20250129-192041_ResUnet3D-8.9.2-1.5-SWAN.CMB_prod2_fold_0_bestvalloss.tf_inference \
    -m $MODEL_DIR/20250129-195344_ResUnet3D-8.9.2-1.5-SWAN.CMB_prod2_fold_1_bestvalloss.tf_inference \
    -m $MODEL_DIR/20250129-192041_ResUnet3D-8.9.2-1.5-SWAN.CMB_prod2_fold_2_bestvalloss.tf_inference \
    -i $IMAGE_DIR/test_SWI_resampled_111_cropped_intensity_normed.nii.gz \
    -o ./predicted/test_cmb.nii.gz
