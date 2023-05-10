#!/bin/bash

python train.py \
  --encoder_num_layers 8 \
  --decoder_num_layers 8 \
  --heads 8 \
  --max_step 1000000 \
  --batch_size_trn_nag 29 \
  --batch_size_trn_pos 3 \
  --batch_size_val 29 \
  --batch_size_token 4096 \
  --save_per_step 100 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --lr 0.0001 \
  --device cuda \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir ./data/uspto50k \
  --intermediate_dir ./intermediate \
  --checkpoint_dir ./checkpoints
