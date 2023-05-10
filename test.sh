#!/bin/bash

python test.py \
  --batch_size_val 32 \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir data/uspto50k \
  --intermediate_dir intermediate \
  --checkpoint_dir checkpoints \
  --checkpoint model_825000_ori.pt \
  --beam_size 10 \
  --search_strategy False
