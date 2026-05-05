#!/usr/bin/env bash

torchrun --standalone --nproc_per_node=2 --master_port=7679 basicsr/train.py -opt options/train_blind_ffhq_modified.yml --launcher pytorch
