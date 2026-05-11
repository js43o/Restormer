#!/usr/bin/env bash
export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638
export PYTHONPATH=$(pwd)

torchrun --standalone --nproc_per_node=1 --master_port=7679 basicsr/test.py -opt options/test_blind_celeba_6m.yml --launcher pytorch
