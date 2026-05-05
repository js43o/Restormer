#!/usr/bin/env bash

torchrun --standalone --nproc_per_node=1 --master_port=7679 basicsr/test.py -opt options/test_blind_88m_lfw.yml --launcher pytorch
