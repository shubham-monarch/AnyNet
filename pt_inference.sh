#! /bin/bash

PATH_TO_DATASET="dset-finetune/"

python3 pt_inference.py --maxdisp 192  --datapath $PATH_TO_DATASET \
    --pretrained checkpoint/sceneflow/sceneflow.tar --datatype other
