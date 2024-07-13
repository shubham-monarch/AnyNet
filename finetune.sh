#! /bin/bash

PATH_TO_DATASET="dataset-finetune/"

python3 finetune.py --maxdisp 192  --datapath $PATH_TO_DATASET \
    --pretrained checkpoint/scenflow/sceneflow.tar --datatype other