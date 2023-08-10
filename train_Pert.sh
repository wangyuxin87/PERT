#! bash

CKPT_NAME='pert_0'
GPU_ID=0

python train_Pert.py --save_path logs/${CKPT_NAME} --gpu_id ${GPU_ID} --save_freq 20 --data_path /path_to_data


