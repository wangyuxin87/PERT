#! bash

CKPT_NAME='pert_0'
GPU_ID=1

python test_Pert.py --logdir logs/${CKPT_NAME} --data_path /path_to_data --save_path logs/${CKPT_NAME} --gpu_id ${GPU_ID}

python evaluatuion_detail.py --target_path logs/${CKPT_NAME}/output --gt_path /path_to_label
python evaluatuion_bi.py --target_path logs/${CKPT_NAME}/output --gt_path /path_to_label --BI False
python evaluatuion_bi.py --target_path logs/${CKPT_NAME}/output --gt_path /path_to_label --BI True


