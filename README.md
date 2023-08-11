# PERT
The offical implementation of PERT: [What is the Real Need for Scene Text Removal? Exploring the Background Integrity and Erasure Exhaustivity Properties](https://ieeexplore.ieee.org/document/10214243) (TIP2023). PERT progressively erases the text region with a balanced multi-stage erasure and Region-base Modification Strategy. The model is simple to be implemented (light weight) and can be easily developed.

# News
10/8 The model is released \
10/8 The code is released.

# Performance

## The Detailed Performance on SCUT-EnsText
 
|        Dataset     	|        Model       	| PSNR 	| MSSIM 	| MSE 	| AGE |  pEPs |  pCEPs |
|:------------------: |:------------------:	|:---------:	|:------:	|:---------:	|:---------:	|:---------:	|:---------:	|      
|    SCUT-EnsText     |  Paper 	|    33.62   	|     97.00  	|    0.0013   	|    2.1850   	|    0.0135   	|    0.0088   	|    
|    SCUT-EnsText   	|      This implementation   	|    34.12   	|     97.06   	|    0.0012   	|    2.1299   	|    0.0125   	|    0.0080   	| 

## The BI and EE property on SCUT-EnsText

|        Dataset     	|        Model       	| BI 	| EE 	|
|:------------------: |:------------------:	|:---------:	|:------:	|
|    SCUT-EnsText     |  Paper 	|    63.55   	|    80.00   	| 
|    SCUT-EnsText   	|      This implementation   	|    64.73   	|    79.46   	| 


# Preparetion

## Requirement

python=3.6.9 & torch==1.8.1+cu111 \
It is the best to use our provided Dockerfile for quick start.

## Dataset

Download [SCUT-EnsText](https://github.com/lcy0604/EraseNet) and put it into your folder.

## Trained model download

Download our trained model from [Google](https://drive.google.com/file/d/1uU8lGUIp62W5HkwyjzY3Mc15O0-_jkKP/view?usp=drive_link) and put it into your folder.

# Evaluation

```bash
bash test_Pert.sh
```
## Evaluation BI-Metric
```bash
python evaluatuion_bi.py --target_path /path_to_output --gt_path /path_to_label --BI True
```
## Evaluation EE-Metric
```bash
python evaluatuion_bi.py --target_path /path_to_output --gt_path /path_to_label --BI False
```

# Train
```bash
bash train_Pert.sh
```

# Citation
If you find our method useful for your reserach, please cite
```bash
@ARTICLE{10214243,
  author={Wang, Yuxin and Xie, Hongtao and Wang, Zixiao and Qu, Yadong and Zhang, Yongdong},
  journal={IEEE Transactions on Image Processing}, 
  title={What is the Real Need for Scene Text Removal? Exploring the Background Integrity and Erasure Exhaustivity Properties}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2023.3290517}}
```
