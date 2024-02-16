# MIBR

This repository is the implementation for Paper "Exploring Multi-Granularity Interests for Long-Term User
Behavior Modeling in CTR Prediction".




## Requirements

* Install Python, Pytorch(>=1.8). We use Python 3.8, Pytorch 1.13.0.
* If you plan to use GPU computation, install CUDA.



## Dataset

We use two public real-world datasets (Taobao, Alipay) in our experiments. We pre-process the data in the same way with ETA and SDIM. You can download the datasets from the links below.

- **Taobao**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=649. 
- **Alipay**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=53. 




## Example

If you have downloaded the source codes, you can train MIBR model.

```
$ cd main
$ python train.py --dataset=[dataset] --strategy=[strategy]  --gpu==[gpu_id] 
```

The options for command parameter "strategy" are ['1,1',  '1+1',  'n,1',  'n+1'].

You can change the model parameters in `./config/configs.py`



## Contact

If you have any questions for our paper or codes, please send an email to demon@mail.ustc.edu.cn.



## Acknowledgment 

Our code is developed based on [GitHub - shenweichen/DeepCTR-Torch: 【PyTorch】Easy-to-use,Modular and Extendible package of deep-learning based CTR models.](
