# Compressed Private Aggregation for Scalable Federated Learning over Massive Networks

![diagram_block](https://user-images.githubusercontent.com/55830582/199750743-e499fdd6-99d5-4c00-9f5a-bc1378f1cc74.png)

## Introduction
In this work we propose a method for Compressed Private Aggregation for Scalable Federated Learning over Massive Networks (CPA), which allows large-scale deployments to simultaneously communicate at extremely low bit-rates while achieving privacy, anonymity, and resilience to malicious users. Please refer to our 
[paper](https://github.com/langnatalie/CPA/files/10206979/CPA.pdf) for more details.


## Usage
This code has been tested on Python 3.7.3, PyTorch 1.8.0 and CUDA 11.1.

### Prerequisite
1. PyTorch=1.8.0: https://pytorch.org
2. scipy
3. tqdm
4. matplotlib
5. torchinfo
6. TensorboardX: https://github.com/lanpa/tensorboardX

### Training
```
python main.py --exp_name cpa --aggregation_method CPA --compression scalarQ --privacy RR --num_users 1000 --epsilon 0.5

```

### Testing
```
python main.py --exp_name cpa --eval 
```
