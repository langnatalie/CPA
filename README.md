# Compressed Private Aggregation for Scalable Federated Learning over Massive Networks
![Nestedcpa](https://github.com/langnatalie/CPA/assets/55830582/0263b1d5-1156-44c5-bb03-2f4fa5a26e30)
## Introduction
In this work we propose a method for Compressed Private Aggregation for Scalable Federated Learning over Massive Networks (CPA), which allows large-scale deployments to simultaneously communicate at extremely low bit-rates while achieving privacy, anonymity, and resilience to malicious users. Please refer to our 
[paper](https://arxiv.org/abs/2308.00540) for more details.


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
