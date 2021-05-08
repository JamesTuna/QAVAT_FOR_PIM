# QAVAT_FOR_PIM
### Quantization-aware and Variation-aware Training (QAVAT)
- This repository provides Pytorch implementation of the QAVAT framework.  
- QAVAT allows user to train robust CNNs against multiplicative Gaussian weight variations.
- Flexibity to sepcify any Activation/Weight (A/W) bitwidth, although more bits and weights the better the performance
- Quick start: Bash scripts to run and test the algorithm are provided. take a look of them under directory ```Exprm_LeNet``` or ```Exprm_VGG```.
![QAVAT](https://github.com/JamesTuna/QAVAT_FOR_PIM/blob/main/CodeIrrelevant/QAVAT.png)  
### Required packages and recommended versions: 
Note: GPU is required to run the codes, please make sure at least one cuda devices is detected.
- ```Pytorch 3.8.5``` ```torchvision 1.71``` ```tensorboardX 2.1``` ```numpy 1.19```
# Quick overview: what to expect
## Fully binarized LeNet-5 is able to do well on MNIST
![](https://github.com/JamesTuna/QAVAT_FOR_PIM/blob/main/CodeIrrelevant/A1W1-LeNet.png)
## VGG model with 5-bit activation and 1-bit weight on CIFAR-10: not so bad
![](https://github.com/JamesTuna/QAVAT_FOR_PIM/blob/main/CodeIrrelevant/A5W1-VGG11.png)
## VGG model with 8-bit activation and 4-bit weight on CIFAR-10: more budget better performance
![](https://github.com/JamesTuna/QAVAT_FOR_PIM/blob/main/CodeIrrelevant/A8W4-VGG11.png)

