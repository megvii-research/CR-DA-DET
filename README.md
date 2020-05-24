## Exploring Categorical Regularization for Domain Adaptive Object Detection
**Chang-Dong Xu**, **Xing-Ran Zhao**, Xin Jin, Xiu-Shen Wei*


This repository is the official PyTorch implementation of paper [Exploring Categorical Regularization for Domain Adaptive Object Detection](). (The work has been accepted by [CVPR2020](http://cvpr2020.thecvf.com/))

## Main requirements

  * **torch == 1.0.0**
  * **torchvision == 0.2.0**
  * **Python 3**

## Environmental settings
This repository is developed using python **3.6.7** on Ubuntu **16.04.5 LTS**. The CUDA nad CUDNN version is **9.0** and **7.4.1** respectively. We use **one NVIDIA 1080ti GPU card** for training and testing. Other platforms or GPU cards are not fully tested.

## Pretrain models
**The pretrain backbone (vgg, resnet) and pretrain DA DET model (ICR-CCR) will be released soon.**

## Usage
The usage of SW-ICR-CCR is same to DA-ICR-CCR. Take DA-ICR-CCR as an example:
```bash
# install
cd DA_Faster_ICR_CCR
python setup.py build develop
# to train DA-Faster-ICR-CCR on cityscape:
sh train_cityscape.sh
# To validate DA-Faster-ICR-CCR on cityscape:
python test_cityscape.py
```

## Data and Format
**The data will be released soon.**

## Citing this repository
If you find this code useful in your research, please consider citing us:
```
@article{CR-DA-DET,
	title={Exploring Categorical Regularization for Domain Adaptive Object Detection},
	author={Chang-Dong Xu and Xing-Ran Zhao and Xin Jin and Xiu-Shen Wei},
	booktitle={CVPR},
	pages={1--8},
	year={2020}
}
```

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: weixs.gm@gmail.com

Chang-Dong Xu: xuchangdong@megvii.com

Xing-Ran Zhao: zhaoxingran@megvii.com

Xin Jin: jinxin@megvii.com
