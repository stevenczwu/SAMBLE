# SAMBLE: Shape-Specific Point Cloud Sampling for an Optimal Trade-Off Between Local Detail and Global Uniformity

<p>
<a href="https://arxiv.org/abs/2504.19581">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://ies.iar.kit.edu/1473_1524.php">
    <img src="https://img.shields.io/badge/Author-Homepage-red" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
</p>

## 🔥 News

**[Feb 2025]** SAMBLE is accepted by CVPR 2025! <br>
**[Mar 2023]** APES is selected as a Highlight by CVPR 2023! <br>
**[Feb 2023]** [APES](https://github.com/JunweiZheng93/APES) (our former work) is accepted by CVPR 2023!


## 🔧 Prerequisites

Create an environment and install dependencies:
```bash
conda create -n apesv2 python=3.9 -y
conda activate apesv2
pip install -r requirements.txt
```
Install PyTorch and Pytorch3D:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
conda install -c fvcore -c iopath -c conda-forge fvcore=0.1.5 iopath=0.1.9 -y
conda install -c pytorch3d pytorch3d=0.7.0 -y
```


## 📦 How to run

### Classification

```bash
# train:
python train_modelnet.py datasets=modelnet usr_config=YOUR/USR/CONFIG/PATH train.ddp.which_gpu=[0,1] train.epochs=2
# test:
python test_modelnet.py datasets=modelnet usr_config=YOUR/USR/CONFIG/PATH train.ddp.which_gpu=[0,1] train.epochs=2
```

### Segmentation
 
```bash
# train:
python train_shapenet.py datasets=shapenet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH test.ddp.which_gpu=[0,1]
# test:
python test_shapenet.py datasets=shapenet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH test.ddp.which_gpu=[0,1]
```

## 📖 Citation

If you are interested in this work, please cite as below:

```text
@inproceedings{wu_2025_samble,
author={Wu, Chengzhi and Wan, Yuxin and Fu, Hao and Pfrommer, Julius and Zhong, Zeyun and Zheng, Junwei and Zhang, Jiaming and Beyerer, J\"urgen},
title={SAMBLE: Shape-Specific Point Cloud Sampling for an Optimal Trade-Off Between Local Detail and Global Uniformity},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2025}
}
```
```text
@inproceedings{wu_2023_attention,
author={Wu, Chengzhi and Zheng, Junwei and Pfrommer, Julius and Beyerer, J\"urgen},
title={Attention-Based Point Cloud Edge Sampling},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}
}
```
