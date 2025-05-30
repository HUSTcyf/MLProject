### MLProject
-------------
### Platform
```
Ubuntu 20.04
CUDA 11.7
NVIDIA GeForce RTX 3090 * 1
Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz * 1
```
### Requirements
```
torch==1.13.0+cu117 
torchvision==0.14.0+cu117
imageio==2.9.0
lmdb==1.2.1
opencv-python==4.5.3.56
pillow
scikit-image
scipy==1.5.4
tensorboard==2.7.0
tensorboardx==2.4
tqdm==4.62.3
six==1.17.0
protobuf==3.20.3
numpy==1.24.4
pyyaml
scikit-learn
matplotlib
lpips
pytorch-fid
torch-dct
```

### Installation
We use conda environment with python 3.9:

`conda create -n dctgan python=3.9 -y` & 
`conda activate dctgan`

Follow the official website to download and install PyTorch (PyTorch with CUDA 11.x will also be fine):

`pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117`

Then install the rest requirements by:

`pip install -r requirements.txt`

### Prepare the dataset
`mkdir datasets`

### Train

### Test
