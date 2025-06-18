### MLProject (DCTGAN)
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
`mkdir datasets` & `cd datasets`

Download the Mstar dataset from the link [MStar](https://pan.baidu.com/s/183OZYvfwezPtAvHx9TItyA?pwd=3ds4) code: 3ds4. Then unzip the file to `./datasets/Mstar`. 

Run `python prepare_mstar.py -i ./datasets/Mstar -o ./datasets/tmp` to extract the images to `./datasets/Mstar.npy`

### Train
Run `bash scripts/train.sh` or 
```sh
python train.py \
--conf configs/mstar_dctgan.yaml \
--output_dir results/mstar_dctgan \
--gpu 0
```
The results will be saved at `./datasets/mstar_dctgan`.

### Test & Metrics
Run `bash scripts/test.sh` or 
```sh
python test.py \
--dir Mstar \
--conf configs.yaml \
--name results/mstar_dctgan \
--gpu 0
```

### Generate SAR detection dataset
Run `python gen_detect.py -o results/`, and the result will be saved at `./results/multi`.
