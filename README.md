# HouseDiffusion
**[HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising](https://arxiv.org/abs/2211.13287)**
<img src='figs/teaser.png' width=100%>
## Installation
**1. Clone our repo and install the requirements:**

Our implementation is based on the public implementation of [guided-diffusion](https://github.com/openai/guided-diffusion). For installation instructions, please refer to their repository. Keep in mind that our current version has not been cleaned and some features from the original repository may not function correctly.

```
git clone https://github.com/aminshabani/house_diffusion.git
cd house_diffusion
pip install -r requirements.txt
pip install -e .
```
**2. Download the dataset and create the datasets directory**

- You can download the datasets from [RPLAN's website](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) or by filling [this](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform) form.
- We also use data preprocessing from House-GAN++ which you can find in [this](https://github.com/sepidsh/Housegan-data-reader) link.
Put all of the processed files from the downloaded dataset in a `datasets` folder in the current directory:

```
house_diffusion
├── datasets
│   ├── rplan
|   |   └── 0.json
|   |   └── 1.json
|   |   └── ...
|   └── ...
└── guided_diffusion
└── scripts
└── ...
```
- We have provided a temporary model that you can download from [Google Drive](https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view?usp=share_link). 

## Running the code

**1. Training**

You can run a single experiment using the following command:
```
python image_train.py --dataset rplan --batch_size 32 --set_name train --target_set 8
```
**2. Sampling**
To sample floorplans, you can run the following command from inside of the `scripts` directory. To provide different visualizations, please see the `save_samples` function from `scripts/image_sample.py`

```
python image_sample.py --dataset rplan --batch_size 32 --set_name eval --target_set 8 --model_path ckpts/exp/model250000.pt --num_samples 64
```
You can also run the corresponding code from `scripts/script.sh`. 


## Citation

```
@article{shabani2022housediffusion,
  title={HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising},
  author={Shabani, Mohammad Amin and Hosseini, Sepidehsadat and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2211.13287},
  year={2022}
}
```
