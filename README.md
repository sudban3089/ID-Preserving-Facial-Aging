# ID-Preserving-Facial-Aging
Identity-Preserving Aging of Face Images via Latent Diffusion Models [IJCB 2023]

## Usage
Create the `ldm` environment by following the steps outlined in [Dreambooth Stable Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion). Check `environment.yaml` file.

## Training

### Model weights
We fine-tune a pre-trained stable diffusion model whose weights can be downloaded from [Hugging Face](https://huggingface.co/CompVis) model card. We use `sd-v1-4-full-ema.ckpt`. You can use any other model depending on your choice but we have not tested the reproducibility of the conference results with other models.

### Data preparation
We need a **Regularization Set** that comprises images depicting distinct individuals (disjoint from the training set) depicting variations in age. We curated a set of 612 images from the [CelebA-Dialog dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html) that serves as image-caption pairs in this work. The age captions are as follows.
- child
- teenager
- youngadults
- middleaged
- elderly
- old
  
Download the Regularization Set used in our work or you can create your own regularization set but we cannot verify the performance with a custom regularization set. 

We need a **Training Set** that comprises images of a specific individual on whom the facial aging will be applied. The training set enables the diffusion model to learn the identity-specific charctristics during training that are then transferred at the time of generation of images with aging/de-aging.

### Fine-tuning using default Dreambooth

```
python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word person
```
### Fine-tuning using custom losses
- Biometric loss
  ```
  python main.py --base configs/stable-diffusion/v1-finetune_biomloss.yaml 
                -t 
                --actual_resume /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word person
  ```
- Contrastive loss
  ```
  python main.py --base configs/stable-diffusion/v1-finetune_contrastiveloss.yaml 
                -t 
                --actual_resume /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word person
  ```

Note:
1. `personalized_captionandimage.py` modifies the original script to include image-caption pairs. The age caption comes fromt he filename of the regularization set images.
2. Change rare token in **line 11** of `personalized_captionandimage.py` and `personalized.py` to a rare identifier of your choice. It is currently hardcoded to *sks*.

## Acknowledgment
This repository is heavily dependent with code borrowed from [Dreambooth Stable Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) repository. All changes made in the scripts and config files need to be incorporated to reproduce the results from the conference paper

## Citation
If you find this code useful or utilize it in your work, please cite:
```
@INPROCEEDINGS {IDFaceAging_IJCB2023,
author = {Sudipta Banerjee* and Govind Mittal* and Ameya Joshi and Chinmay Hegde and Nasir Memon},
booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
title = {Identity-Preserving Aging of Face Images via Latent Diffusion Models},
year = {2023},
}
```
