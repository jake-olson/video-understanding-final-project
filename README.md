# Comparing the Effects of Attention Scheme and Data Quality on Scalability


With recent advances in video understanding models, new technologies have been released at an unprecedented rate. Previously state of the art models, such as the TimeSformer, now lack the newest concepts and technologies. We seek to identify if updating previous SOTA models like the TimeSformer with better data and more sophisticated attention mechanisms increases its performance, or if it is simply an outdated model. Namely, we investigate the effect of different attention schemes and different datasets on model performance. By varying both attention type and dataset used with one base model, we isolate the effects of both factors, thereby proving which one is more effective in increasing model performance with large inputs and high scalability. 

## Installation

To install and develop locally, follow these instructions, sourced from the TimeSformer paper:

- PyTorch version >= 1.8.0
- Python version >= 3.7


First, set up a conda virtual environment

```bash
conda create -n timesformer python=3.7 -y
conda activate timesformer
```

Then install the following packages

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`


Lastly, build the TimeSformer codebase by running:
```
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
python setup.py build develop
```

------

## Running the Default model

-----

## Dataset

We evaluate our AV-MAE on the commonly used video dataset - UCF101, which consists of over 13k (9.5k/3.5k train/val) video clips across 101 action classes, and is grouped into five types: Human-Object Interaction, Body-Motion Only, Human-Human Interaction, Playing Musical Instruments, and Sports. UCF101 contains web videos captured in uncontrolled settings, typically featuring camera movements, diverse lighting conditions, occasional partial occlusions, and occasional frames of low quality. This makes it an ideal resource for developing an advanced and robust encoder capable of getting high-quality video representations.

Download UCF101: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

### Preprocessing

Please follow the instructions in [DATASET.md](videomae/DATASET.md) for data preparation.

## Experiments on UCF101

### Model Training

#### Video Reconstruction Training

```bash
bash scripts/ucf101/videomae_vit_base_patch16_224_tubemasking_ratio_0.75_epoch_3200/pretrain.sh
```

#### Video Classification Training from scratch

```bash
bash scripts/ucf101/videomae_vit_base_patch16_224_tubemasking_ratio_0.75_epoch_3200/finetune.sh
```

#### Video Classification Training using pre-trained model on the K400 dataset

```bash
bash scripts/ucf101/videomae_vit_base_patch16_224_tubemasking_ratio_0.75_epoch_3200/finetune_withpretrained.sh
```

### Loss Logs

File names inside the 'Loss Logs' folder:

- `cls_pretrain`
- `cls_scratch`
- `logs_modified_attention`
- `logs_without_attention_original`
- `original`

### Sample Videos Generated

Folder names that contain our video reconstruction samples:

- `VideoMAE_Original`
- `AV-MAE_withoutAtt`
- `AV-MAE`

## Citation

If you find the repository useful for your work, please cite our paper.

```
@inproceedings{matusz2024scalability,
  title={Comparing the Effects of Attention Scheme and Data Quality on Scalability},
  author={Matusz, DJ, and Olson, Jake},
  year={2024},
  organization={Dartmouth College Computer Science}
}
```
