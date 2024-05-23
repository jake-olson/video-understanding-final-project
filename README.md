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


-----

## Dataset

We evaluate the TimeSformer on multiple datasets, in particular the Kinetics-600 and HowTo100M datasets. Kinetics-600 is an action recognition dataset with around 480k 10-second videos, spanning 600 action categories. HowTo100M is a dataset of narrated videos with instructions on how to perform tasks, comprising 136 million video clips from Youtube videos. We find it important to evaluate on both datasets because the type of videos differ, both in form, intent, and length. Thus, testing on both datasets gives us a better idea of how data quality impacts model performance.



### Kinetics-600 Setup

Kinetics-600 Download: [link](https://github.com/cvdfoundation/kinetics-dataset)

After the download, resize the video to the short edge size of 256, then prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`.

### HowTo100M Setup

The HowTo100M dataset is available [here](https://www.dropbox.com/sh/ttvsxwqypijjuda/AACmJx1CnddW6cVBoc21eSuva?dl=0)

------

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
