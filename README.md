# MIME
This repository contains PyTorch implementations of the models from the paper Utterance-level Dialogue Understanding: [An Empirical Study MIME: MIMicking Emotions for Empathetic Response Generation](./MIME.pdf).

## Overview of MIME
![Alt text](figs/MIME.png?raw=true "Architecture of MIME")

## Setup
Download GloVe vectors from [here](https://www.kaggle.com/thanakomsn/glove6b300dtxt/data) and put it into `vectors/` folder

Next Install the required libraries:
1. Assume you are using conda and have installed Pytorch >= 1.6
2. Install from mime.yml by `conda env update --file mime.yml`

For reproducibility purposes, we provide weights & output result on test dataset in [google drive](https://drive.google.com/drive/folders/1Qab9mH6n6qPrVTP4vtQ0-oGa6GYrD8Lm?usp=sharing). 
you can download the model and move it to `save/saved_model`

 > note: `model1` is the model we used and reported in paper, you can also use `model2` which is a better version by retraining the same model. The difference between `model1` and `model2` is the random seed

## Run code
Dataset is already preprocessed and contained in this repo, [here](https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue) is the source.

### Training
```sh
python main.py
```
This will also generate output file on test under default save path `save/test/summary.txt`

### Testing
```sh
python main.py --test --save_path [output_file_path]
```
By default it will generate `save/test/summary.txt` as output

## Citation
`An Empirical Study MIME: MIMicking Emotions for Empathetic Response Generation. Navonil Majumder, Pengfei Hong, Shanshan Peng, Jiankun Lu, Deepanway Ghosal, Alexander Gelbukh, Rada Mihalcea, Soujanya Poria. EMNLP (2020).`

