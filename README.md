# MIME
MIMicking Emotion for empathetic response

## Overview of MIME
TODO

## Setup
Download GloVe vectors from [here](https://www.kaggle.com/thanakomsn/glove6b300dtxt/data) and put it into `vectors/` folder

Next Install the required libraries:
1. Assume you are using conda and have installed Pytorch >= 1.6
2. Install from mime.yml by `conda env update --file mime.yml`

## Run code
Dataset is already preprocessed and contained in this repo, [here](https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue) is the source.

### Training
```sh
python main.py
```
This will also generate 'summary.txt' file under default save path 'save/test/'

### Testing
```sh
python main.py --test --saved_model_path [your_ckpt_file_path] --save_path [output_file_path]
```

## Citation
TODO

