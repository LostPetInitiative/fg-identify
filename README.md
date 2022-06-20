# Fine Grained Method for individual Identify

This repository contains code to reproduce the best results for this paper: zhirui, Animal Recognition Using Method of Fine-Grained Visual Analysis.

The best model achieved score one dev-hard-lost:

| model      | recall@1 | recall@10 | hit1_n90%@top0.1 | hit10_n90%@top0.1 |
| ----------- |----------|-----------|------------------|-------------------|
| head_swin_bnneck      | 0.3229   | 0.6021    | 0.0            | 0.0282|
| head_swin_bnneck w/ trick | 0.3177   | 0.6016  | 0.3474        | 0.2018  |


# Reproduction of the best results
## Install

Need python version 3.8 or later, linux OS, PyTorch==1.11.0 with cuda, install libararies list in requirements.txt.

```shell
git clone git@github.com:LostPetInitiative/study_spring_2022.git
cd study_spring_2022
cd zhirui
pip install -r requirements.txt
```

## Download

### From zenodo


```shell
# in zhirui folder
bash download.sh
```

### From Kaggle

Go to [kaggle dataset page](https://www.kaggle.com/datasets/evilpsycho42/finegrained?select=dev), download all 4 file one by one.

1. unzip head_swin_bnneck.zip to ./download
2. unzip data_25.zip to ./download
3. unzip dev.zip to ./download
4. move yolov5s.pt to ./download

After that, the download folder shuold like this:

```text
download/
├── data_25
├────── rf100199
├────── ...
├── dev
├────── found
├────── lost
├────── registry.csv
├── head_swin_bnneck
├────── cfg.yaml
├────── last.ckpt
├────── model.ckpt
├── placeholder
└── yolov5s.pt
```

## Data Prepare

```shell
# in zhirui folder
# crop data_25 and dev images
# It will take some time to do it
python crop_image.py
# extract train/val data from data_25/body/head for model training
# It will take some time to do it
python train_data_split.py --data_dir ./download/data_25
python train_data_split.py --data_dir ./download/data_25_body
python train_data_split.py --data_dir ./download/data_25_head
```

## Reproduc best result

Best model (head swin bnneck) trained on head crop dataset, so be care input the head crop dataset.

```shell
# in zhirui folder
python test.py --data_dir ./download/dev_head --save_dir ./download/reproduc --model ./download/head_swin_bnneck
```

Use threshold filter fake image for better hit metric.

```shell
# in zhirui folder
python test.py --data_dir ./download/dev_head --save_dir ./download/reproduc_trick --model ./download/head_swin_bnneck --filt 0.74
```

# Train Model

Modify config file in `config` folder first.

```shell
# in zhirui folder
python train.py --config ./config/your_config_file.yml --device 0
```
