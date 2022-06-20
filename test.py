import os.path

import pandas as pd
import numpy as np

from pathlib import Path
import json
import cv2

import matplotlib.pyplot as plt
from infer import LitModule, get_embeddings, get_similarity, run_predict, load_ckpt
from score import score_preds
from pathlib import Path
import torch
import pytorch_lightning as pl
import timm
import sklearn
import albumentations


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--model', type=str)
    parser.add_argument('--filt', type=float, default=None)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--dev', default=True, type=str)
    args = parser.parse_args()
    gt_path = "/data/hse/data/"
    run_predict(args.save_dir, args.data_dir, args.model, args.filt, args.device)

    score = score_preds(os.path.join(args.save_dir, f"preds.tsv"), args.data_dir, ["dev"], None)
    score = pd.DataFrame(score)
    score.to_csv(os.path.join(args.save_dir, "metric.csv"))
    print(f"Score Result saved in f{os.path.join(args.save_dir, 'metric.csv')}")
    # python test.py --data_dir /data/hse/data/dev --save_dir /data/hse/test --model /data/hse/model/data25_swin_base_patch4_window7_224_ep10_img224_bnneck
