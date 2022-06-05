import math
import cv2
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import torch
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset


from timm.data.transforms_factory import create_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import *
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    Sharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, ShiftScaleRotate, CenterCrop, Resize
)


def get_transform(image_size, mode="train"):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if mode == "train":
        return A.Compose([
                # A.LongestMaxSize(max_size=self.cfg['image_size'], interpolation=cv2.INTER_LANCZOS4, always_apply=True, p=1),
                # A.PadIfNeeded(min_height=self.cfg['image_size'], min_width=self.cfg['image_size']),
                # A.PadIfNeeded(min_height=1280, min_width=1280),
                A.RandomResizedCrop(*image_size, scale=(0.9, 1), p=1),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.ShiftScaleRotate(p=0.5),
                # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                # A.CLAHE(clip_limit=(1, 4), p=0.5),
                # A.OneOf([
                #     A.OpticalDistortion(distort_limit=1.0),
                #     A.GridDistortion(num_steps=5, distort_limit=1.),
                #     A.ElasticTransform(alpha=3),
                # ], p=0.2),
                # A.OneOf([
                #     A.GaussNoise(var_limit=[10, 50]),
                #     A.GaussianBlur(),
                #     A.MotionBlur(),
                #     A.MedianBlur(),
                # ], p=0.2),
                # # A.Resize(self.cfg['image_size'], self.cfg['image_size'], interpolation=cv2.INTER_LANCZOS4),
                # A.OneOf([
                #     JpegCompression(),
                #     Downscale(scale_min=0.1, scale_max=0.15),
                # ], p=0.2),
                # PiecewiseAffine(p=0.2),
                # Sharpen(p=0.2),
                # A.Cutout(max_h_size=int(image_size[0] * 0.1), max_w_size=int(image_size[1] * 0.1),
                #          num_holes=5, p=0.5),

                A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ], p=1.0)
    else:
        return A.Compose([
            # A.CenterCrop(img_size,img_size, p=1.),
            A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], p=1.0)


class LitDataset(Dataset):
    def __init__(self, df, transform, train=True):
        assert "image_path" in df.columns
        if train:
            self.individual_id = df.individual_id.values
        self.image_path = df.image_path.values
        self.transform = transform
        self.train = train

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # image = Image.open(self.image_path[index])
        # print(f"index {index}, {np.array(image).shape}")
        rst = {}
        image = cv2.cvtColor(cv2.imread(self.image_path[index]), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        rst["images"] = image

        if self.train:
            target = self.individual_id[index]
            target = torch.tensor(target, dtype=torch.long)
            rst["target"] = target
        return rst

    def __len__(self) -> int:
        return len(self.individual_id)


class LitDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_csv: str,
            test_csv: str,
            image_size: int,
            batch_size: int,
            num_workers: int,
            full=False,
            train_embed=None,
            test_embed=None,
            **kw,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.train_transform = get_transform(self.hparams.image_size, "train")
        self.valid_transform = get_transform(self.hparams.image_size, "valid")
        self.encoder = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Split train df using fold
        # print("----------------- use fold data set --------------------")
        if self.hparams.full:
            print("----------------- use full data set --------------------")
            train_df = pd.read_csv(self.hparams.train_csv)
            self.encoder = LabelEncoder().fit(train_df["individual_id"])
            train_df["individual_id"] = self.encoder.transform(train_df["individual_id"])
            self.train_dataset = LitDataset(train_df, transform=self.train_transform, train=True)
            self.val_dataset = Subset(self.train_dataset, list(range(100)))
        else:
            train_df = pd.read_csv(self.hparams.train_csv)
            val_df = pd.read_csv(self.hparams.test_csv)
            allow_id = set(train_df.individual_id.values)
            val_df.loc[~val_df.individual_id.isin(allow_id), "individual_id"] = "new_individual"
            self.encoder = LabelEncoder().fit(pd.concat([train_df, val_df], axis=0)["individual_id"])
            train_df["individual_id"] = self.encoder.transform(train_df["individual_id"])
            val_df["individual_id"] = self.encoder.transform(val_df["individual_id"])
            self.val_dataset = LitDataset(val_df, transform=self.valid_transform, train=True)
            self.train_dataset = LitDataset(train_df, transform=self.train_transform, train=True)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )
