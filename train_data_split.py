######################
### Process Data25 ###
### Run it before train####################
### and pass the data path to config file.#
###########################################
import pandas as pd
import cv2
from pathlib import Path

# For original data
# individual_id = []
# image_path = []
# image_size = []
#
# for path in Path("/data/hse/data/data_25/").glob("*/*g"):
#     iid = str(path.parent.name)
#     img = cv2.imread(str(path))
#     if img is not None:
#         individual_id.append(iid)
#         image_path.append(str(path))
#         image_size.append(img.shape[:2])
#
# df = pd.DataFrame({"individual_id": individual_id, "image_path": image_path, "image_size": image_size})
# df = df.sample(frac=1.).reset_index(drop=True)
# train = df.iloc[:-1000]
# val = df.iloc[-1000:].reset_index(drop=True)
# train.to_csv("/data/hse/data/train.csv", index=False)
# val.to_csv("/data/hse/data/val.csv", index=False)

# For body crop data
individual_id = []
image_path = []
image_size = []

for path in Path("/data/hse/data_crop_body/data_25/").glob("*/*g"):
    iid = str(path.parent.name)
    img = cv2.imread(str(path))
    if img is not None:
        individual_id.append(iid)
        image_path.append(str(path))
        image_size.append(img.shape[:2])

df = pd.DataFrame({"individual_id": individual_id, "image_path": image_path, "image_size": image_size})
df = df.sample(frac=1.).reset_index(drop=True)
train = df.iloc[:-1000]
val = df.iloc[-1000:].reset_index(drop=True)
train.to_csv("/data/hse/data_crop_body/train.csv", index=False)
val.to_csv("/data/hse/data_crop_body/val.csv", index=False)

# For head crop data
individual_id = []
image_path = []
image_size = []

for path in Path("/data/hse/data_crop_head/data_25/").glob("*/*g"):
    iid = str(path.parent.name)
    img = cv2.imread(str(path))
    if img is not None:
        individual_id.append(iid)
        image_path.append(str(path))
        image_size.append(img.shape[:2])

df = pd.DataFrame({"individual_id": individual_id, "image_path": image_path, "image_size": image_size})
df = df.sample(frac=1.).reset_index(drop=True)
train = df.iloc[:-1000]
val = df.iloc[-1000:].reset_index(drop=True)
train.to_csv("/data/hse/data_crop_head/train.csv", index=False)
val.to_csv("/data/hse/data_crop_head/val.csv", index=False)
