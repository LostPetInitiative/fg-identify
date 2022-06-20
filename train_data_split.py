######################
### Process Data25 ###
### Run it before train####################
### and pass the data path to config file.#
###########################################
import pandas as pd
import cv2
from pathlib import Path


def train_data_split(data_dir, num_test=1000):
    individual_id = []
    image_path = []
    image_size = []

    data_dir = Path(data_dir)

    for path in data_dir.glob("*/*g"):
        iid = str(path.parent.name)
        img = cv2.imread(str(path))
        if img is not None:
            individual_id.append(iid)
            image_path.append(str(path))
            image_size.append(img.shape[:2])

    df = pd.DataFrame({"individual_id": individual_id, "image_path": image_path, "image_size": image_size})
    df = df.sample(frac=1.).reset_index(drop=True)
    train = df.iloc[:-num_test]
    val = df.iloc[-num_test:].reset_index(drop=True)
    train.to_csv(data_dir / "train.csv", index=False)
    val.to_csv(data_dir / "val.csv", index=False)
    print(f"train/val dataframe saved in {data_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_test', type=int, default=1000)
    args = parser.parse_args()
    train_data_split(args.data_dir, args.num_test)
