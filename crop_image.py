import sys
import cv2

import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def crop_image(model, image, verbose=False):
    try:
        img = load_image(image)
    except:
        return None, None

    if img is None:
        return None, None
    pred = model(img)
    if len(pred.xyxy) > 0:

        xyxy = pred.xyxy[0].cpu().numpy()

        body = xyxy[xyxy[:, 5] == 1, :4].astype(int)

        if len(body) > 0:
            body = [body[:, 0].min(), body[:, 1].min(), body[:, 2].max(), body[:, 3].max()]
            body_crop = img[body[1]: body[3], body[0]:body[2], :]
        else:
            body_crop = img

        head = xyxy[xyxy[:, 5] == 0, :4].astype(int)

        if len(head) > 0:
            head = [head[:, 0].min(), head[:, 1].min(), head[:, 2].max(), head[:, 3].max()]
            head_crop = img[head[1]: head[3], head[0]:head[2], :]
        else:
            head_crop = img

        if verbose:
            f, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img)
            axes[1].imshow(body_crop)
            axes[2].imshow(head_crop)
        else:
            return body_crop, head_crop
    else:
        if verbose:
            f, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img)
            axes[1].imshow(img)
            axes[2].imshow(img)
        else:
            return img, img


if __name__ == "__main__":
    sys.path.append("./yolov5/")
    model = torch.hub.load('./yolov5/', 'custom', path='download/yolov5s.pt', source='local')

    source_path = Path("./download/data_25")
    body_path = Path("./download/data_25_body")
    head_path = Path("./download/data_25_head")

    for img in tqdm(source_path.glob("*/*.*g")):
        body, head = crop_image(model, str(img), False)

        if body is None:
            continue

        body_save_folder = body_path / img.parent.name
        body_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(body_save_folder / img.name, body)

        head_save_folder = head_path / img.parent.name
        head_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(head_save_folder / img.name, head)

    source_path = Path("./download/dev/found/found/")
    body_path = Path("./download/dev_body/found/found")
    head_path = Path("./download/dev_head/found/found")

    for img in tqdm(source_path.glob("*/*.*g")):
        body, head = crop_image(model, str(img), False)

        if body is None:
            continue

        body_save_folder = body_path / img.parent.name
        body_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(body_save_folder / img.name, body)

        head_save_folder = head_path / img.parent.name
        head_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(head_save_folder / img.name, head)

    for source_file in tqdm(source_path.glob("*/*.json")):
        target_file = body_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

        target_file = head_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

    source_path = Path("./download/dev/found/synthetic_lost")
    body_path = Path("./download/dev_body/found/synthetic_lost")
    head_path = Path("./download/dev_head/found/synthetic_lost")

    for img in tqdm(source_path.glob("*/*.*g")):
        body, head = crop_image(model, str(img), False)

        if body is None:
            continue

        body_save_folder = body_path / img.parent.name
        body_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(body_save_folder / img.name, body)

        head_save_folder = head_path / img.parent.name
        head_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(head_save_folder / img.name, head)

    for source_file in tqdm(source_path.glob("*/*.json")):
        target_file = body_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

        target_file = head_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

    source_path = Path("./download/dev/lost/lost")
    body_path = Path("./download/dev_body/lost/lost")
    head_path = Path("./download/dev_head/lost/lost")

    for img in tqdm(source_path.glob("*/*.*g")):
        body, head = crop_image(model, str(img), False)

        if body is None:
            continue

        body_save_folder = body_path / img.parent.name
        body_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(body_save_folder / img.name, body)

        head_save_folder = head_path / img.parent.name
        head_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(head_save_folder / img.name, head)

    for source_file in tqdm(source_path.glob("*/*.json")):
        target_file = body_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

        target_file = head_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

    source_path = Path("./download/dev/lost/synthetic_found")
    body_path = Path("./download/dev_body/lost/synthetic_found")
    head_path = Path("./download/dev_head/lost/synthetic_found")

    for img in tqdm(source_path.glob("*/*.*g")):
        body, head = crop_image(model, str(img), False)

        if body is None:
            continue

        body_save_folder = body_path / img.parent.name
        body_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(body_save_folder / img.name, body)

        head_save_folder = head_path / img.parent.name
        head_save_folder.mkdir(exist_ok=True, parents=True)
        save_image(head_save_folder / img.name, head)

    for source_file in tqdm(source_path.glob("*/*.json")):
        target_file = body_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)

        target_file = head_path / source_file.parent.name / source_file.name
        shutil.copy(source_file, target_file)
