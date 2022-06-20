import os.path
from pathlib import Path, WindowsPath, PurePath
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
import yaml
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from lightgbm import LGBMClassifier
import joblib
import pathlib
import cv2
from torch.utils.data import DataLoader, Dataset
import json
from model import LitModule
from dataset import get_transform


def load_ckpt(ckpt_path, device="cuda:0"):
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    with open(ckpt_path / "cfg.yml") as f:
        cfg = EasyDict(yaml.safe_load(f))
    if os.path.exists(ckpt_path / "last.ckpt"):
        model = eval(cfg['model_type']).load_from_checkpoint(ckpt_path / "last.ckpt").eval()
        print(f"load model from {ckpt_path / 'last.ckpt'}")
    elif os.path.exists(ckpt_path / "model.ckpt"):
        model = eval(cfg['model_type']).load_from_checkpoint(ckpt_path / "model.ckpt").eval()
        print(f"load model from {ckpt_path / 'model.ckpt'}")
    else:
        raise ValueError
    return model.to(device), cfg


def safe_load(file_or_obj):
    if isinstance(file_or_obj, (WindowsPath, str, PurePath)):
        file_or_obj = str(file_or_obj)
        if file_or_obj.endswith("npy"):
            return np.load(file_or_obj, allow_pickle=True)
        elif file_or_obj.endswith("csv"):
            return pd.read_csv(file_or_obj)
        else:
            raise TypeError
    elif isinstance(file_or_obj, (pd.DataFrame, np.ndarray)):
        return file_or_obj
    else:
        raise TypeError


class TestDataset(Dataset):

    def __init__(self, data, image_size):

        self.image_path = data['image_path']
        self.transform = get_transform(image_size, "valid")

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        rst = {}
        image = cv2.cvtColor(cv2.imread(self.image_path[item]), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        rst["images"] = image
        return rst


@torch.inference_mode()
def get_embeddings(ckpt, data_file, device="cuda:0", save=None):
    model, config = load_ckpt(ckpt, device)
    model.eval()
    df = safe_load(data_file)
    embeddings = []
    ds = TestDataset(df, config.image_size)
    dl = DataLoader(ds, batch_size=config.batch_size, drop_last=False, shuffle=False,
                    num_workers=config.num_workers)
    bar = tqdm(enumerate(dl), total=len(dl))
    for step, data in bar:
        batch = {k: torch.tensor(v).to(device) for k, v in data.items()}
        embedding = model(**batch).cpu().numpy()
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)
    # embeddings = normalize(embeddings, axis=1, norm="l2")
    print(f"embeddings size {embeddings.shape}")
    if save is not None:
        np.save(save, embeddings)
    return embeddings


def _get_similarity(query_embed, query_df, answer_embed,
                   answer_df, scale=False, K=200, reduce_func="max", filt=None):
    query_embed = safe_load(query_embed)
    query_df = safe_load(query_df)
    answer_embed = safe_load(answer_embed)
    answer_df = safe_load(answer_df)

    neigh = NearestNeighbors(n_neighbors=K, metric="cosine").fit(answer_embed)

    batch = 211 if len(query_embed) % 217 == 0 else 217
    nn_idxs = []
    nn_distances = []
    for step in tqdm(range(len(query_embed) // batch + 1)):
        distances, idxs = neigh.kneighbors(query_embed[step * batch: (step + 1) * batch], return_distance=True)
        nn_idxs.append(idxs)
        nn_distances.append(distances)
    nn_idxs = np.concatenate(nn_idxs, axis=0)
    nn_distances = np.concatenate(nn_distances, axis=0)

    similarity_df = []
    for i in tqdm(range(len(nn_idxs))):
        answer = answer_df.name[nn_idxs[i]]
        answer_image_path = answer_df.image_path[nn_idxs[i]]
        distances = nn_distances[i]
        subset_preds = pd.DataFrame(np.stack([answer, answer_image_path, distances], axis=1),
                                    columns=['answer', 'answer_image_path', 'similarity'])
        subset_preds['query'] = query_df.name[i]
        subset_preds['query_image_path'] = query_df.image_path[i]
        similarity_df.append(subset_preds)
    similarity_df = pd.concat(similarity_df).reset_index(drop=True)
    similarity_df['similarity'] = (1 - similarity_df['similarity'].values).astype(float).round(3)
    if filt is not None:
        similarity_df.loc[similarity_df.similarity >= filt, "similarity"] = 0.0
    if reduce_func == "max":
        simi_reduce = similarity_df.groupby(['query_image_path', 'answer']).similarity.max().reset_index()
    elif reduce_func == "mean":
        simi_reduce = similarity_df.groupby(['query_image_path', 'answer']).similarity.mean().reset_index()
    else:
        raise ValueError
    simi_reduce['query'] = simi_reduce['query_image_path'].map(similarity_df.set_index("query_image_path")['query'].to_dict())
    simi_reduce = simi_reduce.sort_values('similarity', ascending=False).reset_index(drop=True)
    del similarity_df
    if scale:
        mx = simi_reduce['similarity'].max()
        mn = simi_reduce['similarity'].min()
        simi_reduce.loc[:, "similarity"] = (simi_reduce.similarity - mn) / (mx - mn)
    return simi_reduce


def get_similarity(query_df, query_embed, answer_df, answer_embed,
                   scale=False, K=200, reduce_func="max", save=None, filt=None):
    query_embed = safe_load(query_embed)
    query_df = safe_load(query_df)
    answer_embed = safe_load(answer_embed)
    answer_df = safe_load(answer_df)

    simi = []
    animals = query_df.animal_type.unique()

    for animal in animals:
        query_idx= query_df[query_df.animal_type == animal].index.tolist()
        answer_idx = answer_df[answer_df.animal_type == animal].index.tolist()
        simi.append(_get_similarity(query_embed[query_idx], query_df.iloc[query_idx].reset_index(drop=True),
                                    answer_embed[answer_idx], answer_df.iloc[answer_idx].reset_index(drop=True),
                                    scale=scale, K=K, reduce_func=reduce_func, filt=filt))
    simi = pd.concat(simi, axis=0).reset_index(drop=True)
    print(f"similarity size {simi.shape}")
    if save is not None:
        simi.to_csv(save, index=False)
    return simi


def get_prediction(similarity_df, method='simple', N=100):
    similarity_df = safe_load(similarity_df)

    if method == "simple":
        predictions = {}
        for i, row in similarity_df.iterrows():
            if row['query'] in predictions:
                if len(predictions[row['query']]["answer"]) == N:
                    continue
                predictions[row['query']]['answer'].append(row['answer'])
                predictions[row['query']]['similarity'].append(row['similarity'])
            else:
                predictions[row['query']] = {"answer": [row['answer']], "similarity": [row.similarity]}

    matched_1 = {}
    matched_3 = {}
    matched_10 = {}

    for q in predictions.keys():
        s = np.array(predictions[q]['similarity'])

        matched_1[q] = s[0]
        matched_3[q] = np.mean(s[:3]) + s[0]
        matched_10[q] = np.mean(s[:10]) + s[0]

        predictions[q]['answer'] = ",".join(predictions[q]['answer'])
        predictions[q]['similarity'] = ",".join([str(p) for p in predictions[q]['similarity']])
        # del predictions[q]['similarity']

    predictions = pd.DataFrame(predictions).T.reset_index().rename({"index": "query"}, axis=1)
    predictions['matched_1'] = predictions['query'].map(matched_1)
    predictions['matched_3'] = predictions['query'].map(matched_3)
    predictions['matched_10'] = predictions['query'].map(matched_10)
    return predictions


def create_input_file(path):
    folders = list(Path(path).glob("*"))
    ids = []
    animal_types = []
    images = []

    for folder in folders:
        individual_id = folder.name
        animal_type = json.load(open(folder / "card.json"))['animal']
        image = [str(f) for f in folder.glob("*.*g")]

        ids.extend([individual_id] * len(image))
        animal_types.extend([animal_type] * len(image))
        images.extend(image)
    df = pd.DataFrame({"name": ids, "animal_type": animal_types, "image_path": images})
    df['name'] = df['name'].astype("str")
    df['animal_type'] = df['animal_type'].astype("int")
    return df


def run_predict(save_dir, data_dir, model, filt=None, device='cuda:0'):
    lost_query_path = Path(data_dir) / "lost/lost/"
    lost_answer_path = Path(data_dir) / "lost/synthetic_found/"
    found_query_path = Path(data_dir) / "found/found/"
    found_answer_path = Path(data_dir) / "found/synthetic_lost/"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    lost_query = save_dir / "lost_query.csv"
    lost_answer = save_dir / "lost_answer.csv"
    found_query = save_dir / "found_query.csv"
    found_answer = save_dir / "found_answer.csv"

    create_input_file(lost_query_path).to_csv(lost_query, index=False)
    create_input_file(lost_answer_path).to_csv(lost_answer, index=False)
    create_input_file(found_query_path).to_csv(found_query, index=False)
    create_input_file(found_answer_path).to_csv(found_answer, index=False)

    lost_query_emb = save_dir / f"lost_query.npy"
    lost_answer_emb = save_dir /  f"lost_answer.npy"
    found_query_emb = save_dir / "found_query.npy"
    found_answer_emb = save_dir / "found_answer.npy"

    get_embeddings(model, lost_query, device, lost_query_emb)
    get_embeddings(model, lost_answer, device, lost_answer_emb)
    get_embeddings(model, found_query, device, found_query_emb)
    get_embeddings(model, found_answer, device, found_answer_emb)

    simi_lost = save_dir / "lost_simi.csv"
    simi_found = save_dir / "found_simi.csv"
    get_similarity(lost_query, lost_query_emb, lost_answer, lost_answer_emb, save=simi_lost, filt=filt)
    get_similarity(found_query, found_query_emb, found_answer, found_answer_emb, save=simi_found, filt=filt)

    pred_lost = get_prediction(simi_lost)
    pred_found = get_prediction(simi_found)
    preds = pd.concat([pred_lost, pred_found], axis=0).reset_index(drop=True)
    preds.to_csv(save_dir / "preds.tsv", index=False, sep='\t')
    print(f"save in {save_dir / 'preds.tsv'}")
