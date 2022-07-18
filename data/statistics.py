import argparse
from load_data import load_LSUN
import numpy as np
from tqdm import tqdm
import os


from torch import argsort

def stats():
    tr_dataset, val_dataset = load_LSUN(args)
    stats_tr = np.zeros(11)
    stats_val = np.zeros(11)

    for idx in tqdm(range(len(tr_dataset))):
        data = tr_dataset.__getitem__(idx)
        label = data[4]
        stats_tr[label] += 1
    stats_tr = stats_tr / len(tr_dataset) * 100

    for idx in tqdm(range(len(val_dataset))):
        data = val_dataset.__getitem__(idx)
        label = data[8]
        stats_val[label] += 1
    stats_val = stats_val / len(val_dataset) * 100

    with open("/home/luc/models/RoomNet-Pytorch/data/statistics.txt", "w") as file:
        file.write(f"****************   TRAIN DATASET\n")
        for i in range(11):
            file.write(f"type {str(i)}: {stats_tr[i]: 2.2f} %\n")
        file.write("\n")
        file.write(f"****************   VAL DATASET\n")
        for i in range(11):
            file.write(f"type {str(i)}: {stats_val[i]: 2.2f} %\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out_path', type=str, default='/home/luc/models/RoomNet-Pytorch')
    parser.add_argument('--data_root', type=str, default='/home/luc/models/RoomNet-Pytorch/data/processed')
    args = parser.parse_args()

    stats()