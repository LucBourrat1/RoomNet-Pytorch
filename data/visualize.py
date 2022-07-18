import cv2
from module import get_im
import argparse
from load_data import load_LSUN
import numpy as np
import os

def visualize(args):
    tr_dataset, val_dataset = load_LSUN(args)
    if args.data == "train":
        dataset = tr_dataset
    else:
        dataset = val_dataset

    idx = np.random.randint(0, len(dataset))

    flag = 0
    for i in range(idx, idx+4):
        data = dataset.__getitem__(i)
        if args.data == "train":
            img, layout, label = data[0], data[1], data[4]
        else:
            img, layout, label = data[0], data[1], data[8]
        z = get_im(img, layout, label)
        z = cv2.putText(z, f"ID: {str(i)} | Room type: {str(label.item())}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,255,0), 1, cv2.LINE_AA)
        if flag == 0:
            out_img = z
            flag = 1
        else:
            out_img = np.hstack((out_img, z))

    cv2.imwrite(os.path.join(args.out_path, "visu.jpg"), out_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, default='train')
    parser.add_argument('--out_path', type=str, default='/home/luc/models/RoomNet-Pytorch')
    parser.add_argument('--data_root', type=str, default='/home/luc/models/RoomNet-Pytorch/data/processed')
    args = parser.parse_args()

    visualize(args)