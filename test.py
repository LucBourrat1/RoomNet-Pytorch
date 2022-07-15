from unittest import result
import torch
import numpy as np
import os
import cv2
import time
import argparse
from module import get_im, roomnet
from torch.utils.data import DataLoader
from data.load_data import load_LSUN
from tqdm import tqdm


def test(args):
  outdir=os.path.join(args.out_path, 'test')
  model_dir=os.path.join(args.weights_dir, 'best.pth')
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  # initiate the model and load checkpoint
  device = "cpu"
  model = roomnet()
  model.load_state_dict(torch.load(model_dir))
  model = model.to(device)
  model.eval()

  fout = open(os.path.join(outdir, 'acc_test.txt'), 'w') # Luc: modified out text file

  # Get the test_dataloader
  _, val_dataset = load_LSUN(args)
  val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=8)

  # Loop over test data
  idx = 0
  for data in tqdm(val_loader):
    start = time.time()
    x, lay_gt, label_gt, name, im_notr = data[0].to(device), data[1].to(device), data[8].to(device), data[9], data[10].to(device)

    # Forward pass input to get model outputs
    pred_class, pred_lay = model(x)
    c_out = torch.argmax(pred_class, axis=1)
    acc = (c_out == label_gt).sum()/label_gt.shape[0]
    fout.write(f"{idx} {acc}\n")

    # For each image in batch, output image, gt_layout and pred_layout
    for j in range(args.batch_size):
      img = im_notr[j]
      outim_pred = get_im(img, pred_lay[j], c_out[j], str(j))
      outim_gt = get_im(img, lay_gt[j], label_gt[j], str(j))
      result_img = np.hstack((img, outim_gt, outim_pred))
      cv2.imwrite(os.path.join(outdir, f"{str(idx*args.batch_size+j+1).zfill(4)}.jpg"), result_img)

    idx += 1
    stop = time.time()
    print(f"[batch: {idx}] [time: {(stop-start)//60: 2.0f} min {(stop-start)%60: 2.0f} s] [accuracy: {acc*100: 3.1f} %]")
    input("continue?")

  fout.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--out_path', type=str, default='/home/luc/Dev/RoomNet-Pytorch/output')
  parser.add_argument('--weights_dir', type=str, default='/home/luc/Dev/RoomNet-Pytorch/module/weights')
  parser.add_argument('--batch_size', type=int, default=20)
  parser.add_argument('--data_root', type=str, default='/home/luc/Dev/RoomNet-Pytorch/data/processed')


  parser.add_argument('--total_epoch', type=int, default=225)
  parser.add_argument('--gpu', type=str, default='3')
  args = parser.parse_args()

  test(args)
  
  # batch_size=20
  # s_in=320
  # s_out=40
  # max_epoch=225
  # l_list=[0,8,14,20,24,28,34,38,42,44,46, 48]

  # datapath='/home/mcg/Data/LSUN/data'
  # datadir='/home/luc/models/roomnet/training_data'
  # val_datadir='/home/luc/models/roomnet/validation_data'

  # test()