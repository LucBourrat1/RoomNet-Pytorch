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

def post_process_img(img):
  mean=[0.5, 0.5, 0.5]
  std=[0.5, 0.5, 0.5]
  for i in range(3):
    img[i,:,:] = img[i,:,:] * std[i] + mean[i]
  img = (img * 255.).permute(1,2,0).to("cpu").numpy().astype(np.uint8)
  return img

def test(args):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  outdir=os.path.join(args.out_path, 'test')

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  # initiate the model and load checkpoint
  model = roomnet()
  model.load_state_dict(torch.load(args.weights_dir))
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
    im, lay_gt, label_gt = data[0].to(device), data[1].to(device), data[8].to(device)

    # Forward pass input to get model outputs
    pred_class, pred_lay = model(im)
    c_out = torch.argmax(pred_class, axis=1)
    acc = (c_out == label_gt).sum()/label_gt.shape[0]
    fout.write(f"{idx} {acc}\n")

    # For each image in batch, output image, gt_layout and pred_layout
    for j in range(im.shape[0]):
      img_processed = post_process_img(im[j])
      outim_pred = get_im(im[j], pred_lay[j], c_out[j])
      outim_gt = get_im(im[j], lay_gt[j], label_gt[j])
      result_img = np.hstack((img_processed, outim_gt, outim_pred))
      cv2.imwrite(os.path.join(outdir, f"{str(idx*args.batch_size+j+1).zfill(4)}.jpg"), result_img)

    idx += 1
    stop = time.time()
    print(f"[batch: {idx}] [time: {(stop-start)//60: 2.0f} min {(stop-start)%60: 2.0f} s] [accuracy: {acc*100: 3.1f} %]")
    input("continue?")

  fout.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--out_path', type=str, default='/home/luc/models/RoomNet-Pytorch/output')
  parser.add_argument('--weights_dir', type=str, default='/home/luc/models/RoomNet-Pytorch/module/weights/modified_reg_loss1/weights.pth')
  parser.add_argument('--batch_size', type=int, default=20)
  parser.add_argument('--data_root', type=str, default='/home/luc/models/RoomNet-Pytorch/data/processed')
  args = parser.parse_args()

  test(args)