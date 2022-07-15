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
  #device = "cuda"
  model = roomnet()
  model.load_state_dict(torch.load(model_dir))
  # model = model.to(device)
  model.eval()

  fout=open(os.path.join(outdir, 'acc_test.txt'), 'w') # Luc: modified out text file

  # Get the test_dataloader
  _, val_dataset = load_LSUN(args)
  val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=8)

  # Loop over test data
  for data in tqdm(val_loader):
    input, lay_gt, mask_f, mask_b, label_gt = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[8].to(device)

    # Forward pass to get model outputs
    pred_class, pred_lay = model(input)
    c_out = np.argmax(pred_class, axis=1)
    
    

  start_time=time.time()
  fetchworker=BatchFetcher(val_datadir,False, False)
  fetchworker.start()
  total_step=fetchworker.get_max_step()
  print('total steps'), total_step
  for i in range(total_step):
    im_in,lay_gt, label_gt,names=fetchworker.fetch()
    net.set_feed(im_in, lay_gt, label_gt,i)
    pred_class, pred_lay=net.run_result(sess)

    c_out=np.argmax(pred_class, axis=1)
    c_gt=np.argmax(label_gt, axis=1)
    acc=np.mean(np.array(np.equal(c_out, c_gt), np.float32))
    fout.write('%s %s\n'%(i, acc))

    for j in range(batch_size):
      img = im_in[j]
      # print class_label, label2
      outim = get_im(img, pred_lay[j], c_out, str(j))
      outim2 = get_im(img, lay_gt[j], c_gt, str(j))
      outpath=os.path.join(outdir, str(i))
      if not os.path.exists(outpath):
        os.makedirs(outpath)
      cv2.imwrite(os.path.join(outpath, '%s_gt_%s.jpg' % (names[j], class_label)), outim2)
      cv2.imwrite(os.path.join(outpath, '%s_pred_%s.jpg' % (names[j], label2)), outim)
      cv2.imwrite(os.path.join(outpath, '%s.jpg' % (names[j])), img * 255)
    print('[step: %d] [time: %s] [acc: %s]'%(i, time.time()-start_time, acc))
    net.print_loss_acc(sess)
  fetchworker.shutdown()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--out_path', type=str, default='/home/luc/models/RoomNet-Pytorch/output')
  parser.add_argument('--weights_dir', type=str, default='/home/luc/models/RoomNet-Pytorch/module/weights')
  parser.add_argument('--batch_size', type=int, default=20)


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