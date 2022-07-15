import tensorflow as tf
import numpy as np
from tf_utils import *
from net import *
import os
from BatchFetcher import *
import cv2
import shutil
import time
import argparse
from get_res import get_im
batch_size=20
s_in=320
s_out=40
max_epoch=225
l_list=[0,8,14,20,24,28,34,38,42,44,46, 48]

datapath='/home/mcg/Data/LSUN/data'
datadir='/home/luc/models/roomnet/training_data'
val_datadir='/home/luc/models/roomnet/validation_data'

def test(args):
  outdir=os.path.join(args.out_path, 'test')
  model_dir=os.path.join(args.out_path, 'model')
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess=tf.Session(config=config)
  device='/gpu:0'
  if args.gpu==1:
    device='/gpu:1'
  with tf.device(device):
    if args.net=='vanilla':
      net=RoomnetVanilla()
    if args.net=='rcnn':
      net=RcnnNet()
    net.build_model()
  start_step=net.restore_model(sess, model_dir)
  print('restored')
  fout=open(os.path.join(outdir, 'acc_test.txt'), 'w') # Luc: modified out text file
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