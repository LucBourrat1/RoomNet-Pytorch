import argparse
from email.mime import image
import os
from data.load_data import load_LSUN
from module.net import roomnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np


def get_data(args):
    # dataset
    tr_dataset, val_dataset = load_LSUN(args)
    tr_loader = DataLoader(tr_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    return tr_loader, val_loader


def get_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = roomnet()
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 200], gamma=0.2)

    # criterion
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    return model, optimizer, scheduler, criterion_cls, criterion_reg


@torch.no_grad()
def test_model(test_loader, model, crit_cls, crit_reg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    # Compute Accuracy
    total_nb = 0
    total_cls_acc = []
    total_cls_loss = []
    total_reg = []
    total_loss = []
    
    # Iterate through test dataset
    for data in tqdm(test_loader):
        input, layout, mask_f, mask_b, label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[8].to(device)

        # Forward pass only to get logits/output
        with torch.no_grad():
            out_cls, out_key = model(input)

        # Get predictions metrics
        total_nb += label.size(0) # total number of labels for this batch

        # cls_accuracy
        _, predicted_cls = torch.max(out_cls, 1)
        acc_cls = (label==predicted_cls).sum()/input.shape[0]*100
        total_cls_acc.append(acc_cls.item())

        # cls_loss
        loss_cls = crit_cls(out_cls, label)
        total_cls_loss.append(loss_cls.item())

        # loss_reg
        loss_reg = compute_reg_loss(
                layout, out_key, label, mask_f,mask_b, crit_reg
                )
        total_reg.append(loss_reg.item())

        # total loss
        loss = loss_cls * 5 + loss_reg
        #loss = -loss_cls * 5 + loss_reg # A TESTER
        total_loss.append(loss.item())

    cls_accuracy = np.array(total_cls_acc).mean()
    cls_loss = np.array(total_cls_loss).mean()
    reg_loss = np.array(total_reg).mean()
    loss = np.array(total_loss).mean()
  
    return cls_accuracy, cls_loss, reg_loss, loss


def compute_reg_loss(lay_gt, lay_pred, label_gt, mask_f, mask_b, crit):
  reg_loss = 0
  l_list=[0,8,14,20,24,28,34,38,42,44,46,48]
  batch_size = lay_gt.shape[0]
  
  for i in range(batch_size):
    begin = l_list[label_gt[i]]
    end = l_list[label_gt[i]+1]
    lay1 = lay_gt[i,begin:end,:,:]
    lay2 = lay_pred[i,begin:end,:,:]
    maskf = mask_f[i,begin:end,:,:]
    maskb = mask_b[i,begin:end,:,:]
    reg_loss += crit(lay1, lay2)
    reg_loss += 100 * crit(lay1*maskf, lay2*maskf)
    reg_loss += crit(lay1*maskb, lay2*maskb)
    return reg_loss


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get model and data
    tr_loader, val_loader = get_data(args)
    model, optimizer, scheduler, criterion_cls, criterion_reg = get_model(args) 

    # tensorboard
    tb = SummaryWriter(log_dir="runs/only_reg_loss")
    
    # Forward
    for epoch in range(args.total_epoch):
        start = time.time()
        # Train
        model.train()
        print(f"EPOCH: {epoch}")
        total_nb = 0
        
        total_cls_acc = []
        total_cls_loss = []
        total_reg = []
        total_loss = []

        for data in tqdm(tr_loader):
            input, layout = data[0].to(device), data[1].to(device)
            mask_f, mask_b, label = data[2].to(device), data[3].to(device), data[4].to(device)
            #grid = torchvision.utils.make_grid(input)
            # tb.add_image("input image", input[0])
            out_cls, out_key = model(input)
            
            total_nb += label.size(0)
            # Loss Computation            
            loss_cls = criterion_cls(out_cls, label)
            total_cls_loss.append(loss_cls.item())
            # Accuracy Computation
            _, pred_cls = torch.max(out_cls, axis=1)
            acc_cls = (label==pred_cls).sum()/input.shape[0]*100
            total_cls_acc.append(acc_cls.item())

            # Compute Reg loss and total loss

            # loss_reg_forward = criterion_reg(out_key * mask_f, layout * mask_f)
            # loss_reg_background = criterion_reg(out_key * mask_b, layout * mask_b)
            # loss_reg = loss_reg_forward + loss_reg_background * 0.2
            # total_reg.append(loss_reg)
            loss_reg = compute_reg_loss(
                layout, out_key, label, mask_f,mask_b, criterion_reg
                )
            total_reg.append(loss_reg.item())
            #loss = loss_cls * 5 + loss_reg
            loss = + loss_reg
            total_loss.append(loss.item())
           
            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Train metrics
        tr_cls_acc = np.array(total_cls_acc).mean()
        tr_cls_loss = np.array(total_cls_loss).mean()
        tr_reg_loss = np.array(total_reg).mean()
        tr_loss = np.array(total_loss).mean()

        print(f"TRAIN LOOP [EPOCH {epoch}]")
        print(f"epoch loss: {tr_loss}")
        print(f"epoch accuracy: {tr_cls_acc}")
        print(f"epoch class loss: {tr_cls_loss}")
        print(f"epoch reg loss: {tr_reg_loss}")
        
        # Test metrics
        val_acc, val_cls_loss, val_reg_loss, val_loss = test_model(val_loader,
                                                                   model,
                                                                   criterion_cls,
                                                                   criterion_reg)
        
        print(f"VAL LOOP [EPOCH {epoch}]")
        print(f"epoch loss: {val_loss}")
        print(f"epoch accuracy: {val_acc}")
        print(f"epoch class loss: {val_cls_loss}")
        print(f"epoch reg loss: {val_reg_loss}")
        
        # Log metrics in Tensorboard
        tb.add_scalar("train_loss", tr_loss, epoch)
        tb.add_scalar("train_reg_loss", tr_reg_loss, epoch)
        tb.add_scalar("train_cls_loss", tr_cls_loss, epoch)
        tb.add_scalar("train_cls_acc", tr_cls_acc, epoch)
        tb.add_scalar("val_loss", val_loss, epoch)
        tb.add_scalar("val_reg_loss", val_reg_loss, epoch)
        tb.add_scalar("val_cls_loss", val_cls_loss, epoch)
        tb.add_scalar("val_acuracy", val_acc, epoch)

        # Save weights if better than actual best one or if epoch == 1
        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(args. save_pth, "best.pth"))
            best_loss = val_loss
            with open("/content/data/weights/best.txt", "w") as f:
                f.write(f"Best model for epoch {epoch} with:\n")
                f.write(f"val loss = {val_loss: 0.4e}\n")
                f.write(f"val accuracy = {val_acc: 0.4e}\n")

        if val_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(args. save_pth, "best.pth"))
            with open("/content/data/weights/best.txt", "w") as f:
                f.write(f"Best model for epoch {epoch} with:\n")
                f.write(f"val loss = {val_loss: 0.4e}\n")
                f.write(f"val accuracy = {val_acc: 0.4e}\n")

        # Reduce LR    
        scheduler.step()

        # return compute time for the epoch
        stop = time.time()
        print(f"time for epoch {epoch}: {(stop - start)//60: 2.0f} min {(stop - start)%60: 2.0f} s")
        print(f"[train_loss: {tr_loss: 1.4e}] [val_loss: {val_loss: 1.4e}] [val accuracy: {val_acc: 1.4e}]\n\n")

    tb.close()

    # Save Model
    torch.save(model.state_dict(), "/content/data/weights/weights.pth")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default='/home/luc/models/RoomNet-Pytorch/data/processed')
    parser.add_argument('--save_pth', type=str, default='/home/luc/models/RoomNet-Pytorch/module/weights')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--total_epoch', type=int, default=225)
    parser.add_argument('--gpu', type=str, default='3')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    run(args)