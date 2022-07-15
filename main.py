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
import torchvision


def test_model(test_loader, model, crit):
    device = 'cuda'
    crit_cls, crit_reg = crit[0], crit[1]
    # Calculate Accuracy
    total_nb = 0 
    total_cls_acc = 0
    total_cls_loss = []
    total_reg = []
    total_loss = []
    
    # Iterate through test dataset
    for data in test_loader:
        input, layout, mask_f, mask_b, label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[8].to(device)

        # Forward pass only to get logits/output
        with torch.no_grad():
            out_cls, out_key = model(input)

        # Get predictions metrics
        total_nb += label.size(0) # total number of labels for this batch

        # cls_accuracy
        _, predicted_cls = torch.max(out_cls, 1)
        total_cls_acc += (predicted_cls == label).sum()

        # cls_loss
        loss_cls = crit_cls(out_cls, label)
        total_cls_loss.append(loss_cls)

        # loss_reg
        loss_reg_forward = crit_reg(out_key * mask_f, layout * mask_f)
        loss_reg_background = crit_reg(out_key * mask_b, layout * mask_b)
        loss_reg = loss_reg_forward + loss_reg_background * 0.2
        total_reg.append(loss_reg)

        # total loss
        loss = loss_cls * 5 + loss_reg
        total_loss.append(loss)

    cls_accuracy = 100 * float(total_cls_acc) / float(total_nb)
    cls_loss = sum(total_cls_loss)/total_nb
    reg_loss = sum(total_reg)/total_nb
    loss = sum(total_loss)/total_nb
  
    return cls_accuracy, cls_loss, reg_loss, loss


def run(args):
    device = 'cuda'
    
    # dataset
    tr_dataset, val_dataset = load_LSUN(args)
    tr_loader, val_loader = DataLoader(tr_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=8), DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    
    # model
    tb = SummaryWriter()
    model = roomnet()
    model = model.to(device)
    
    
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 200], gamma=0.2)

    # criterion
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    

    # Forward
    for epoch in range(args.total_epoch):
        # Train
        model.train()
        print(f"EPOCH: {epoch}")
        total_nb = 0 
        total_cls_loss = []
        total_reg = []
        total_loss = []

        for data in tqdm(tr_loader):
            input, layout, mask_f, mask_b, label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            #grid = torchvision.utils.make_grid(input)
            # tb.add_image("input image", input[0])
            out_cls, out_key = model(input) 
            
            total_nb += label.size(0)
            # Loss Calculation            
            loss_cls = criterion_cls(out_cls, label)
            total_cls_loss.append(loss_cls)
            # Gradient Manipulation (balance background and foreground gradients)
            loss_reg_forward = criterion_reg(out_key * mask_f, layout * mask_f)
            loss_reg_background = criterion_reg(out_key * mask_b, layout * mask_b)
            loss_reg = loss_reg_forward + loss_reg_background * 0.2
            total_reg.append(loss_reg)
            loss = loss_cls * 5 + loss_reg
            total_loss.append(loss)
           
            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Train metrics
        tr_cls_loss = sum(total_cls_loss)/total_nb
        tr_reg_loss = sum(total_reg)/total_nb
        tr_loss = sum(total_loss)/total_nb
        
        # Test metrics
        model.eval()
        crit = (criterion_cls, criterion_reg)
        val_acc, val_cls_loss, val_reg_loss, val_loss = test_model(val_loader, model, crit)

        # Log metrics in Tensorboard
        tb.add_scalar("train_loss", tr_loss, epoch)
        tb.add_scalar("train_reg_loss", tr_reg_loss, epoch)
        tb.add_scalar("train_cls_loss", tr_cls_loss, epoch)
        tb.add_scalar("val_loss", val_loss, epoch)
        tb.add_scalar("val_reg_loss", val_reg_loss, epoch)
        tb.add_scalar("val_cls_loss", val_cls_loss, epoch)
        tb.add_scalar("val_acuracy", val_acc, epoch)

        # Save weights if better than actual best one or if epoch == 1
        if epoch == 0:
            torch.save(model.state_dict(), "/home/luc/models/RoomNet-Pytorch/module/weights/best.pth")
            best_loss = val_loss
            with open("/home/luc/models/RoomNet-Pytorch/module/weights/best.txt", "w") as f:
                f.write(f"Best model for epoch {epoch} with:\n")
                f.write(f"val loss = {val_loss: 0.4e}\n")
                f.write(f"val accuracy = {val_acc: 0.4e}\n")

        elif val_loss < best_loss:
            torch.save(model.state_dict(), "/home/luc/models/RoomNet-Pytorch/module/weights/best.pth")
            with open("/home/luc/models/RoomNet-Pytorch/module/weights/best.txt", "w") as f:
                f.write(f"Best model for epoch {epoch} with:\n")
                f.write(f"val loss = {val_loss: 0.4e}\n")
                f.write(f"val accuracy = {val_acc: 0.4e}\n")

        # Reduce LR    
        scheduler.step()

    tb.close()

    # Save Model
    torch.save(model.state_dict(), "/home/luc/models/RoomNet-Pytorch/module/weights/weights.pth")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default='/home/luc/models/RoomNet-Pytorch/data/processed')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--total_epoch', type=int, default=225)
    parser.add_argument('--gpu', type=str, default='3')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    run(args)