import argparse
import os
from data.load_data import load_LSUN
from module.net import roomnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def run(args):
    device = 'cuda'
    
    # dataset
    tr_dataset, val_dataset = load_LSUN(args)
    tr_loader, val_loader = DataLoader(tr_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=8), DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    
    # model
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

        for data in tqdm(tr_loader):
            input, layout, mask_f, mask_b, label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            out_cls, out_key = model(input)
            
            # Loss Calculation            
            loss_cls = criterion_cls(out_cls, label)
            # Gradient Manipulation (balance background and foreground gradients)
            loss_reg_forward = criterion_reg(out_key * mask_f, layout * mask_f)
            loss_reg_background = criterion_reg(out_key * mask_b, layout * mask_b)
            loss_reg = loss_reg_forward + loss_reg_background * 0.2
            
            loss = loss_cls * 5 + loss_reg
            
            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc = ''
            
        
        # # Validation
        # model.eval()
        # for data in val_loader:
            
        #     # Metrics
        #     acc = ''
        
        
        # Reduce LR    
        scheduler.step()


    # Save Model






if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default='/home/dataset/LSUN/processed/')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--total_epoch', type=int, default=225)
    parser.add_argument('--gpu', type=str, default='3')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    run(args)