import argparse
from module.net import roomnet
from data.load_data import load_LSUN
import torch
import torch.nn as nn

def run(args):
    # dataset
    tr_dataset, val_dataset = load_LSUN()
    tr_loader, val_loader = '', ''
    
    # optimizer
    optimizer = torch.optim.SGD(lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 200], gamma=0.2)
    
    # model
    model = roomnet()

    # criterion
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    

    # Forward
    for epoch in range(args.total_epoch):
        # Train
        model.train()
        for data in tr_loader:
            input, label, keypoint = data[0].to('cuda'), data[1].to('cuda'), data[2].to('cuda')
            out_cls, out_key = model(input)
            
            # Loss Calculation            
            loss_cls = criterion_cls(out_cls, label)
            # Gradient Manipulation (balance background and foreground gradients)
            loss_reg_forward = criterion_reg(out_key[:, mask, mask, label], keypoint)
            loss_reg_background = criterion_reg(out_key[:, mask, mask, label], keypoint)
            loss_reg = loss_reg_forward + loss_reg_background * 0.1
            
            loss = loss_cls * 5 + loss_reg
            
            # Step
            optimizer.no_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc = ''
            
        
        # Validation
        model.eval()
        for data in val_loader:
            input, label, keypoint = data[0].to('cuda'), data[1].to('cuda'), data[2].to('cuda')
            out_cls, out_key = model(input)
            
            # Metrics
            acc = ''
        
        
        # Reduce LR    
        scheduler.step()


    # Save Model






if __name__=='__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--lr', type=float, default=0.00001)
	parser.add_argument('--total_epoch', type=int, default=225)
	args = parser.parse_args()