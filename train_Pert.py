import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils_new import *
from torch.optim.lr_scheduler import MultiStepLR
from loss_pert import loss_pert
from networks_sfnet import *
from data.dataloader import ErasingData

parser = argparse.ArgumentParser(description="Pert_train")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[60,100,160], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="/checkpoint", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="/path_to_data",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
numOfGPUs = torch.cuda.device_count()

def main():

    print('Loading dataset ...\n')
    dataset_train = ErasingData(opt.data_path, (512, 512), training=True)
    loader_train = DataLoader(dataset_train, batch_size=opt.batch_size,
                            shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = Pert(use_GPU=opt.use_gpu)

    # loss function
    criterion = loss_pert()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()
        if numOfGPUs >= 1:
            print('Parallel Computing!\n')
            model = nn.DataParallel(model, device_ids=range(numOfGPUs))
            criterion = nn.DataParallel(criterion, device_ids=range(numOfGPUs))

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)),
                              strict=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        print('yesss!!!')
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train, target_mask, path) in enumerate(loader_train):

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train, target_mask = input_train.cuda(), target_train.cuda(), target_mask.cuda()

            out_train, x_list, out_mask, mask_list, out_list = model(input_train)
            pixel_metric = criterion(target_train, out_train, out_mask, target_mask, mask_list, out_list, x_list)
            loss = -pixel_metric
            loss = loss.sum()
            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _, out_mask, _, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            out_mask = torch.clamp(out_mask, 0., 1.)
            print("[epoch %d][%d/%d] loss: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            step += 1

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    main()
