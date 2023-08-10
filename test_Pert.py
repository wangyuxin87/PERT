import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils_new import *

from networks_sfnet import *
import time
from collections import OrderedDict
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Pert_Test")
parser.add_argument("--logdir", type=str, default="/logdir/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/path_to_data", help='path to training data')
parser.add_argument("--save_path", type=str, default="/checkpoint", help='path to save results')
parser.add_argument("--load_epoch", type=str, default="200", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    os.makedirs(opt.save_path, exist_ok=True)
    model = Pert(opt.use_GPU)
    if opt.use_GPU:
        model = model.cuda()
        model = nn.DataParallel(model)
    chpt_path = os.path.join(opt.logdir, 'net_epoch' + opt.load_epoch + '.pth')
    if os.path.exists(chpt_path):
        state_dict = torch.load(chpt_path)
    else:
        chpt_path2 = os.path.join(opt.logdir, 'net_latest.pth')
        print('Can not find {}. Use {}'.format(chpt_path, chpt_path2))
        state_dict = torch.load(chpt_path2)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    time_test = 0
    count = 0
    if not os.path.exists(opt.save_path + '/output'):
        os.makedirs(opt.save_path + '/output')
    if not os.path.exists(opt.save_path + '/mask'):
        os.makedirs(opt.save_path + '/mask')
    for img_name in tqdm(os.listdir(opt.data_path)):
        if is_image(img_name):

            img_path = os.path.join(opt.data_path, img_name)

            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            w,h,_ = y.shape

            if h % 32 != 0:
                h_new = h + (32 - h % 32)
            else:
                h_new = h
            if w % 32 != 0:
                w_new = w + (32 - w % 32)
            else:
                w_new = w
            y = cv2.resize(y, dsize=(h_new, w_new))

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad(): #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, x_list, out_mask, mask_list, _ = model(y)
                out = torch.clamp(out, 0., 1.)
                out_brn = torch.clamp(x_list[-1], 0., 1.)
                out_mask = torch.clamp(out_mask, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
                save_out_brn = np.uint8(255 * out_brn.data.cpu().numpy().squeeze())   #back to cpu
                save_out_mask = np.uint8(255 * out_mask.data.cpu().numpy().squeeze())   #back to cpu

            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())
                save_out_brn = np.uint8(255 * out_brn.data.numpy().squeeze())
                save_out_mask = np.uint8(255 * out_mask.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            save_out_brn = save_out_brn.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out_brn)
            save_out_brn = cv2.merge([r, g, b])

            save_out_mask = save_out_mask.transpose(1, 2, 0)
            b1, g1, r1 = cv2.split(save_out_mask)
            save_out_mask = cv2.merge([r1, g1, b1])
            cv2.imwrite(os.path.join(opt.save_path+'/output', img_name), save_out)
            cv2.imwrite(os.path.join(opt.save_path+'/mask', img_name), save_out_mask)
            count += 1

if __name__ == "__main__":
    main()

