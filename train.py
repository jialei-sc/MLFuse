from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from network import MLFNet
from dataloader import Fusionset
from utils import mkdir
import time
import argparse
import log
import copy
from tqdm import tqdm
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NWORKERS = 0

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--exp_name', type=str, default='MT_experiments', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments/MT', help='log folder path')
parser.add_argument('--root', type=str, default='./train_dataset', help='train_val data path')  # 184
parser.add_argument('--save_path', type=str, default='./train_result/', help='model and pics save path')
parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--epoch', type=int, default=501, help='training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batchsize')  #24\12
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='MultiTask_qiuck_start_',
                    help='Name of the tensorboard summmary')

args = parser.parse_args()
writer = SummaryWriter(comment=args.summary_name)

# ==================
# init
# ==================
io = log.IOStream(args)
io.cprint(str(args))
toPIL = transforms.ToPILImage()
np.random.seed(1)  # to get the same images and leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Read Data
# ==================
dataset = Fusionset(io, args, args.root, get_patch=128, gray=True, partition='train_without_val')

train_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size, drop_last=True, shuffle=True)
# val_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
#                         sampler=valid_sampler)

torch.cuda.synchronize()
start = time.time()

# ==================
# Init Model
# ==================
model = MLFNet().to(device)   

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = CosineAnnealingLR(optimizer, args.epoch)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# Model Training
# ==================
loss_train = []
loss_val = []
mkdir(args.save_path)


print('============ Training Begins ===============')
model.train()

for epoch in tqdm(range(args.epoch)):
    total_edg_loss_per_iter_refresh = 0.
    total_ssd_loss_per_iter_refresh = 0.
    total_cmk_loss_per_iter_refresh = 0.
    total_fusion_loss_per_iter_refresh = 0.
    total_task_loss_per_iter_refresh = 0.

    total_edg_loss_per_epoch_refresh = 0.
    total_ssd_loss_per_epoch_refresh = 0.
    total_cmk_loss_per_epoch_refresh = 0.
    total_fusion_loss_per_epoch_refresh = 0.
    total_task_loss_per_epoch_refresh = 0.

    for index, image in enumerate(train_loader):
        img_1 = image[0].to(device)
        img_2 = image[1].to(device)
        
        img_vsm_1 = image[2].to(device)  
        img_vsm_2 = image[3].to(device)  

        optimizer.zero_grad()

        img_fuse, l_edg, l_cmk, l_ssd = model(img_1.float(), img_2.float())

        # save fused image
        image_fused = toPIL(img_fuse[0].squeeze(0).detach().cpu())

        if index % 40 == 0:
            image_fused.save(
                os.path.join(args.save_path, args.summary_name + '_epoch' + str(epoch) + '_' + str(
                    index) + '_train.png'))

        total_edg_loss_per_iter_refresh = model._eloss(img_1, l_edg)
        total_ssd_loss_per_iter_refresh = model._dloss(img_2, l_ssd, img_vsm_2)
        total_cmk_loss_per_iter_refresh = model._closs(img_1, img_2, l_cmk)
        total_fusion_loss_per_iter_refresh = model._floss(img_1, img_2, img_fuse)
        total_task_loss_per_iter_refresh = 0.3*total_edg_loss_per_iter_refresh + 3*total_ssd_loss_per_iter_refresh + 0.5*total_cmk_loss_per_iter_refresh + total_fusion_loss_per_iter_refresh
        total_task_loss_per_iter_refresh.backward()
        optimizer.step()
        # step = step + 1
        #----------------
        # total task loss
        #----------------
        total_task_loss_per_epoch_refresh += total_task_loss_per_iter_refresh
        total_edg_loss_per_epoch_refresh += total_edg_loss_per_iter_refresh
        total_ssd_loss_per_epoch_refresh += total_ssd_loss_per_iter_refresh
        total_cmk_loss_per_epoch_refresh += total_cmk_loss_per_iter_refresh
        total_fusion_loss_per_epoch_refresh += total_fusion_loss_per_iter_refresh


    print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (
        epoch, args.epoch, total_task_loss_per_epoch_refresh / (len(train_loader))))
    writer.add_scalar('Train/total_task_loss', total_task_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_edg_loss', total_edg_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_cmk_loss', total_cmk_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_ssd_loss', total_ssd_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_fusion_loss', total_fusion_loss_per_epoch_refresh / (len(train_loader)),epoch)

    loss_train.append(total_task_loss_per_epoch_refresh / (len(train_loader)))
    scheduler.step()

    # ==================
    # Model Saving
    # ==================
    # save model every epoch
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
    }
    if epoch % 100 == 0:
        torch.save(state, os.path.join(args.save_path,
                                   args.summary_name + 'epoch_' + str(epoch) + '_' + '.pth'))
torch.cuda.synchronize()
end = time.time()

# save best model
# minloss_index = loss_val.index(min(loss_val))
# print("The min loss in validation is obtained in %d epoch" % (minloss_index))
print("The training process has finished! Take a break! ")
