import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

import torch.optim.lr_scheduler as lr_scheduler
import sys
import os
import argparse

# from utils import AverageMeter
# import torch.backends.cudnn as cudnn
from data_loader import *
import numpy as np
import net

# import lfw
import time
import math

# import bfm
import config

# from PIL import Image
from utils import *
from tqdm import tqdm

from tensorboardX import SummaryWriter

from myeval import EvalTool

# from loss_batch import loss_vdc_3ddfa


def save_model(model, path):
    print("Saving...")
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


# 79077 8631
parser = argparse.ArgumentParser(description="Face recognition with CenterLoss")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")

parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--verify", "-v", default=False, action="store_true", help="Verify the net"
)
parser.add_argument("--gpu", default="0", help="select the gpu")
parser.add_argument("--net", default="sphere", help="type of network")
parser.add_argument("--loss", default="cos", help="type of loss fuction")
parser.add_argument(
    "--loadfile",
    "-l",
    default="/data3/jdq/fs2_81000.cl",
    help="model parameter filename",
)
parser.add_argument(
    "--savefile", "-S", default="../dict.cl", help="model parameter filename"
)
parser.add_argument("--param-fp-train", default="./train.configs/param_aligned.pkl")
parser.add_argument(
    "--filelists-train", default="./train.configs/train_aug_120x120.list.train"
)
parser.add_argument("--epoch", "-e", default=50, help="training epoch")
parser.add_argument("--lfw_vallist", "-vl", default="/data1/jdq/lfw_crop/")
parser.add_argument("--lfw_pairlist", "-pl", default="../lfw_pair.txt")
parser.add_argument("--batch_size", "-b", help="batch_size", default=40, type=int)
parser.add_argument("--lqy", default="lqy", help="private name of log")


print("*************")
args = parser.parse_args()
dict_file = args.loadfile
normallize_mean = [0.485, 0.456, 0.406]
normallize_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=normallize_mean, std=normallize_std)
transform_train = transforms.Compose(
    [
        # transforms.RandomResizedCrop(112,scale=(0.9,1),ratio=(1, 1)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([transforms.RandomRotation(10)]),
        transforms.RandomCrop((112, 96)),
        # transforms.Resize((112, 112)),
        transforms.ToTensor(),
        normalize,
    ]
)
transform_eval = transforms.Compose(
    [
        # transforms.Resize((112, 96)),
        transforms.ToTensor(),
        normalize,
    ]
)

lr = args.lr

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# num_gpus=len(args.gpu.split(","))
# gpu_ids = range(num_gpus)
# torch.cuda.set_device(gpu_ids[0])
# use_cuda = torch.cuda.is_available() and True
# device = torch.device("cuda" if use_cuda else "cpu")

num_gpus = len(args.gpu.split(","))
gpu_ids = []
for i in range(num_gpus):
    gpu_ids.append(int(args.gpu.split(",")[i]))
print("num of GPU is " + str(num_gpus))
print("GPU is " + str(gpu_ids))
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(gpu_ids[0])

config.use_cuda = use_cuda
config.device = device

print("loading model...")
if args.net == "sphere":
    model = net.sphere64a(pretrained=False, model_root=dict_file, stage=0)
    load_model(model, dict_file)
else:
    model = mobilenet_2(num_classes=512)

model = model.to(device)
if not args.verify:
    trainset = DDFADataset(
        root="/data/jdq/train_aug_120x120_aligned",
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transform_train,
    )

    # trainset = MSCelebShapeDataSet(indexfile="/ssd-data/jdq/imgc2/file_path_list_imgc2.txt",max_number_class=args.number_of_class, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=False, num_workers=0, sampler = sampler)

    evalset = AFLW2000DataSet(
        root="/data/jdq/AFLW2000-3D/AFLW2000_align/", transform=transform_eval
    )
    eval_loader = torch.utils.data.DataLoader(
        evalset, batch_size=60, shuffle=False, num_workers=16
    )

optimizer4nn = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4
)
# optimizer4nn = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=5e-4, momentum=0.9)
# optimizer4nn = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=5e-4)

from loss_batch import loss_vdc_3ddfa

criterion = [loss_vdc_3ddfa()]


# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer4nn, patience=40, verbose=True)
scheduler = optim.lr_scheduler.StepLR(optimizer4nn, step_size=1)

if len(gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    optimizer4nn = nn.DataParallel(optimizer4nn, device_ids=gpu_ids).module
iter_num = 0
train_loss = 0
correct = 0
total = 0
eval_loss = 0
eval_loss_v = 0
board_loss_every = len(train_loader) // 100  # 32686//100
print("board_loss_every " + str(board_loss_every) + "...")

writer = SummaryWriter(
    "../tmp_log/train_stage4_"
    + args.net
    + "_"
    + args.loss
    + "_"
    + str(args.lr)
    + "_"
    + time.strftime("%m-%d:%H-%M-%S", time.localtime(time.time()))
    + "_("
    + args.lqy
    + ")"
    + "/"
)

eval_tool = EvalTool(transform=transform_eval, tb_writer=writer)


def train(epoch):
    global lr, iter_num, total, correct, train_loss, eval_loss, eval_loss_v, board_loss_every
    sys.stdout.write("\n")
    print("Training... Epoch = %d" % epoch)

    model.train()
    # for batch_idx,(data, target) in enumerate(train_loader):
    for batch_idx, (data, target, _) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        model.train()
        data, target = data.to(device), target.to(device)

        _, shape, camera_exp = model(data)

        loss = torch.mean(criterion[0](camera_exp, shape, target))

        x_step = batch_idx + (epoch - 1) * len(train_loader)

        if x_step % board_loss_every == 0:
            writer.add_scalar("loss", loss.item(), x_step)

        optimizer4nn.zero_grad()
        if loss < 500:
            loss.backward()
        else:
            continue
        optimizer4nn.step()

        if x_step % 5000 == 0 and x_step != 0:
            eval_tool.update_tb(
                model, x_step, eval_ytf=(x_step % 30000 == 0), emb_idx=1, mask=13
            )
        continue


if args.verify:
    if not os.path.exists(dict_file):
        print("Cannot find the model!")
    else:
        print("Loading...\n")
        print("evaluting...")

else:

    for epoch in range(int(args.epoch)):
        train(epoch + 1)
        scheduler.step()
        if epoch % 2 == 0:
            save_model(
                model, "../model/train_stage4_epoch_%d_" % epoch + args.net + ".pkl"
            )
