import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# import warnings
import torch.optim.lr_scheduler as lr_scheduler
import sys
import os
import argparse
from data_loader import *
import numpy as np
import net
import time
import math
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from mobilenet_v2 import mobilenet_2
from fr_loss import (
    Arcface,
    CosLinear,
    CosLoss,
    softmaxLinear,
    softmaxLoss,
    RingLoss,
    CenterLoss,
)
from tensorboardX import SummaryWriter
from myeval import EvalTool
from utils import *
import config

##################################################################################################
def save_model(model, path):
    print("Saving...")
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


######################################################################################################
print("--->build argumentParser...")
parser = argparse.ArgumentParser(description="Face recognition with CenterLoss")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--verify", "-v", default=False, action="store_true", help="Verify the net"
)
parser.add_argument("--gpu", default="0", type=str, help="select the gpu")
parser.add_argument("--net", default="sphere", help="type of network")
parser.add_argument("--loss", default="cos", help="type of loss fuction")
parser.add_argument(
    "--number_of_class", "-nc", default=8631, type=int, help="The number of the class"
)
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
parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
args = parser.parse_args()
#######################################################################################################


normallize_mean = [0.485, 0.456, 0.406]
normallize_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=normallize_mean, std=normallize_std)
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((112, 96)),
        transforms.ToTensor(),
        normalize,
    ]
)
transform_eval = transforms.Compose(
    [transforms.Resize((112, 96)), transforms.ToTensor(), normalize]
)
#######################################################################################################


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))
gpu_ids = range(num_gpus)
print("num of GPU is " + str(num_gpus))
print("GPU is " + str(gpu_ids))
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(gpu_ids[0])
config.use_cuda = use_cuda
config.device = device
#######################################################################################################
dict_file = args.loadfile
lr = args.lr
print("--->loading model of " + args.net + "...")
if args.net == "sphere":
    model = net.sphere64a(pretrained=False, model_root=dict_file).to(device)
else:
    model = mobilenet_2(num_classes=512).to(device)

#######################################################################################################
if not args.verify:
    trainset = VGG2MixDataset(
        max_number_class=args.number_of_class,
        indexfile="../file_path_list_vgg2.txt",
        transform=transform_train,
        ddfa_root=config.ddfa_root,
        ddfa_filelists=args.filelists_train,
        ddfa_param_fp=args.param_fp_train,
        mix=False,
    )

    sample_weight = trainset.get_weight()
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size * num_gpus, shuffle=True, num_workers=16
    )
#####################################################################################################################


if args.loss == "cos":
    ip = CosLinear(in_features=512, out_features=args.number_of_class).to(device)
    criterion = CosLoss(num_cls=args.number_of_class, alpha=0.1).to(device)
    optimizer4nn = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": ip.parameters()}],
        lr=args.lr,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )
    use_linear = True
elif args.loss == "softmax":
    ip = softmaxLinear(in_features=512, out_features=args.number_of_class).to(device)
    criterion = softmaxLoss(num_cls=args.number_of_class).to(device)
    optimizer4nn = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": ip.parameters()}],
        lr=args.lr,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )
    use_linear = True
else:
    criterion = Arcface(embedding_size=512, classnum=args.number_of_class).to(device)
    use_linear = False
    optimizer4nn = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": criterion.parameters()}],
        lr=args.lr,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )

ex_criterion = CenterLoss(num_classes=args.number_of_class, dim_hidden=512).to(device)
model = nn.DataParallel(model, device_ids=gpu_ids)
# optimizer4nn = nn.DataParallel(optimizer4nn, device_ids=gpu_ids).module
criterion = nn.DataParallel(criterion, device_ids=gpu_ids)
ex_criterion = nn.DataParallel(ex_criterion, device_ids=gpu_ids)
if use_linear:
    ip = nn.DataParallel(ip, device_ids=gpu_ids)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer4nn, patience=40, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer4nn, step_size=1)

lr_change1 = int(1.0 * len(train_loader))  # 0.01
lr_change2 = int(2.0 * len(train_loader))  # 0.001
lr_change3 = int(3.0 * len(train_loader))  # 0.0001

if args.loss == "softmax":
    lr_change1 = int(0.1 * len(train_loader))
    lr_change2 = int(1.2 * len(train_loader))
    lr_change3 = int(2.4 * len(train_loader))
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer4nn, milestones=[lr_change1, lr_change2, lr_change3], gamma=0.1
)
iter_num = 0
train_loss = 0
correct = 0
total = 0
eval_loss = 0
eval_loss_v = 0
board_loss_every = len(train_loader) // 100  # 32686//100
print("board_loss_every " + str(board_loss_every) + "...")
board_eval_every = len(train_loader) // 10  # 32686//100
print("board_eval_every " + str(board_eval_every) + "...")

writer = SummaryWriter(
    "../tmp_log/train_stage1_"
    + args.net
    + "_"
    + args.loss
    + "_"
    + str(args.lr)
    + "_"
    + time.strftime("%m-%d:%H-%M-%S", time.localtime(time.time()))
    + "/"
)

eval_tool = EvalTool(
    batch_size=args.batch_size * num_gpus, transform=transform_eval, tb_writer=writer
)
#######################################################################################################################
def train(epoch):
    global lr, iter_num, total, correct, train_loss, eval_loss, eval_loss_v, board_loss_every
    sys.stdout.write("\n")
    print("--->Training... Epoch = %d" % epoch)
    model.train()

    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):

        x_step = batch_idx + (epoch - 1) * len(train_loader)
        if x_step % board_eval_every == 0:
            eval_tool.update_tb(
                model,
                x_step,
                eval_ytf=(x_step % (board_eval_every * 10) == 0),
                emb_idx=0,
                mask=8,
            )
            model.train()
        scheduler.step()
        data = data.to(device)
        for key in target.keys():
            target[key] = torch.tensor(target[key]).to(device).float()
        feat, _, _, _ = model(data)
        # print(feat)

        if use_linear:
            loss = torch.mean(criterion(ip(feat), target["id"].long()))
            ex_loss = ex_criterion(target["id"].long(), feat)
            loss = loss + ex_loss
            # loss = torch.mean(loss)
        else:
            loss = torch.mean(criterion(feat, target["id"].long()))
            ex_loss = ex_criterion(target["id"].long(), feat)
            loss = loss + config.weight_fr_center_loss * ex_loss
            # loss = torch.mean(loss)
        # print(loss)
        # batch_idx=0,1,2,3...
        # print loss
        if x_step % board_loss_every == 0:
            writer.add_scalar("loss", loss.item(), x_step)
            writer.add_scalar("center_loss", ex_loss.item(), x_step)

        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()

        # if x_step % board_eval_every == 0 :
        #     eval_tool.update_tb(model, x_step, eval_ytf = (x_step % (board_eval_every*10) == 0), emb_idx = 0, mask = 8)
        #     model.train()

        # if x_step == 10:W
        #     save_model(model, '../model/test.pkl')
        #     print(model.parameters())
        #     # lmd criterion[0] why
        #     save_model(criterion, '../model/test_ip.pkl')

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
        if epoch % 1 == 0:
            save_model(
                model, "../model/train_stage1_epoch_%d_" % epoch + args.loss + ".pkl"
            )
            # lmd criterion[0] why
            save_model(
                criterion, "../model/train_stage1_%d_" % epoch + args.loss + "_ip.pkl"
            )
