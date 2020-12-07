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
import random

# import bfm
import config

# from PIL import Image
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from myeval import EvalTool
from GAN import Discriminator

# from loss_batch import loss_vdc_3ddfa # I can't solve the problem


def save_model(model, path):
    print("Saving...")
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location="cuda:0"))


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
parser.add_argument("-b", "--batch_size", help="batch_size", default=80, type=int)

print("*************")
args = parser.parse_args()
dict_file = args.loadfile
normallize_mean = [0.485, 0.456, 0.406]
normallize_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=normallize_mean, std=normallize_std)
transform_train = transforms.Compose(
    [transforms.CenterCrop((112, 96)), transforms.ToTensor(), normalize]
)
transform_eval = transforms.Compose([transforms.ToTensor(), normalize])

lr = args.lr


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
config.device_ids = gpu_ids
print("loading model...")
if args.net == "sphere":
    model = net.sphere64a(pretrained=False, model_root=dict_file, stage=3).to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    # load_model(model, dict_file)

else:
    model = mobilenet_2(num_classes=512)

model = model.to(device)
if not args.verify:
    trainset = DDFADataset(
        root=config.ddfa_root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transform_train,
    )

    # trainset = MSCelebShapeDataSet(indexfile="/ssd-data/jdq/imgc2/file_path_list_imgc2.txt",max_number_class=args.number_of_class, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size * num_gpus, shuffle=True, num_workers=4
    )
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=False, num_workers=0, sampler = sampler)

    evalset = AFLW2000DataSet(root=config.aflw_data_root_path, transform=transform_eval)
    eval_loader = torch.utils.data.DataLoader(
        evalset, batch_size=args.batch_size * num_gpus, shuffle=False, num_workers=4
    )

optimizer4nn = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4
)
# optimizer4nn = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=5e-4, momentum=0.9)
# optimizer4nn = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=5e-4)

from loss_batch import loss_vdc_3ddfa

criterion = loss_vdc_3ddfa().to(device)
criterion = nn.DataParallel(criterion, device_ids=gpu_ids)
lr_change1 = int(4.0 * len(train_loader))
lr_change2 = int(8.0 * len(train_loader))
lr_change3 = int(10.0 * len(train_loader))
lr_change4 = int(10.0 * len(train_loader))
lr_p = pow(20, 1.0 / lr_change1)
lr_d = (
    lambda x_step: (lr_p ** x_step)
    / (int(x_step > lr_change1) * 4 + 1)
    / (int(x_step > lr_change2) * 4 + 1)
)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer4nn, milestones=[lr_change1, lr_change2, lr_change3, lr_change4], gamma=0.1
)
##############################################################################
discriminator_activation_function = torch.relu
d_hidden_size = 1024
d_output_size = 1
sgd_momentum = 0.9
D = Discriminator(
    input_size=99,
    hidden_size=d_hidden_size,
    output_size=d_output_size,
    f=discriminator_activation_function,
).cuda()
D = torch.nn.DataParallel(D, device_ids=gpu_ids)
d_optimizer = torch.optim.Adam(
    D.parameters(), lr=config.d_learning_rate, weight_decay=5e-4
)
# scheduler_d_optimizer = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[lr_change1,lr_change2], gamma=0.2)
scheduler_d_optimizer = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lr_d)
criterion_B = torch.nn.BCELoss().cuda()
criterion_B = torch.nn.DataParallel(criterion_B, device_ids=gpu_ids)
##############################################################################


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
    config.result_dir
    + "tmp_log/train_stage3_"
    + config.prefix
    + "_"
    + args.loss
    + "_"
    + str(args.lr)
    + "_"
    + time.strftime("%m-%d:%H-%M-%S", time.localtime(time.time()))
    + "/"
)

eval_tool = EvalTool(transform=transform_eval, criterion=criterion, tb_writer=writer)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer4nn, patience=board_loss_every*40, verbose=True)
loss_mean = 10000
reject_num = 0
feat_norm = 0
norm_shape = 0
norm_exp = 0


def get_distribution_sampler():
    # print(config.x)
    return lambda b: torch.Tensor(
        config.gmm_data[np.random.randint(0, 6000000, size=b)]
    )


# def get_distribution_sampler():
#     return lambda n1, n2, b: torch.cat((torch.Tensor(np.random.randn(b,n1) * config.shape_ev.reshape((1,-1))) \
#     ,torch.Tensor(np.random.randn(b,n2) * config.exp_ev.reshape((1,-1)))), dim=1)


def extract(v):
    return v.data.storage().tolist()


def train(epoch):
    global lr, iter_num, total, correct, train_loss, eval_loss, eval_loss_v, board_loss_every, loss_mean, reject_num, norm_shape, norm_exp
    sys.stdout.write("\n")
    print("Training... Epoch = %d" % epoch)
    d_sampler = get_distribution_sampler()
    dre, dfe = 0, 0
    g_index = 100
    regular_loss_o = 0
    model.train()
    # for batch_idx,(data, target) in enumerate(train_loader):
    for batch_idx, (data, target, _) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        x_step = batch_idx + (epoch - 1) * len(train_loader)

        ######################################## discriminitor ################################################
        batch_size = data.shape[0]
        g_index += 1
        d_index = 0
        _, shape, mid, camera_exp = model(data)

        if config.re_type_gan:
            regular_loss_o = criterion_B(
                D(shape), Variable(torch.ones([batch_size, 1]).cuda())
            ).mean()
            if g_index > config.g_steps and (
                regular_loss_o < 3 or g_index > config.g_steps * 5
            ):
                d_index = 0
                # _, pred_shape, feat, pose_expr= model(data)
                exp_para = camera_exp[:, 7:36]
                # para = torch.cat((pred_shape,exp_para),dim=1)
                para = shape
                # loss, loss_fr, loss_3d,loss_center, _ , d_fake_data = criterion(feat, pose_expr, pred_shape, target)
                d_fake_data = para.detach()

                while d_index < config.d_steps or (
                    dfe > 0.5 and d_index < config.d_steps * 2
                ):

                    # 1. Train D on real+fake
                    g_index = 0
                    d_optimizer.zero_grad()
                    d_index += 1

                    #  1A: Train D on real
                    d_real_data = Variable(d_sampler(batch_size)).cuda()
                    d_real_decision = D(d_real_data)
                    d_real_error = criterion_B(
                        d_real_decision, Variable(torch.ones([batch_size, 1]).cuda())
                    )
                    d_real_error.mean().backward()  # compute/store gradients, but don't change params
                    d_optimizer.step()

                    d_optimizer.zero_grad()
                    d_fake_decision = D(d_fake_data)
                    d_fake_error = criterion_B(
                        d_fake_decision, Variable(torch.zeros([batch_size, 1]).cuda())
                    )  # zeros = fake
                    d_fake_error.mean().backward()
                    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
                    d_fake_error = criterion_B(
                        d_fake_decision, Variable(torch.zeros([batch_size, 1]).cuda())
                    )
                    dre, dfe = (
                        sum(extract(d_real_error)) / len(extract(d_real_error)),
                        sum(extract(d_fake_error)) / len(extract(d_fake_error)),
                    )
                # print(dfe)
        ###############################################################################################

        data, target = data.to(device), target.to(device)

        exp_para = camera_exp[:, 7:36]

        # centers = centers.data.cpu().numpy()
        # print(pred_shape)
        # print( np.linalg.norm(pred_shape, axis=1).mean())
        # print( np.linalg.norm(centers, axis=1).mean())
        # feat = feat.data.cpu().numpy()

        # exp_para = exp_para.data.cpu().numpy()
        # #print()
        # norm_exp = np.linalg.norm(exp_para, axis=1).mean()
        # std_exp = np.linalg.norm(exp_para, axis=1).std()

        loss_3d, re_s, re_e = criterion(camera_exp, shape, target)
        if config.re_type_gan:
            regular_loss_o = criterion_B(
                D(shape), Variable(torch.ones([batch_size, 1]).cuda())
            ).mean()
            regular_loss = regular_loss_o * config.mix_loss_weight_adv
        else:
            regular_loss = re_s * config.tik_shape_weight
        loss = (
            loss_3d.mean() + regular_loss.mean() + config.tik_exp_weight * re_e.mean()
        )
        # print(regular_loss.mean())
        # loss = torch.mean(loss)

        # loss_mean = loss_mean*0.99 +0.01*loss.data.cpu().numpy()
        # loss_max = loss_mean*10
        if x_step % board_loss_every == 0:
            shape = shape.data.cpu().numpy()
            exp_para = exp_para.data.cpu().numpy()
            norm_shape = np.linalg.norm(shape, axis=1).mean()
            norm_exp = np.linalg.norm(exp_para, axis=1).mean()
            std_shape = np.linalg.norm(shape, axis=1).std()
            loss_3d = loss_3d.data.cpu().numpy().mean()

            # writer.add_scalar('loss_max', loss_max, x_step)
            writer.add_scalar(
                "reject_rate", 1.0 * reject_num / board_loss_every, x_step
            )
            writer.add_scalar("norm/shape", norm_shape, x_step)
            writer.add_scalar("norm/shape_std", std_shape, x_step)
            writer.add_scalar("norm/norm_std", norm_shape / std_shape, x_step)
            writer.add_scalar("norm/exp", norm_exp, x_step)
            writer.add_scalar("loss/loss_d", dfe, x_step)
            writer.add_scalar("loss/loss_3d", loss_3d, x_step)
            writer.add_scalar(
                "loss/loss_g",
                regular_loss.data.cpu().numpy() / config.mix_loss_weight_adv,
                x_step,
            )
            reject_num = 0
            writer.add_scalar("loss", loss.item(), x_step)

        scheduler.step()
        scheduler_d_optimizer.step()

        optimizer4nn.zero_grad()
        # if loss < loss_max :
        loss.backward()
        # else:
        # reject_num+=1
        # continue

        # loss.backward()
        optimizer4nn.step()
        if x_step % board_eval_every == 0 and x_step != 0:
            eval_tool.update_tb(
                model,
                x_step,
                eval_ytf=(x_step % (board_eval_every * 10) == 0),
                emb_idx=1,
                mask=12,
            )
            model.train()

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
                model,
                config.result_dir
                + "model/train_stage3_epoch_%d_" % epoch
                + config.prefix
                + ".pkl",
            )
