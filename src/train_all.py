import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import warnings
from torch.autograd import Variable

warnings.filterwarnings("ignore")

import torch.optim.lr_scheduler as lr_scheduler
import sys
import os
import argparse

# from utils import AverageMeter
import torch.backends.cudnn as cudnn
from data_loader import *
import numpy as np
import net
import lfw
import time
import math
import bfm
import config
from GAN import Discriminator
from PIL import Image
from utils import *
from tqdm import tqdm

# from mobilenet_v1 import mobilenet_1
# from mobilenet_v2 import mobilenet_2

from fr_loss import CosLinear, CosLoss, softmaxLinear, softmaxLoss, RingLoss, CenterLoss
from tensorboardX import SummaryWriter

from myeval import eval_lfw, eval_cfp_fp, EvalTool
from torch.utils.data.sampler import WeightedRandomSampler
import GAN


def save_model(model, path):
    print("Saving...")
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location="cuda:0"))


# 79077
parser = argparse.ArgumentParser(description="Face recognition with CenterLoss")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--verify", "-v", default=False, action="store_true", help="Verify the net"
)
parser.add_argument("--gpu", default="0", help="select the gpu")
parser.add_argument("--net", default="sphere", help="type of network")
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


print("*************")
args = parser.parse_args()
dict_file = args.loadfile
normallize_mean = [0.485, 0.456, 0.406]
normallize_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=normallize_mean, std=normallize_std)
transform_train = transforms.Compose(
    [transforms.RandomCrop((112, 96)), transforms.ToTensor(), normalize]
)

transform_train_3ddfa = transforms.Compose(
    [transforms.Resize((112, 112)), transforms.ToTensor(), normalize]
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
# d = GAN.Discriminator()
config.use_cuda = use_cuda
config.device = device

from loss_batch import (
    _parse_param_batch,
    loss_vdc_3ddfa,
    mixed_loss_batch,
    mixed_loss_FR_batch,
)
from pytorch_3DMM import BFMA_batch

print("loading model...")

if args.net == "sphere":
    model = net.sphere64a(pretrained=False, model_root=dict_file)

else:
    model = mobilenet_2(num_classes=512)
    load_model(model, "../model/epoch_0_mobile.pkl")

model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=gpu_ids)
if not args.verify:
    # trainset = MyDataSet(root="/data1/jdq/imgc2/",max_number_class=args.number_of_class, transform=transform_train)

    trainset = VGG2MixDataset(
        max_number_class=args.number_of_class,
        indexfile="../file_path_list_vgg2.txt",
        transform=transform_train,
        ddfa_root="/data/jdq/train_aug_120x120_aligned",
        ddfa_filelists=args.filelists_train,
        ddfa_param_fp=args.param_fp_train,
        mix=True,
    )
    """
    trainset = VGG2MixDataset(
        max_number_class=args.number_of_class,
        indexfile="/ssd-data/jdq/imgc2/file_path_list_celeb_crop.txt",
        transform = transform_train
    )
    """

    sample_weight = trainset.get_weight()
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=True, num_workers=16)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size * num_gpus,
        shuffle=False,
        num_workers=16,
        sampler=sampler,
    )
    lr_change1 = int(config.lr_change1 * len(train_loader))
    lr_change2 = int(config.lr_change2 * len(train_loader))
    lr_change3 = int(config.lr_change3 * len(train_loader))
    lr_change4 = int(config.lr_change4 * len(train_loader))
    ip = CosLinear(in_features=99, out_features=args.number_of_class)
    #############################################################################################
    discriminator_activation_function = torch.relu
    d_hidden_size = 1024
    d_output_size = 1
    # d_learning_rate = 2e-4
    sgd_momentum = 0.9
    D = Discriminator(
        input_size=128,
        hidden_size=d_hidden_size,
        output_size=d_output_size,
        f=discriminator_activation_function,
    ).cuda()
    D = torch.nn.DataParallel(D, device_ids=gpu_ids)
    d_optimizer = torch.optim.SGD(
        D.parameters(),
        lr=config.d_learning_rate,
        momentum=sgd_momentum,
        weight_decay=5e-4,
    )

    ##############################################################################################

    # ip = softmaxLinear (in_features = 512, out_features = args.number_of_class)

    # fr_loss_sup = RingLoss(loss_weight=0.01)
    fr_loss_sup = CenterLoss(num_classes=args.number_of_class, dim_hidden=99)
    criterion = mixed_loss_FR_batch(
        fr_ip=ip,
        fr_loss=CosLoss(num_cls=args.number_of_class, alpha=0.4),
        fr_loss_sup=fr_loss_sup,
        d=D,
    ).to(device)
    criterion = torch.nn.DataParallel(criterion, device_ids=gpu_ids)
    optimizer4nn = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": ip.parameters()}],
        lr=args.lr,
        weight_decay=5e-4,
    )
    optimizer4cl = torch.optim.SGD(
        fr_loss_sup.parameters(), lr=config.center_lr, momentum=sgd_momentum
    )
    # optimizer4cl = torch.nn.DataParallel(optimizer4cl,device_ids=gpu_ids).module
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer4nn,
        milestones=[lr_change1, lr_change2, lr_change3, lr_change4],
        gamma=0.1,
    )
    scheduler_center = optim.lr_scheduler.MultiStepLR(
        optimizer4cl,
        milestones=[lr_change1, lr_change2, lr_change3, lr_change4],
        gamma=0.1,
    )
    scheduler_d_optimizer = optim.lr_scheduler.MultiStepLR(
        d_optimizer,
        milestones=[lr_change1, lr_change2, lr_change3, lr_change4],
        gamma=0.1,
    )
    board_loss_every = len(train_loader) // 100  # 32686//100
    print("board_loss_every " + str(board_loss_every) + "...")
    board_eval_every = len(train_loader) // 10  # 32686//100
    print("board_eval_every " + str(board_eval_every) + "...")
    criterion_B = torch.nn.BCELoss().cuda()
    criterion_B = torch.nn.DataParallel(criterion_B, device_ids=gpu_ids)

    # else:

    evalset = AFLW2000DataSet(
        root="/data/jdq/AFLW2000-3D/AFLW2000_align/", transform=transform_eval
    )
    eval_loader = torch.utils.data.DataLoader(
        evalset, batch_size=60 * num_gpus, shuffle=False, num_workers=16
    )
    evalset_micc = MICCDataSet(
        root=config.micc_image_root,
        filelist=config.micc_filelist,
        transform=transform_train,
    )
    # self.evalset_bu3def = data_loader.BU3DEFDataSet(img_root=config.bu_image_root, target_root=config.bu_obj_root, filelist=config.bu_filelist, transform=config.transform_eval_fs)
    eval_loader_micc = torch.utils.data.DataLoader(
        evalset_micc, batch_size=40 * num_gpus, shuffle=False, num_workers=1
    )
    # evalset_micc =


iter_num = 0
train_loss = 0
correct = 0
total = 0
eval_loss = 0
eval_loss_v = 0


writer = SummaryWriter(
    "../tmp_log/train_all_"
    + args.net
    + "_"
    + str(args.lr)
    + "_"
    + time.strftime("%m-%d:%H-%M-%S", time.localtime(time.time()))
    + "/"
)
eval_tool = EvalTool(batch_size=20, transform=transform_eval, tb_writer=writer)
load_model(model, dict_file)
reject_num = 0
loss_mean = 10000
loss_max = 0
norm_shape = 0
norm_feat = 0


def get_distribution_sampler():
    return lambda n, m, b: torch.Tensor(
        np.concatenate(
            (
                np.random.randn(b, n) * config.shape_ev.reshape((1, -1)),
                np.random.randn(b, m) * config.exp_ev.reshape((1, -1)),
            ),
            axis=1,
        )
    )


# def get_distribution_sampler():
#     return lambda n1, n2, b: torch.cat((torch.Tensor(np.random.randn(b,n1) * config.shape_ev.reshape((1,-1))) \
#     ,torch.Tensor(np.random.randn(b,n2) * config.exp_ev.reshape((1,-1)))), dim=1)


def extract(v):
    return v.data.storage().tolist()


for param in fr_loss_sup.parameters():
    print("2:", param)


def train(epoch):
    global t1, lr, eval_epoch, iter_num, total, correct, train_loss, eval_loss, eval_loss_v, loss_mean, reject_num, loss_max, norm_shape, norm_feat
    sys.stdout.write("\n")
    print("Training... Epoch = %d" % epoch)

    model.train()
    d_sampler = get_distribution_sampler()
    dre, dfe = 0, 0
    g_index = 100
    regular_loss_o = 0
    # for batch_idx,(data, target) in enumerate(train_loader):
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        x_step = batch_idx + (epoch - 1) * len(train_loader)
        if x_step % board_eval_every == 0 and x_step != 0:
            eval_tool.update_tb(
                model,
                x_step,
                eval_ytf=(x_step % (board_eval_every * 10) == 0 and x_step != 0),
                emb_idx=1,
                mask=4 + 8,
            )

        model.train()
        data = data.to(device)
        for key in target.keys():
            target[key] = torch.tensor(target[key]).to(device).float()

        scheduler.step()
        scheduler_center.step()
        scheduler_d_optimizer.step()

        ######################################## discriminitor ################################################

        batch_size = data.shape[0]

        g_index += 1
        d_index = 0
        if (
            g_index > config.g_steps
            and (regular_loss_o < 5 or g_index > config.g_steps * 10)
            and (regular_loss_o < 2 or g_index > config.g_steps * 3)
        ):
            d_index = 0
            _, pred_shape, feat, pose_expr = model(data)
            exp_para = pose_expr[:, 7:36]
            para = torch.cat((pred_shape, exp_para), dim=1)
            # loss, loss_fr, loss_3d,loss_center, _ , d_fake_data = criterion(feat, pose_expr, pred_shape, target)
            d_fake_data = para.detach()
            while (
                d_index < config.d_steps
                or (dfe > 0.69 and d_index < config.d_steps * 5)
                or (dfe > 20 and d_index < config.d_steps * 200)
            ):
                # 1. Train D on real+fake
                g_index = 0
                d_optimizer.zero_grad()
                d_index += 1

                #  1A: Train D on real
                d_real_data = Variable(d_sampler(99, 29, batch_size)).cuda()
                d_real_decision = D(d_real_data)
                # print(d_real_decision)
                # print(torch.ones([2,1]))
                d_real_error = criterion_B(
                    d_real_decision, Variable(torch.ones([batch_size, 1]).cuda())
                )
                # ones = true
                d_real_error.mean().backward()  # compute/store gradients, but don't change params

                d_optimizer.step()
                d_optimizer.zero_grad()

                #  1B: Train D on fake
                # d_gen_input = model

                # d_fake_data, outputs = model(data)  # detach to avoid training G on these labels

                # print(d_fake_data)
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion_B(
                    d_fake_decision, Variable(torch.zeros([batch_size, 1]).cuda())
                )  # zeros = fake
                # print(d_fake_error)
                d_fake_error.mean().backward()
                d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                dre, dfe = (
                    sum(extract(d_real_error)) / len(extract(d_real_error)),
                    sum(extract(d_fake_error)) / len(extract(d_fake_error)),
                )
                # print(dfe)

        #######################################################################################################

        # print device
        _, pred_shape, feat, pose_expr = model(data)
        exp_para = pose_expr[:, 7:36]
        # exp_para =  pose_expr[:, 7:36]
        para = torch.cat((pred_shape, exp_para), dim=1)
        # print(para)
        loss, loss_fr, loss_3d, loss_center, _, centers, weighted_centers = criterion(
            feat, pose_expr, pred_shape, target
        )
        regular_loss_o = criterion_B(
            D(para), Variable(torch.ones([batch_size, 1]).cuda())
        ).mean()
        # weighted_centers = weighted_centers.mean()
        regular_loss = regular_loss_o * config.mix_loss_weight_adv
        # print(regular_loss_o)

        # optimizer4cl.zero_grad()
        # regular_loss.backward()
        # optimizer4cl.step()
        # #model.zero_grad()
        # criterion.zero_grad()

        # _, pred_shape, feat, pose_expr= model(data)
        # loss, loss_fr, loss_3d, loss_center, _ , centers = criterion(feat, pose_expr, pred_shape, target)

        # print('111:',regular_loss,'222:',centers)
        loss = loss.mean() + regular_loss
        loss_fr = loss_fr.mean()
        loss_3d = loss_3d.mean()
        loss_center = loss_center.mean()

        pred_shape = pred_shape.data.cpu().numpy()
        centers = centers.data.cpu().numpy()
        # print(pred_shape)
        # print( np.linalg.norm(pred_shape, axis=1).mean())
        # print( np.linalg.norm(centers, axis=1).mean())
        feat = feat.data.cpu().numpy()
        exp_para = exp_para.data.cpu().numpy()
        # print()
        norm_exp = np.linalg.norm(exp_para, axis=1).mean()
        std_exp = np.linalg.norm(exp_para, axis=1).std()
        norm_center = np.linalg.norm(centers, axis=1).mean()
        norm_shape = (
            np.linalg.norm(pred_shape, axis=1).mean() * 0.01 + 0.99 * norm_shape
        )
        norm_feat = np.linalg.norm(feat, axis=1).mean() * 0.01 + 0.99 * norm_feat
        loss_mean = loss_mean * 0.99 + 0.01 * loss.item()
        loss_max = loss_mean * 10

        # print(norm_shape)
        # loss = torch.mean((feat- feat_back) ** 2)
        #        print '%f, %f, %f, %f' % (loss[0].data.cpu().numpy(), loss_land, loss_fr, loss_3d)

        # scheduler.step(loss)

        # regular_loss.backward(retain_graph=True)

        # optimizer4cl.zero_grad()
        # optimizer4nn.zero_grad()
        # loss.backward()
        # optimizer4nn.step()
        # optimizer4cl.step()

        optimizer4nn.zero_grad()
        loss.backward(retain_graph=True)
        optimizer4nn.step()
        model.zero_grad()
        criterion.zero_grad()
        optimizer4cl.zero_grad()
        weighted_centers.backward()
        optimizer4cl.step()

        # regular_loss.backward()

        # for param in fr_loss_sup.parameters():
        #    param.grad.data *= (1. / config.mix_loss_weight_fr_center_loss)

        # optimizer4nn.step()

        if x_step % board_loss_every == 0:
            writer.add_scalar("loss", loss.item(), x_step)
            # writer.add_scalar('loss_land', loss_land, x_step)
            writer.add_scalar("loss_fr", loss_fr.item(), x_step)
            writer.add_scalar("loss_3d", loss_3d.item(), x_step)
            writer.add_scalar(
                "reject_rate", 1.0 * reject_num / board_loss_every, x_step
            )
            writer.add_scalar("loss_max", loss_max, x_step)
            writer.add_scalar("loss_d", dfe, x_step)
            writer.add_scalar(
                "loss_g", regular_loss / config.mix_loss_weight_adv, x_step
            )
            norm_mean, norm_std = eval_3d()
            # if norm_mean<600:
            #     for param_group in optimizer4nn.param_groups:
            #         param_group['lr'] = param_group['lr']/10
            writer.add_scalar("norm_exp", norm_exp, x_step)
            writer.add_scalar("std_exp", std_exp, x_step)
            writer.add_scalar("norm_mean", norm_mean, x_step)
            writer.add_scalar("norm_std", norm_std, x_step)
            writer.add_scalar("norm_shape", norm_shape, x_step)
            writer.add_scalar("norm_feat", norm_feat, x_step)
            writer.add_scalar("center_loss", loss_center / (norm_shape ** 2), x_step)
            writer.add_scalar("norm_center", norm_center, x_step)
            reject_num = 0

        # continue
        # scheduler.step(loss[0].data.cpu().numpy())
        # scheduler.step(loss_fr)
        # eval_3d(1)


#        continue
# gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(target)


def eval_3d():
    predicts = np.array([])
    model.eval()
    # print("Evaluating...")
    for batch_idx, (data, target, imname) in enumerate(eval_loader_micc):
        data, target = data.to(device), target.to(device)
        _, pred_shape, feat, pose_expr = model(data)
        pred_shape = pred_shape.data.cpu().numpy()
        norm = np.linalg.norm(pred_shape, axis=1)
        predicts = np.append(predicts, norm)
        # print(np.linalg.norm(pred_shape, axis=1))
    # print(predicts.shape)
    # model.train()
    return predicts.mean(), predicts.std()


t1 = time.time()
# for epoch in range(5):
#     train(epoch+1)

# eval_3d(0)
# exit()

if args.verify:
    if not os.path.exists(dict_file):
        print("Cannot find the model!")
    else:
        print("Loading...\n")
        print("evaluting...")
        eval_3d()

else:

    for epoch in range(int(args.epoch)):
        train(epoch + 1)

        if epoch % 1 == 0:
            save_model(model, "../model/epoch_%d_" % epoch + args.net + ".pkl")
            save_model(criterion, "../model/epoch_%d_" % epoch + args.net + "_ip.pkl")
        # eval_3d(epoch)
        # exit()
