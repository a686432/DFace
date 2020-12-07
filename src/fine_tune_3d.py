import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import warnings

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

from loss_batch import _parse_param_batch, loss_vdc_3ddfa, mixed_loss_batch
from pytorch_3DMM import BFMA, BFMA_batch, BFMA_3DDFA, BFMA_batch
from PIL import Image
from utils import *
from tqdm import tqdm
from mobilenet_v1 import mobilenet_1


def save_model(model, path):
    print ("Saving...")
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


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
    "--number_of_class", "-nc", default=79077, type=int, help="The number of the class"
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

print ("*************")
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
        transforms.ToTensor(),
        normalize,
    ]
)

transform_train_3ddfa = transforms.Compose(
    [
        # transforms.RandomResizedCrop(112,scale=(0.9,1),ratio=(1, 1)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([transforms.RandomRotation(10)]),
        # transforms.RandomCrop((112,96)),
        # transforms.Resize((112, 112)),
        # transforms.CenterCrop((112, 96)),
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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))
gpu_ids = range(num_gpus)
torch.cuda.set_device(gpu_ids[0])
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

print ("loading model...")
if args.net == "sphere":
    model = net.sphere64a(pretrained=False, model_root=dict_file)
    load_model(model, "../model/epoch_90_sphere.pkl")
else:
    model = mobilenet_1()
    load_model(model, "../model/epoch_0_mobile.pkl")

model = model.to(device)
if not args.verify:
    # trainset = MyDataSet(root="/data1/jdq/imgc2/",max_number_class=args.number_of_class, transform=transform_train)

    trainset = DDFADataset(
        root="/ssd-data/lmd/train_aug_120x120_aligned",
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transform_train_3ddfa,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=16
    )

    evalset = AFLW2000DataSet(
        root="/data2/lmd_jdq/AFLW2000-3D/AFLW2000_align/", transform=transform_eval
    )
    eval_loader = torch.utils.data.DataLoader(
        evalset, batch_size=60, shuffle=False, num_workers=16
    )

optimizer4nn = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4
)
# optimizer4nn = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=5e-4, momentum=0.9)
# optimizer4nn = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=5e-4)

# optimizer4nn = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
criterion = [mixed_loss_batch(), loss_vdc_3ddfa()]

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer4nn, patience=400, verbose=True
)

if len(gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    optimizer4nn = nn.DataParallel(optimizer4nn, device_ids=gpu_ids).module
iter_num = 0
train_loss = 0
correct = 0
total = 0
eval_loss = 0
eval_loss_v = 0


def train(epoch):
    global t1, lr, eval_epoch, iter_num, total, correct, train_loss, eval_loss, eval_loss_v
    sys.stdout.write("\n")
    print ("Training... Epoch = %d" % epoch)

    model.train()
    for batch_idx, (data, target, target_lms) in enumerate(train_loader):
        # for batch_idx,(data, target, target_lms) in tqdm(enumerate(train_loader), total = len(train_loader)):
        data, target = data.to(device), target.to(device)
        # print device
        feat, pred_shape, pose_expr, feat_back = model(data)

        loss = criterion[1](pose_expr, pred_shape, target)
        # loss = torch.mean((feat- feat_back) ** 2)

        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()
        print "%f \r" % loss[0].data.cpu().numpy(),
        sys.stdout.flush()
        continue
        scheduler.step(loss[0].data.cpu().numpy())
        # eval_3d(1)
        # continue
        gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(target)
        # continue
        if batch_idx % 1000 == 0:
            i = 0
            pred_face = BFMA.BFMA(
                pred_shape[i].reshape(99, 1),
                pose_expr[i, 7:36].reshape(29, 1),
                pose_expr[i, :7],
            )
            gt_face = BFMA_3DDFA.BFMA_3DDFA(
                gt_shape[i].reshape(99, 1), gt_exp[i].reshape(29, 1), None
            )

            pimg = recover_img_to_PIL(
                data[i].cpu() if use_cuda else data[i], normallize_mean, normallize_std
            )
            pimg.save("imgs/rst%d_%d_ori.jpg" % (epoch, batch_idx))
            pimg = pred_face.mesh2image_PIL(pimg)
            # pimg = gt_face.mesh2image_PIL(pimg, gt_face.transform_face_vertices_manual(gt_rotate[i], gt_offset[i]))

            pimg.save("imgs/rst%d_%d.jpg" % (epoch, batch_idx))

            pred_face.mesh2off(
                "off/rst%d_%d_camera.off" % (epoch, batch_idx), use_camera=True
            )  # , mesh = gt_face.transform_face_vertices_manual(gt_rotate[i], gt_offset[i]))
            gt_face.mesh2off(
                "off/rst%d_%d_camera_gt.off" % (epoch, batch_idx),
                use_camera=False,
                mesh=gt_face.transform_face_vertices_manual(gt_rotate[i], gt_offset[i]),
            )

            # gt_face.mesh2off("off/rst%d_%d.off" % (epoch, batch_idx), use_camera = False)
            # exit()
        continue


def eval_3d(epoch):
    predicts = []
    labels = []
    model.eval()
    print ("Evaluating...")
    total = 0
    count = np.zeros(1000)
    for batch_idx, (data, target) in enumerate(eval_loader):
        data, target = data.to(device), target.to(device)
        zero_shape = torch.zeros(99 * data.shape[0]).reshape(-1, 99).to(device)
        _, pred, pose_expr = model(data)

        for i in range(pred.shape[0]):

            pred_face = BFMA.BFMA(
                pred[i].reshape(99, 1),
                pose_expr[i, 7:36].reshape(29, 1),
                pose_expr[i, :7],
            )
            # gt_face = BFMA_3DDFA.BFMA_3DDFA(gt_shape[i].reshape(99, 1), gt_exp[i].reshape(29, 1), None)

            pimg = recover_img_to_PIL(
                data[i].cpu() if use_cuda else data[i], normallize_mean, normallize_std
            )
            pimg.save("imgs/eval_rst%d_%d_ori.jpg" % (epoch, i))
            pimg = pred_face.mesh2image_PIL(pimg)
            # pimg = gt_face.mesh2image_PIL(pimg, gt_face.transform_face_vertices_manual(gt_rotate[i], gt_offset[i]))

            pimg.save("imgs/eval_rst%d_%d.jpg" % (epoch, i))

            # pred_face.mesh2off("off/rst%d_%d_camera.off" % (epoch, i), use_camera = True)#, mesh = gt_face.transform_face_vertices_manual(gt_rotate[i], gt_offset[i]))
            # gt_face.mesh2off("off/rst%d_%d_camera_gt.off" % (epoch, i), use_camera = False, mesh = gt_face.transform_face_vertices_manual(gt_rotate[i], gt_offset[i]))
            # exit()
        exit()

        face = BFMA_batch.BFMA_batch(
            pred[:, :99], pose_expr[:, 7:36], pose_expr[:, 0:7]
        )

        pred_lms = (
            face.project_outer_landmark(use_not_moved=True)
            .transpose(2, 1)
            .reshape(target.shape[0], -1)
        )
        # pred_lms[:, 1::2] = 112 - pred_lms[:, 1::2]
        x_max, x_min, y_max, y_min = (
            target[:, ::2].max(1, True)[0],
            target[:, ::2].min(1, True)[0],
            target[:, 1::2].max(1, True)[0],
            target[:, 1::2].min(1, True)[0],
        )
        d_normalize = torch.sqrt((x_max - x_min) * (y_max - y_min))
        pts = pred_lms - target.float()
        # print(pred_lms)
        # print(target)
        # print(pts)
        # exit()
        pts = (
            (pts ** 2)
            .float()
            .reshape(target.shape[0], -1, 2)
            .sum(2, True)[:, :, 0]
            .sqrt()
            .mean(1, True)
        )

        rst = pts / d_normalize.float()
        total += rst.shape[0]
        for i in range(1000):
            count[i] = count[i] + torch.sum(rst < i * 1.0 / 1000)
        continue

    count = count * 1.0 / total
    # import numpy as np

    prn = np.load("../../PRNet/prn_aflw_rst_not_align.npz")
    _3ddfa = np.load("/data2/lmd2/3DDFA/3ddfa_aflw_rst.npz")
    import matplotlib

    matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt

    x_range = 1000

    x = np.linspace(0, x_range / 1000.0, x_range)
    y = count * 100

    y_prn = prn["arr_0"] * 100
    y_3ddfa = _3ddfa["arr_0"] * 100

    plt.figure()

    plt.xlim(0, 0.1)
    plt.ylim(20, 100)
    plt.plot(x, y[:x_range], color="red", label="ours")
    plt.plot(x, y_prn[:x_range], color="green", label="prn")
    plt.plot(x, y_3ddfa[:x_range], color="yellow", label="3ddfa")
    plt.legend(loc="lower right")
    plt.xlabel("NME normalized by bounding box size (%)")
    plt.ylabel("Number of images (%)")
    plt.title("Alignment Accuracy on AFLW2000 Dataset(68 points)")

    plt.savefig("./imgs/NME_%d.jpg" % epoch)


t1 = time.time()
# for epoch in range(5):
#     train(epoch+1)


if args.verify:
    if not os.path.exists(dict_file):
        print ("Cannot find the model!")
    else:
        print ("Loading...\n")
        print ("evaluting...")
        eval_3d()

else:

    for epoch in range(int(args.epoch)):
        # train(epoch+1)
        # if epoch % 5 == 0:
        #    save_model(model, '../model/epoch_%d_' % epoch + args.net + '.pkl')
        eval_3d(epoch)
        exit()
