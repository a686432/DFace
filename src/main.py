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
from data_loader import MyDataSet, LfwDataSet, FSDataSet, AFLW2000DataSet
import numpy as np
import net
import lfw
import time
import math
import bfm

from loss_batch import mixed_loss_batch
from pytorch_3DMM import BFMA, BFMA_batch
from PIL import Image
from utils import *


parser = argparse.ArgumentParser(description="Face recognition with CenterLoss")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--verify", "-v", default=True, action="store_true", help="Verify the net"
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
        # transforms.RandomCrop((112,96)),
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
transform_eval_fs = transforms.Compose(
    [transforms.CenterCrop((112, 96)), transforms.ToTensor(), normalize]
)


lr = args.lr

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))
gpu_ids = range(num_gpus)
torch.cuda.set_device(gpu_ids[0])
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

print ("loading model...")
model = net.sphere64a(pretrained=True, model_root=dict_file)
model = model.to(device)
evalset = AFLW2000DataSet(
    root="/data2/lmd_jdq/AFLW2000-3D/AFLW2000_align/", transform=transform_eval
)
if not args.verify:
    trainset = MyDataSet(
        root="/data1/jdq/imgc2/",
        max_number_class=args.number_of_class,
        transform=transform_train,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=16
    )
evalsetfs = FSDataSet(root="", filelist="./1.txt", transform=transform_eval_fs)


def asymmetric_euclidean_loss(pred, target):
    l1, l2 = 1.0, 3.0
    gamma_plus = torch.abs(target)
    gamma_pred_plus = torch.sign(target) * pred
    gamma_max = torch.max(gamma_plus, gamma_pred_plus)
    return torch.mean(
        l1 * torch.sum((gamma_plus - gamma_max) ** 2, 1)
        + l2 * torch.sum((gamma_pred_plus - gamma_max) ** 2, 1)
    )


eval_loader = torch.utils.data.DataLoader(
    evalset, batch_size=60, shuffle=False, num_workers=16
)
eval_loader_fs = torch.utils.data.DataLoader(
    evalsetfs, batch_size=1, shuffle=False, num_workers=1
)

optimizer4nn = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4
)
# optimizer4nn = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=5e-4, momentum=0.9)

criterion = [asymmetric_euclidean_loss, nn.MSELoss(), mixed_loss_batch()]
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
        data, target = data.to(device), target.to(device)
        # print device
        _, pred, vertices, target_vertices, pose_expr = model(data, target)

        # print(pose_expr)
        # exit()
        gt_label = {}
        gt_label["lm"] = target_lms.to(device)
        gt_label["shape"] = target.to(device)

        loss = criterion[2](pose_expr, pred, gt_label)

        loss_eval = criterion[1](pred, target)
        loss_eval_v = criterion[1](vertices, target_vertices)

        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()

        # print(model.state_dict()['A_BFMS.weight'])

        train_loss += loss.item()
        eval_loss += loss_eval.item()
        eval_loss_v += loss_eval_v.item()
        total += target.size(0)
        # correct += predicted.eq(target).sum().item()
        t2 = time.time() - t1
        # t1 = time.time()
        # print(loss.item())
        sys.stdout.write(
            "%d/%d Loss: %.6f | C_Loss: %.6f |  E_Loss: %.6f | EC_Loss: %.6f | EV_Loss: %.6f | EVC_Loss: %.6f | Time: %.3f"
            % (
                iter_num + 1,
                len(train_loader),
                train_loss / (iter_num % 1500 + 1),
                loss.item(),
                eval_loss / (iter_num % 1500 + 1),
                loss_eval.item(),
                eval_loss_v / (iter_num % 1500 + 1),
                loss_eval_v.item(),
                t2,
            )
        )

        iter_num += 1
        # exit()
        if batch_idx < len(train_loader) - 1:
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\r")
        sys.stdout.flush()
        # if (iter_num)%100==0:
        # 	 print(target[0][:30])
        # 	 print(pred[0][:30])
        # 	 print(torch.sqrt(torch.mean((target[0] - pred[0]) ** 2)))
        if (iter_num) % 1000 == 1:
            # if	(iter_num)<=12:

            shape_para = pred[0, 0:99].unsqueeze(1)
            exp_para = pose_expr[0, 7:36].unsqueeze(1)
            camera_para = pose_expr[0, 0:7]
            face = BFMA.BFMA(shape_para, exp_para, camera_para)
            # continue
            # face.mesh2off("off/rst%d_%d.off" % (epoch, iter_num), use_camera = True)
            pimg = recover_img_to_PIL(
                data[0].cpu() if use_cuda else data[0], normallize_mean, normallize_std
            )
            pimgt = face.mesh2image_PIL(pimg)
            pimgt.save("imgs/rst%d_%d.jpg" % (epoch, iter_num))
            # pimg = face.landmark2image_PIL(gt_label['lm'][0], pimg, use_gt_landmark = True)
            """
			for i in range(5):
				pimgt = face.mesh2image_PIL(pimg, i * 0.1)
				pimgt.save('imgs/rst%d_%d_%d.jpg' % (epoch, iter_num, i))
			"""
            # pimg = face.landmark2image_PIL(gt_label['lm'][0], pimg, use_gt_landmark = True)
            pimg = face.landmark21_2image_PIL(
                gt_label["lm"][0], pimg, use_gt_landmark=False
            )
            pimg.save("imgs/rst%d_%d_lm.jpg" % (epoch, iter_num))
            pimg = face.landmark21_2image_PIL(
                gt_label["lm"][0], pimgt, use_gt_landmark=False
            )
            pimgt.save("imgs/rst%d_%d_lm_mesh.jpg" % (epoch, iter_num))

        if (iter_num) % 1500 == 0:
            train_loss = 0
            eval_loss = 0
            total = 0
            correct = 0
            eval_loss_v = 0
            bfm.vertices2off(
                args.savefile + "_" + str(iter_num) + ".off",
                vertices[0].data.cpu().numpy().reshape(-1, 3),
            )
            bfm.vertices2off(
                args.savefile + "_" + str(iter_num) + "T.off",
                target_vertices[0].data.cpu().numpy().reshape(-1, 3),
            )
            # torch.cuda.empty_cache()
            sys.stdout.write("\n")
            sys.stdout.flush()
            state = {
                "dict": (model).state_dict(),
                "item_num": iter_num,
                #'center_loss': center_loss.state_dict(),
                #'best_accu': accuracy,s
                #'threshold': Threshold,
                #'delta_threshold': deltaThreshold,
                #'weight': criterion[1].loss_weight,
            }

            print ("Saving...\n")
            torch.save(state, args.savefile + "_" + str(iter_num) + ".cl")
            # Best_Accu = np.mean(accuracy)
            # not_save = 0
            # eval2(eval_epoch)
            # lr /= 2
            # for param_group in optimizer4nn.module.param_groups:
            # 	 param_group['lr'] = lr
            # print("Modify lr to %.5f" % lr)
            # model.eval()
            # model.train()

        continue

        if iter_num == 20000:
            lr /= 10
            for param_group in optimizer4nn.param_groups:
                param_group["lr"] = lr
            print ("Modify lr to %.5f" % lr)
        if iter_num == 40000:
            lr /= 10
            optimizer4nn.param_groups[0]["lr"] = lr
            print ("Modify lr to %.5f" % lr)
        if iter_num == 80000:
            lr /= 10
            optimizer4nn.param_groups[0]["lr"] = lr
            print ("Modify lr to %.5f" % lr)
        if iter_num == 150000:
            exit


def eval():
    # global Best_Accu,lr,not_save
    predicts = []
    labels = []
    model.eval()
    for batch_idx, (_, _2, data1, data2, target) in enumerate(eval_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        ip1, _ = model(data1)
        ip2, _ = model(data2)
        cosdistance = ip1 * ip2
        cosdistance = cosdistance.sum(dim=1) / (
            ip1.norm(dim=1).reshape(-1) * ip2.norm(dim=1).reshape(-1) + 1e-12
        )
        label = target.data.cpu().numpy()
        cosdistance = cosdistance.data.cpu().numpy()
        predicts.append(cosdistance)
        labels.append(label)
        sys.stdout.write(str(batch_idx) + "/" + str(len(eval_loader)) + "\r")
    predicts = np.array(predicts).reshape(6000, -1)
    labels = np.array(labels).reshape(6000, -1)

    accuracy = []
    thd = []
    folds = lfw.KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        best_thresh = lfw.find_best_threshold(
            thresholds, predicts[train], labels[train]
        )
        accuracy.append(lfw.eval_acc(best_thresh, predicts[test], labels[test]))
        thd.append(best_thresh)
    print (
        "LFWACC={:.4f} std={:.4f} thd={:.4f}".format(
            np.mean(accuracy), np.std(accuracy), np.mean(thd)
        )
    )


#  if not os.path.exists(savefile_r):
# 		 print("Cannot find the model!")
# 	 else:
# 		 print("Loading...\n")
# 		 state = torch.load(savefile_r)
# 		 state_dict = state['dict']
# 		 model.load_state_dict(state_dict)
# 		 eval2(0)


def eval2():
    predicts = []
    labels = []
    model.eval()
    print (len(eval_loader))
    total = 0
    count = np.zeros(100)
    for batch_idx, (data, target) in enumerate(eval_loader):
        data, target = data.to(device), target.to(device)
        zero_shape = torch.zeros(99 * data.shape[0]).reshape(-1, 99).to(device)
        _, pred, vertices, target_vertices, pose_expr = model(data, zero_shape)

        face = BFMA_batch.BFMA_batch(
            pred[:, :99], pose_expr[:, 7:36], pose_expr[:, 0:7]
        )

        pred_lms = (
            face.project_outer_landmark(use_not_moved=True)
            .transpose(2, 1)
            .reshape(target.shape[0], -1)
        )

        x_max, x_min, y_max, y_min = (
            target[:, ::2].max(1, True)[0],
            target[:, ::2].min(1, True)[0],
            target[:, 1::2].max(1, True)[0],
            target[:, 1::2].min(1, True)[0],
        )
        d_normalize = torch.sqrt((x_max - x_min) * (y_max - y_min))
        pts = pred_lms - target.float()
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
        for i in range(100):
            count[i] = count[i] + torch.sum(rst < i * 1.0 / 100)
        continue
        """
		x_max, x_min, y_max, y_min = pred_lms[:, ::2].max(1, True)[0], pred_lms[:, ::2].min(1, True)[0], pred_lms[:, 1::2].max(1, True)[0], pred_lms[:, 1::2].min(1, True)[0]
		d_normalize = torch.sqrt((x_max - x_min) * (y_max - y_min))
		pts = ( pred_lms - target.float())
		#pts[target < 0] = 0

		pts = (pts**2).float().reshape(target.shape[0],-1, 2).sum(2, True)[:, :, 0].sqrt()
		pts[target[:, ::2] < 0] = 0
		tmp = target[:, ::2]
		tmp[tmp > 0] = 1
		tmp[tmp < 0] = 0
		tmp = tmp.sum(1, True).float()
		rst = (pts.sum(1, True) / tmp) / d_normalize
		#rst = (( pts/ d_normalize)[target[:, ::2] > 0])
		total += rst.shape[0]
		for i in range(100):
			count[i] = count[i] + torch.sum(rst < i * 1.0 / 100)
		#print count / total
		#print total
		#exit()
		#break
		continue
		"""
        for epoch in range(60):
            shape_para = pred[epoch, 0:99].unsqueeze(1)
            exp_para = pose_expr[epoch, 7:36].unsqueeze(1)
            camera_para = pose_expr[epoch, 0:7]
            face = BFMA.BFMA(shape_para, exp_para, camera_para)
            # continue
            # face.mesh2off("off/rst%d_%d.off" % (epoch, iter_num), use_camera = True)
            pimg = recover_img_to_PIL(
                data[epoch].cpu() if use_cuda else data[epoch],
                normallize_mean,
                normallize_std,
            )
            pimgt = face.mesh2image_PIL(pimg)
            pimgt.save("imgs/rst%d_%d.jpg" % (epoch, iter_num))
            # pimg = face.landmark2image_PIL(gt_label['lm'][0], pimg, use_gt_landmark = True)
            """
			for i in range(5):
				pimgt = face.mesh2image_PIL(pimg, i * 0.1)
				pimgt.save('imgs/rst%d_%d_%d.jpg' % (epoch, iter_num, i))
			"""
            # pimg = face.landmark2image_PIL(gt_label['lm'][0], pimg, use_gt_landmark = True)
            pimg = face.landmark21_2image_PIL(target[epoch], pimg, use_gt_landmark=True)
            pimg.save("imgs/rst%d_%d_lm.jpg" % (epoch, iter_num))
            pimg = face.landmark21_2image_PIL(
                target[epoch], pimgt, use_gt_landmark=True
            )
            pimgt.save("imgs/rst%d_%d_lm_mesh.jpg" % (epoch, iter_num))

        exit(0)
        t2 = time.time() - t1
        # t1 = time.time()
        print (
            "Loss: %.6f | P_Loss: %.6f | V_Loss: %.6f | Time: %.3f "
            % (loss.item(), loss_eval.item(), loss_eval_v.item(), t2)
        )
    count = count * 1.0 / total
    # import numpy as np

    prn = np.load("../../PRNet/prn_aflw_rst_not_align.npz")
    _3ddfa = np.load("/data2/lmd2/3DDFA/3ddfa_aflw_rst.npz")
    import matplotlib

    matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt

    x_range = 30

    x = np.linspace(0, x_range / 100.0, x_range)
    y = count * 100

    y_prn = prn["arr_0"] * 100
    y_3ddfa = _3ddfa["arr_0"] * 100

    plt.figure()

    plt.plot(x, y[:x_range], color="red", label="ours")
    plt.plot(x, y_prn[:x_range], color="green", label="prn")
    plt.plot(x, y_3ddfa[:x_range], color="yellow", label="3ddfa")
    plt.legend(loc="lower right")
    plt.xlabel("NME normalized by bounding box size (%)")
    plt.ylabel("Number of images (%)")
    plt.title("Alignment Accuracy on AFLW2000 Dataset(68 points)")

    plt.savefig("./imgs/easyplot.jpg")

    print count


t1 = time.time()
# for epoch in range(5):
# 	 train(epoch+1)


if args.verify:
    if not os.path.exists(dict_file):
        print ("Cannot find the model!")
    else:
        print ("Loading...\n")
        print ("evaluting...")
        eval2()

else:
    for epoch in range(100):
        train(epoch + 1)
