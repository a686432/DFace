import cv2
from utils import _load, _dump
import numpy as np

filelists = "./train.configs/train_aug_120x120.list.train"
param_fp = "./train.configs/param_all_norm_2.pkl"
param_fp_new = "./train.configs/param_aligned.pkl"

import os
import struct
import math
import numpy as np
import torch

PI = 3.1415926
import config
from pytorch_3DMM import BFMA_3DDFA_batch
from src import matlab_cp2tform
from PIL import Image, ImageDraw
from tqdm import tqdm


def alignment_3ddfa(src_img, src_pts, trans):
    """
    trans: 12
    """
    of = 0
    a = 112 / 112
    b = 96 / 96
    ref_pts = [
        [30.2946 * b + of, 51.6963 * a + of],
        [65.5318 * b + of, 51.5014 * a + of],
        [48.0252 * b + of, 71.7366 * a + of],
        [33.5493 * b + of, 92.3655 * a + of],
        [62.7299 * b + of, 92.2041 * a + of],
    ]
    crop_size = (96 + of * 2, 112 + of * 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = matlab_cp2tform.get_similarity_transform_for_cv2(s, r)
    # [r, r, tx]
    # [r, r, ty]

    # tfm

    # print(tfm)
    # exit
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    m = tfm[:2, :2]
    m = np.matmul(m, m.transpose(1, 0))
    scale = np.sqrt(m[0, 0])

    rotate_x = torch.zeros(16).reshape(4, 4)
    rotate_x[0, 0], rotate_x[1, 1], rotate_x[2, 2], rotate_x[3, 3] = 1, -1, -1, 1
    rotate_x[1, 3] = 120

    new_trans = torch.zeros(16).reshape(4, 4)
    new_trans[:3, :4] = trans.reshape(3, 4)
    new_trans[3, 3] = 1

    tfm_4 = torch.zeros(16).reshape(4, 4)
    tfm_4[:2, :2] = torch.from_numpy(tfm[:2, :2])
    tfm_4[:2, 3] = torch.from_numpy(tfm[:2, 2])
    tfm_4[3, 3] = 1
    tfm_4[2, 2] = scale

    new_trans = torch.mm(rotate_x, new_trans)
    new_trans = torch.mm(tfm_4, new_trans)
    rotate_x[1, 3] = 112
    new_trans = torch.mm(rotate_x, new_trans)

    return face_img, new_trans


def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = torch.zeros(N * 99).float().reshape(N, 99)
    alpha_exp = torch.zeros(N * 29).float().reshape(N, 29)
    if config.use_cuda:
        alpha_shp, alpha_exp = alpha_shp.cuda(), alpha_exp.cuda()

    alpha_shp[:, :40] = param[:, 12:52]
    alpha_exp[:, :10] = param[:, 52:]
    return p, offset, alpha_shp, alpha_exp


with open(filelists) as f:
    lines = f.readlines()  # [:4000]

root = "/ssd-data/lmd/train_aug_120x120"
params = torch.from_numpy(_load(param_fp))
meta = _load("./train.configs/param_whitening_2.pkl")
param_mean = torch.from_numpy(meta.get("param_mean"))
param_std = torch.from_numpy(meta.get("param_std"))
params = params * param_std + param_mean
print(params.shape)
for i in tqdm(range(len(lines))):
    test_num = i
    file = os.path.join(root, lines[test_num][:-1])
    m = params[test_num, :12].reshape(3, 4)[:, :3]
    scale = np.sqrt(np.matmul(m, m.transpose(1, 0))[0, 0])
    target = params[test_num].unsqueeze(0)
    gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(target)
    gt_face = BFMA_3DDFA_batch.BFMA_3DDFA_batch(gt_shape, gt_exp, None)
    # gt_face = BFMA_3DDFA_batch.BFMA_3DDFA_batch()
    # print(gt_rotate.shape)
    # print(gt_face.transform_face_vertices_manual(gt_rotate.cuda(), gt_offset.cuda()).shape)
    lms = gt_face.transform_face_vertices_manual(gt_rotate.cuda(), gt_offset.cuda())[
        0, :2, [4409, 12130, 8190, 5391, 10919]
    ].transpose(1, 0)
    # print(lms.data.cpu().numpy())
    lms[:, 1] = 120 - lms[:, 1]

    img = Image.open(file)

    img = np.asarray(img)[:, :, [2, 1, 0]]

    img, new_trans = alignment_3ddfa(img, lms, params[test_num, :12])
    file = file.replace("120x120", "120x120_aligned")

    cv2.imwrite(file, img)
    new_trans = new_trans[:3].reshape(-1).data.cpu()
    params[test_num, :12] = new_trans

    continue
    exit()

    # test new trans
    print(new_trans[:3])
    target[0, :12] = new_trans[:3].reshape(-1)
    gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(target)
    gt_face = BFMA_3DDFA_batch.BFMA_3DDFA_batch(gt_shape, gt_exp, None)

    lms = gt_face.transform_face_vertices_manual(gt_rotate.cuda(), gt_offset.cuda())[
        0, :2, BFMA_3DDFA_batch.BFMA_3DDFA_batch.outer_landmark
    ].transpose(1, 0)
    print(lms.data.cpu().numpy())
    lms[:, 1] = 112 - lms[:, 1]
    print(lms.data.cpu().numpy())
    cv2.imwrite("imgs/test_warp.jpg", img)

    img = Image.open("imgs/test_warp.jpg")

    drawObject = ImageDraw.Draw(img)
    # draw predicted landmarks
    for i in range(68):
        pred_point = lms[i]
        drawObject.ellipse(
            (
                pred_point[0] - 1,
                pred_point[1] - 1,
                pred_point[0] + 1,
                pred_point[1] + 1,
            ),
            fill="red",
        )

    img.save("imgs/test_warp_lms.jpg")
    exit()

params = ((params - param_mean) / param_std).data.cpu().numpy()
_dump(param_fp_new, params)
