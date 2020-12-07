import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch_3DMM import BFMA_batch, BFMA_3DDFA_batch
import config
import sys
from utils import *
from fr_loss import CosLoss, RingLoss
import BFMF_batch
import BFMG_batch


class mixed_loss_batch(nn.Module):
    def __init__(self):
        super(mixed_loss_batch, self).__init__()

    def forward(self, pred_camera_exp, pred_shape, gt_lable):
        batch_size = pred_camera_exp.shape[0]

        shape_para = pred_shape[:, 0:99]
        exp_para = pred_camera_exp[:, 7:36]
        camera_para = pred_camera_exp[:, 0:7]
        gt_2d_landmark = gt_lable["lm"]
        gt_shape = gt_lable["shape"]
        # gt_expr = gt_lable['expr']

        face = BFMA_batch.BFMA_batch(shape_para, exp_para, camera_para)
        # gt_face_bfmz = BFMZ_batch.BFMZ_batch(gt_shape, gt_expr)

        loss_2d_landmark = face.get_2d_landmark_loss(gt_2d_landmark)

        # loss_3d_bfmz = face.get_3D_loss_bfmz(gt_face_bfmz.get_face_vertex())
        tikhonov_regularization = face.get_tikhonov_regularization()

        mixed_loss = (
            loss_2d_landmark.reshape(-1, 1)
            + config.loss_weight_tik * tikhonov_regularization
        )  # + config.loss_weight_bfmz_3d * loss_3d_bfmz
        #
        # print(loss_2d_landmark.reshape(-1, 1).shape)
        # print(tikhonov_regularization.shape)
        # exit()
        loss = torch.mean(mixed_loss)

        return loss


def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = torch.zeros(N * 99).float().reshape(N, 99)
    alpha_exp = torch.zeros(N * 29).float().reshape(N, 29)
    if config.use_cuda:
        alpha_shp, alpha_exp = alpha_shp.to(config.device), alpha_exp.to(config.device)

    alpha_shp[:, :40] = param[:, 12:52]
    alpha_exp[:, :10] = param[:, 52:]
    return p, offset, alpha_shp, alpha_exp


class loss_vdc_3ddfa(nn.Module):
    def __init__(self):
        super(loss_vdc_3ddfa, self).__init__()
        self.bfma_3ddfa = BFMA_3DDFA_batch.BFMA_3DDFA_batch()
        self.bfmf = BFMG_batch.BFMG_batch()
        self.weights_landmarks = torch.ones(68)
        # self.weights_landmarks[17:68] = self.weights_landmarks[17:68]*10
        self.register_buffer("weights_landmark", self.weights_landmarks)

    def forward(self, pred_camera_exp, pred_shape, target):

        batch_size = pred_camera_exp.shape[0]
        gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(target)

        shape_para = pred_shape[:, 0:99]
        exp_para = pred_camera_exp[:, 7:36]
        camera_para = pred_camera_exp[:, 0:7]
        # print(shape_para.device, exp_para.device, camera_para.device, target.device)
        pred_face, face_rotated, face_proj = self.bfmf(
            shape_para, exp_para, camera_para
        )
        gt_face, gt_face_rotated, gt_face_proj = self.bfma_3ddfa(
            gt_shape, gt_exp, gt_rotate, gt_offset
        )
        mse = self.bfma_3ddfa.get_landmark_68(gt_face_proj) - self.bfmf.get_landmark_68(
            face_proj
        )
        vdc_landmark = torch.norm(mse, p=2, dim=1)

        regular_loss_shape = self.bfmf.get_batch_distribution_regular_shape(shape_para)
        regular_loss_exp = self.bfmf.get_tikhonov_regularization_exp(exp_para)
        # regular_loss =get_tikhonov_regularization(self, Shape_Para, Exp_Para)
        loss = config.mix_loss_weight_3d_vdc_lm * torch.mean(
            vdc_landmark.reshape(batch_size, -1), dim=1
        )  # config.loss_weight_tik*regular_loss
        """
        print(1, torch.mean(torch.mean(config.mix_loss_weight_3d_vdc * vdc.reshape(batch_size, -1), dim = 1) ))
        print(2, torch.mean(config.mix_loss_weight_3d_vdc_lm * torch.mean(vdc_landmark.reshape(batch_size, -1), dim = 1)))
        print(3, torch.mean(loss_pdc))

        """
        return loss, regular_loss_shape, regular_loss_exp


class mixed_loss_FR_batch(nn.Module):
    def __init__(self, fr_ip, fr_loss, fr_loss_sup=None, d=None):
        super(mixed_loss_FR_batch, self).__init__()
        self.fr_ip = fr_ip
        self.fr_loss = fr_loss
        self.loss_3d_func = loss_vdc_3ddfa()
        self.fr_loss_sup = fr_loss_sup
        self.D = d
        # self.bfma =BFMA_batch.BFMA_batch()
        # RingLoss(loss_weight=0.01)

    def forward(self, feature_fr, pred_camera_exp, pred_shape, gt_label):
        batch_size = pred_camera_exp.shape[0]

        # shape_para = pred_shape[:, 0:99]
        # exp_para = pred_camera_exp[:, 7:36]
        # camera_para = pred_camera_exp[:, 0:7]

        fr_embedding = pred_shape

        # face = self.bfma(shape_para, exp_para, camera_para)
        # face = face.cuda()

        # gt_2d_landmark = gt_label['lm']
        # loss_2d_landmark = (gt_label['ind_lm']) * (face.get_2d_landmark_loss(gt_2d_landmark) + config.loss_weight_tik * face.get_tikhonov_regularization().reshape(-1))
        # loss_2d_landmark = (gt_label['ind_lm']) * (face.get_2d_landmark_loss(gt_2d_landmark) + config.loss_weight_tik * face.get_batch_distribution_regular(shape_para, exp_para).reshape(-1))
        # loss_2d_landmark = torch.sum(loss_2d_landmark) / torch.sum(gt_label['ind_lm']) if torch.sum(gt_label['ind_lm']) >= 1 else loss_2d_landmark * 0
        # loss_2d_landmark = torch.Tensor([0]).cuda()

        gt_id = gt_label["id"].long()
        loss_fr = gt_label["ind_id"].float() * self.fr_loss(
            self.fr_ip(fr_embedding), gt_id
        )

        # weighted_center_loss = None
        weights = get_weighted(pred_camera_exp[:, 7:36], pred_camera_exp[:, 0:7])

        loss_center, centers, weighted_center_loss = (
            self.fr_loss_sup(gt_id, fr_embedding, gt_label["ind_id"].float(), weights)
            if self.fr_loss_sup is not None
            else 0
        )
        # gan_loss = self.D()
        # weighted_center_loss = loss_center*weights.reshape(batch_size,1)
        loss_center = (
            torch.sum(loss_center) / torch.sum(gt_label["ind_id"].float())
            if torch.sum(gt_label["ind_id"]) >= 1
            else torch.sum(loss_center * 0)
        )

        weighted_center_loss = (
            torch.sum(weighted_center_loss) / torch.sum(gt_label["ind_id"].float())
            if torch.sum(gt_label["ind_id"]) >= 1
            else torch.sum(weighted_center_loss * 0)
        )
        # loss_center = loss_center / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_center * 0)
        loss_fr = (
            torch.sum(loss_fr) / torch.sum(gt_label["ind_id"].float())
            if torch.sum(gt_label["ind_id"]) >= 1
            else torch.sum(loss_fr * 0)
        )

        loss_fr_all = loss_center * config.mix_loss_weight_fr_center_loss + loss_fr
        weighted_center_loss = (
            weighted_center_loss
            * config.mix_loss_weight_fr_center_loss
            * config.mix_loss_weight_fr
        )
        # loss_fr = feature_fr
        # print(gt_label['ind_id'].float())

        gt_3d = gt_label["3d"]
        loss_3d, regular_loss_shape, regular_loss_exp = self.loss_3d_func(
            pred_camera_exp, pred_shape, gt_3d
        )

        loss_3d = (
            torch.sum(gt_label["ind_3d"].float() * loss_3d)
            / torch.sum(gt_label["ind_3d"].float())
            if torch.sum(gt_label["ind_3d"]) >= 1
            else torch.sum(loss_3d * 0)
        )
        loss = (
            loss_3d
            + config.mix_loss_weight_fr * loss_fr_all
            + regular_loss_exp * config.loss_weight_tik_exp
        )
        # print(loss)
        # print(loss.shape, loss_3d.shape, loss_2d_landmark.shape)
        # print "%f %f %f %f\r" %  (torch.mean(loss_2d_landmark).data.cpu().numpy(), torch.mean(loss_fr).data.cpu().numpy(),torch.mean(loss_3d).data.cpu().numpy(), loss.data.cpu().numpy())
        # loss = loss_fr

        return (
            loss,
            loss_fr,
            loss_3d / config.mix_loss_weight_3d_vdc_lm,
            loss_center,
            regular_loss_shape,
            centers,
            weighted_center_loss,
        )


def get_weighted(exp_para, pose_para):
    """
    get pose_texture
    get exp_norm
    """
    R = Q2R_batch(pose_para[:, 0:4])
    x, y, z = matrix2angle(R)
    return (x.cos() + y.cos() + z.cos() + 1 / (1 + torch.norm(exp_para, dim=1))) / 4


def asymmetric_euclidean_loss(pred, target):
    l1, l2 = 1.0, 3.0
    gamma_plus = torch.abs(target)
    gamma_pred_plus = torch.sign(target) * pred
    gamma_max = torch.max(gamma_plus, gamma_pred_plus)
    return torch.mean(
        l1 * torch.sum((gamma_plus - gamma_max) ** 2, 1)
        + l2 * torch.sum((gamma_pred_plus - gamma_max) ** 2, 1)
    )
