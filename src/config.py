import torch
import numpy as np
import scipy.io as sio
import pickle


server = "lab"
mode = "vae_sc5"


if server == "zj":
    eval_fr_data_root_path = "/data/jdq/eval_dbs/"
    aflw_data_root_path = "/data/jdq/AFLW2000-3D/AFLW2000_align/"
    ddfa_root = "/data/jdq/train_aug_120x120_aligned"
    fr_data_list = "/data/jdq/faces_vgg_112x112/vgg2_file_list.pkl"
    result_dir = "/data/jdq/"

else:
    aflw_data_root_path = "/ssd-data/jdq/dbs/AFLW2000-3D/AFLW2000_align/"
    eval_fr_data_root_path = "/ssd-data/jdq/eval_dbs/"
    ddfa_root = "/ssd-data/jdq/train_aug_120x120_aligned"
    result_dir = "/data0/jdq/"
    model_prefix = "/data0/jdq/model/"


if mode == "pca":
    re_type_gan = False
    g_model_path = "../propressing/pca_model.npz"
    prefix = "pca"
    tik_shape_weight = 0.0010
    tik_exp_weight = 0.0010
elif mode == "ae_p":
    re_type_gan = True
    g_model_path = "../propressing/p2_model.npz"
    prefix = "ae_p2"
    gmm_data = np.load("../propressing/p2_gmm.npy")
    tik_exp_weight = 0.02
elif mode == "ae":
    re_type_gan = True
    g_model_path = "../propressing/g_model.npz"
    prefix = "ae2"
    gmm_data = np.load("../propressing/g_gmm.npy")
    tik_exp_weight = 0.02
elif mode == "vae":
    re_type_gan = False
    g_model_path = model_prefix + "vae_model.npz"
    prefix = "vae2"
    tik_shape_weight = 0.0005
    tik_exp_weight = 0.0005
elif mode == "vae_p":
    re_type_gan = False
    g_model_path = model_prefix + "vae_p_model.npz"
    prefix = "vae_p"
    tik_shape_weight = 0.001
    tik_exp_weight = 0.001
elif mode == "vae_p2":
    re_type_gan = False
    g_model_path = model_prefix + "vae_p2_model.npz"
    prefix = "vae_p2"
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == "vae_f":
    re_type_gan = False
    g_model_path = model_prefix + "vae_f_model.npz"
    prefix = "vae_f_0.05"
    tik_shape_weight = 0.05
    tik_exp_weight = 0.05
elif mode == "vae_s":
    re_type_gan = False
    g_model_path = model_prefix + "vae_soft_model.npz"
    prefix = "vae_s_0.0004"
    tik_shape_weight = 0.0004
    tik_exp_weight = 0.0004
elif mode == "vae_s2":
    re_type_gan = False
    g_model_path = model_prefix + "vae_soft2_model.npz"
    prefix = "vae_s2_0.01"
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == "vae_sc2":
    re_type_gan = False
    g_model_path = model_prefix + "vae_sc2_model.npz"
    prefix = "vae_sc2_0.01"
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == "vae_sc4":
    re_type_gan = False
    g_model_path = model_prefix + "vae_sc4_model.npz"
    prefix = "vae_sc4_0.01"
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == "vae_sc5":
    re_type_gan = False
    g_model_path = model_prefix + "vae_sc5_model.npz"
    prefix = "vae_sc5_0.001"
    tik_shape_weight = 0.001
    tik_exp_weight = 0.001


our_model = "train.configs/"


##############################################################
prn_rst = "../propressing/prn_aflw_rst_not_align.npz"
ddfa_rst = "../propressing/3ddfa_aflw_rst.npz"

# prn_rst = '../propressing/prn_aflw_rst_not_align_47_point.npz'
# ddfa_rst = '../propressing/3ddfa_aflw_rst_47_point.npz'
##################################################################


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
gpu_ids = range(1)
regenerate_data = False
use_boundary_lms = True

img_w = 224
img_h = img_w


log_interval = 400
loss_weight_tik_shape = 0
weight_fr_center_loss = 1e-4
loss_weight_bfmz_3d = 0.1


# gmm_data = np.load('/data0/jdq/model/pq_gmm.npy')

mix_loss_weight_fr = 20.0
mix_loss_weight_2d = 1.0
mix_loss_weight_3d_vdc = 0.0
mix_loss_weight_3d_vdc_lm = 1
mix_loss_weight_fr_center_loss = 1 * 1e-3 / mix_loss_weight_fr
mix_loss_weight_adv = 0.5 * mix_loss_weight_3d_vdc_lm

loss_weight_tik_exp = 0.5  # 1e-12*mix_loss_weight_3d_vdc_lmss
loss_weight_reg_batch = 0.02

d_learning_rate = mix_loss_weight_adv / (5 * 1e5)
training_data = -1
d_steps = 1
g_steps = 1

lr_change1 = 1.0
lr_change2 = 1.5
lr_change3 = 2.0
lr_change4 = 3.0
center_lr = 5e-1 / mix_loss_weight_fr_center_loss  # 1e4

epoch = 20
lr = 0.001
batch_size = 128
lr_decay = 10
num_gpu = 1


use_v2 = True

test_img = True
test_img_path = "imgs/crop3783.jpg"

train_3ddfa = True
# best epoch = 50, loss_weight_tik = 0.05, lr decay = 5, data num = 1000
# data_anh_model = (sio.loadmat(data_anh_model_mat_path, squeeze_me=True,struct_as_record=False))['BFM']
# shape_ev = data_anh_model.shapeEV
# # exp_ev = data_anh_model.shapeEV
# exp_ev = data_anh_model.expEV

# f_model = np.load(f_model_path)
# x = f_model['x']
# gmm = pickle.load(open('../propressing/gmm.pkl','rb'))

# X,Y = gmm.sample(99)
# print(np.random.shuffle(X))
# print(exp_ev)
# exp_ev = torch.from_numpy(exp_ev.astype(np.float32))
