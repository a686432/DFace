import torch
import numpy as np
import scipy.io as sio
import pickle 
from torchvision import  transforms


server = 'lab'
mode = 'bfm'


if server == 'zj':
    eval_fr_data_root_path = '/data/jdq/eval_dbs/'
    aflw_data_root_path = "/data/jdq/AFLW2000-3D/AFLW2000_align/"
    ddfa_root = "/data/jdq/train_aug_120x120_aligned"
    fr_data_list = '/data/jdq/faces_vgg_112x112/vgg2_file_list.pkl'
    result_dir = '/data/jdq/'
    micc_image_root = "/data/jdq/output3_img"
    micc_obj_root = '/data/jdq/florence'
    micc_filelist = './selected_file.txt'
    
elif server == 'lab':
    aflw_data_root_path = "/data/jdq/dbs/AFLW2000-3D/AFLW2000_align/"
    eval_fr_data_root_path = '/data/jdq/eval_dbs/'
    ddfa_root = "/data/jdq/train_aug_120x120_aligned"
    result_dir = '/data/jdq/'
    model_prefix = '/data/jdq/model/' 
    micc_filelist = './selected_file.txt'
    micc_image_root = "/data/jdq/output3_img"
    micc_obj_root = '/data/jdq/florence'
    

elif server == 'lab2':
    aflw_data_root_path = "/ssd-data/jdq/dbs/AFLW2000-3D/AFLW2000_align/"
    eval_fr_data_root_path = '/ssd-data/jdq/eval_dbs/'
    ddfa_root = "/ssd-data/jdq/train_aug_120x120_aligned"
    result_dir = '/data0/jdq/'
    model_prefix = '/data0/jdq/model/' 
    micc_filelist = './selected_file.txt'
    micc_image_root = "/ssd-data/jdq/output3_img"
    micc_obj_root = '/ssd-data/jdq/florence'



if mode=='pca':
    re_type_gan = False
    g_model_path = '../propressing/pca_model.npz'
    prefix = 'pca_No4_2_16'
    tik_shape_weight = 0.01
    tik_exp_weight = 0.05
if mode=='bfm':
    re_type_gan = False
    g_model_path = '../propressing/pca_model.npz'
    prefix = 'bfm_No6_9'
    tik_shape_weight = 10
    tik_exp_weight = 0.001
    checkpoint_warm_pixel = '/data/jdq/model_s/B3_4'
    checkpoint_warm3d = '/data/jdq/model_s/B3_4'
elif mode == "Flame":
    re_type_gan = False
    g_model_path = '../propressing/pca_model.npz'
    tik_shape_weight = 0.005
    tik_exp_weight = 0.005
    prefix = 'Flame'
elif mode=="linear":
    re_type_gan = False
    g_model_path = '../propressing/linear_model.npz'
    prefix = 'linear'
    tik_shape_weight = 0.020
    tik_exp_weight = 0.005
elif mode == "Clinear":
    re_type_gan = False
    g_model_path = '../propressing/Clinear_model.npz'
    prefix = 'Clinear_N2_7'
    tik_shape_weight = 2
    tik_exp_weight = 0.005
    checkpoint_warm_pixel = '/data/jdq/model_s/C2_0_1'
    checkpoint_warm3d = '/data/jdq/model_s/C2_0_1'
elif mode == 'VC_nonlinear':
    re_type_gan = False
    mlc_path = '../propressing/VC_nonlinear_mlc.pkl'
    g_model_path = '../propressing/VC_nonlinear_model.npz'
    prefix = 'VC_nonlinear'
    tik_shape_weight = 0.02
    tik_exp_weight = 0.02
elif mode == 'C_nonlinear':
    re_type_gan = False
    mlc_path = '../propressing/C_nonlinear_mlc.pkl'
    g_model_path = '../propressing/C_nonlinear_model.npz'
    prefix = 'C_nonlinear_N0_1'
    checkpoint_warm_pixel = '/data/jdq/model_s/N0_0'
    checkpoint_warm3d = '/data/jdq/model_s/N0_0'
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == 'vae_p':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_p_model.npz'
    prefix = 'vae_p'
    tik_shape_weight = 0.001
    tik_exp_weight = 0.001
elif mode == 'vae_p2':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_p2_model.npz'
    prefix = 'vae_p2'
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == 'vae_f':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_f_model.npz'
    prefix = 'vae_f_0.05'
    tik_shape_weight = 0.05
    tik_exp_weight = 0.05
elif mode == 'vae_s':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_soft_model.npz'
    prefix = 'vae_s_0.0004'
    tik_shape_weight =0.0004
    tik_exp_weight = 0.0004
elif mode == 'vae_s2':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_soft2_model.npz'
    prefix = 'vae_s2_0.01'
    tik_shape_weight =0.01
    tik_exp_weight = 0.01
elif mode == 'vae_sc2':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_sc2_model.npz'
    prefix = 'vae_sc2_0.01'
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == 'vae_sc4':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_sc4_model.npz'
    prefix = 'vae_sc4_0.01'
    tik_shape_weight = 0.01
    tik_exp_weight = 0.01
elif mode == 'vae_sc5':
    re_type_gan = False
    g_model_path = model_prefix + 'vae_sc5_model.npz'
    prefix = 'vae_sc5_0.001'
    tik_shape_weight = 0.001
    tik_exp_weight = 0.001


our_model = 'train.configs/'


##############################################################
prn_rst = '../propressing/prn_aflw_rst_not_align.npz'
ddfa_rst = '../propressing/3ddfa_aflw_rst.npz'

# prn_rst = '../propressing/prn_aflw_rst_not_align_47_point.npz'
# ddfa_rst = '../propressing/3ddfa_aflw_rst_47_point.npz'
##################################################################
use_ConvTex = True 

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
gpu_ids=range(1)
regenerate_data = False
use_boundary_lms = True 

img_w = 96
img_h = 112
	
	
log_interval = 400
#loss_weight_tik_shape = 0
weight_fr_center_loss = 1e-4

use_pixel = True

weight_pixel_loss = 1
weight_pixel_loss_raw = 1
weight_albedo_reg = 5e3 * weight_pixel_loss
weight_albedo_smooth = 20 * weight_pixel_loss
weight_albedo_sharp = 20 * weight_pixel_loss
weight_res_albedo_loss = 1e2 * weight_pixel_loss

weight_albedo_flip = 0
proxy_loss_weight = 0.1
weight_loss_perc_im = 0.25 * weight_pixel_loss
weight_loss_perc_im_raw = 0.25 * weight_pixel_loss
filp_threshold = 0
weight_res_shape_loss = 400
var_weight_shape = 1
weight_res_landmark = 1
weight_shape_smooth_loss = 1e6
# 2e6


soft_start_pixel_loss = True
soft_start_perceptual_loss = True
start_from_warm3d = True
start_from_warmpixel = True

if start_from_warm3d:
    soft_start_pixel_loss = False

if start_from_warmpixel:
    soft_start_perceptual_loss = False
    start_from_warm3d = False


use_flip = True
use_perceptual_loss = True
use_res_shape = True
use_proxy = False
use_edge_lm = False
use_shape_smooth = True
use_face_recognition_constraint = True
use_albedo_res = True
use_confidence_map =  True
use_mix_data = True
use_center_loss = True
use_weighted_center_loss = True and use_center_loss
if not start_from_warmpixel:
    use_face_recognition_constraint = False
    use_mix_data = False
    use_center_loss = False
    use_res_shape = False
    use_shape_smooth = False



evalation_path = '/data/jdq/model_s/B6_3.pkl'

if mode == 'bfm':
    use_res_shape = False
    use_shape_smooth = False

if use_pixel == False:
    weight_pixel_loss = 0
    weight_pixel_loss_raw = 0

weight_edge_lm =0
weight_edge_lm_raw = 0

loss_weight_bfmz_3d = 0.1


#gmm_data = np.load('/data0/jdq/model/pq_gmm.npy')

mix_loss_weight_fr = 20.0
mix_loss_weight_2d = 1.0
mix_loss_weight_3d_vdc = 0.0
mix_loss_weight_3d_vdc_lm = 1

mix_loss_weight_fr_center_loss = 1e-5*mix_loss_weight_fr
mix_loss_weight_adv =0.5*mix_loss_weight_3d_vdc_lm

loss_weight_tik_exp = 0.5 #1e-12*mix_loss_weight_3d_vdc_lmss
loss_weight_reg_batch = 0.02

d_learning_rate =  mix_loss_weight_adv/(5*1e5)
training_data = -1
d_steps = 1
g_steps = 1
 
lr_change1 = 2.0
lr_change2 = 3.0
lr_change3 = 4.0
lr_change4 = 5.0
center_lr = 2
#1e5*mix_loss_weight_fr_center_loss #1e4 

epoch = 20
lr = 0.001
batch_size = 128
lr_decay = 10
num_gpu = 1




use_v2 = True

test_img = True
test_img_path = 'imgs/crop3783.jpg'

train_3ddfa = True


transform_raw = transforms.Compose([
            transforms.CenterCrop((112,96)),    
            transforms.ToTensor()
        ])

# index = None

def process_uv(uv_coords, uv_h = 112, uv_w = 112):
    uv_coords[:,0] = uv_coords[:,0]*(uv_h - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

C = sio.loadmat('../propressing/Model_Tex.mat')
colors = C['tex'].T; triangles = C['tri'].T-1; uv_coords=C['uv']
uv_h = uv_w = 128


##################################FLame config#####################################


flame_model_path='../propressing/generic_model.pkl'
static_landmark_embedding_path = '../propressing/flame_static_embedding.pkl'
dynamic_landmark_embedding_path = '../propressing/flame_dynamic_embedding.npy'




shape_params=199
expression_params=29
pose_params=6
# Training hyper-parameters
use_face_contour = True
use_3D_translation= False
optimize_eyeballpose=False
optimize_neckpose = False













##################################################################################
# uv_coords = process_uv(uv_coords, uv_h, uv_w)

#best epoch = 50, loss_weight_tik = 0.05, lr decay = 5, data num = 1000
# data_anh_model = (sio.loadmat(data_anh_model_mat_path, squeeze_me=True,struct_as_record=False))['BFM']
# shape_ev = data_anh_model.shapeEV
# # exp_ev = data_anh_model.shapeEV
# exp_ev = data_anh_model.expEV

# f_model = np.load(f_model_path)
# x = f_model['x']
#gmm = pickle.load(open('../propressing/gmm.pkl','rb'))

#X,Y = gmm.sample(99)
#print(np.random.shuffle(X))
#print(exp_ev)
#exp_ev = torch.from_numpy(exp_ev.astype(np.float32))
