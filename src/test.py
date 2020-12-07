import torch
import numpy as np
import scipy.io as sio

data_root_path = "/data2/FRC/"
data_300W_3D_path = data_root_path + "crop/"
data_300W_3D_pkl_path = data_root_path + "300W_3D_crop_test_3d.pkl"
# data_ms_celeb_path = "/data2/lmd2/imgc/"
# data_ms_celeb_pkl_path = "/data2/lmd2/imgc/ms_celeb_4000.pkl"

# project_root = '/data2/lmd2/face_reconstruction/'
project_root = "./"
state_dict_path = project_root + "state_dict/params_resnet101_face_v2_115.pkl"
state_dict_train_shape_path = project_root + "state_dict/params_resnet101_anh.pkl"
state_dict_runmodel_path = (
    project_root
    + "state_dict/params_resnet101_face_v2_0.010000_50_0.010000_256_5_-1_2018-08-28-23-06.pkl"
)

data_anh_model_mat_path = project_root + "pytorch_3DMM/mat/BaselFaceModel_mod.mat"

anh_zhu_index_map_path = project_root + "pytorch_3DMM/mat/anh_zhu_index_map.npy"
outer_landmark_index_cand_path = (
    project_root + "pytorch_3DMM/mat/outer_landmark_index_cand.npy"
)


micc_obj_root = "/data/jdq/florence"
micc_image_root = "/data/jdq/output3_img"
micc_filelist = "./selected_file.txt"


data_anh_model = (
    sio.loadmat(data_anh_model_mat_path, squeeze_me=True, struct_as_record=False)
)["BFM"]
shape_ev = data_anh_model.shapeEV
exp_ev = data_anh_model.expEV

a = np.linalg.norm(np.random.randn(100000, 29) * exp_ev, axis=1)
print(a.mean(), a.std())
