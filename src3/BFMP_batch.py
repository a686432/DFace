import numpy as np
import scipy.io    as sio
import torch
from torch.autograd import Variable
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt, pi
from utils import *
import config
from misc import *
import torch.nn as nn
import time
from collections import OrderedDict
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    OpenGLOrthographicCameras,
    SfMOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturedSoftPhongShader,
    HardPhongShader,
    interpolate_face_attributes,
    hard_rgb_blend,
    softmax_rgb_blend,
    BlendParams,
    rasterize_meshes,
    get_world_to_view_transform,
    interpolate_texture_map
)

class BasicBlock(nn.Module):
    
    def __init__(self,planes,expansion=1):
        super(BasicBlock,self).__init__()
        m = OrderedDict()
        m['fc1'] = nn.Linear(planes, planes*expansion)
        m['bn1'] = nn.BatchNorm1d(planes*expansion)
        #m['relu1'] = nn.PReLU(planes*expansion)
        m['tanh'] = nn.Tanh()
        # m['fc2'] = nn.Linear(planes*expansion, planes*expansion)
        # m['relu2'] = nn.PReLU(planes*expansion)
        m['fc3'] = nn.Linear(planes*expansion, 53215*3)
        # m['relu2'] = nn.PReLU(planes)
        self.planes = planes
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        out = self.group1(x)
        return out


class BFMG_batch(nn.Module):
    def __init__(self,dim=199):
        super(BFMG_batch, self).__init__()
        """    
        A batch version for BFMF      
        A 3DMM class similar to the BFM proposed by us
        A face model can be presented as:
            face_mesh  = mu + w * Shape_para + w_expression * Exp_para
            where mu is the mean face, w is the shape basis, w_expression is the expression basis
        
        According to 3DDFA, the shape basis and expression basis come from BFM and FaceWarehouse respectively
        Atttibutes:
            mu: A 1-D array of the mean face mesh vertex, 140970 * 1, [x0, y0, z0, x1, y1, z1, ..., xn-1, yn-1, zn-1]
            index:    A 2-D array of the face mesh triangle index, 93323 * 3, [[i00, i10, i20], [i01, i11, i21], ..., [i0n-1, i1n-1, i2n-1]]
            w: A 1-D array of the shape_base, 140970 * 199
            w_expression:  1-D array of the expression base, 140970 * 29
            inner_landmark: A 1-d array of the 68 inner landmark indices
            outer_landmark: : A 1-d array of the 68 outter landmark indices
        """
        self.dim = dim
        sym_map = torch.Tensor(np.load("../propressing/sym_flip.npy")).long()
        shape_adj_raw =  np.load('../propressing/shape_smooth.npz')
        adj_matrix = torch.Tensor(shape_adj_raw['adj_matrix']).long()
        adj_mask = torch.Tensor(shape_adj_raw['adj_mask'].astype('int'))
        # if config.use_res_shape:
        #     self.mlc = BasicBlock(199)
        bfm=np.load("../propressing/bfmz.npz")
        res_var =  torch.Tensor(np.load("../propressing/res_var.npy"))
        w_expression_tensor = torch.Tensor(bfm['w_expression']).reshape(-1,3,29).reshape(-1,29).t()/1000
        mu_expression_tensor =torch.Tensor(bfm['mu_expression']).reshape(-1,3).reshape(-1)/1000
        #########################################################################
        g_model = np.load(config.g_model_path)
        outer_landmark_tensor = torch.Tensor(g_model['landmarks'].astype(int)).long()
        #print(bfm['mu_shape'].shape)
        mu_shape = bfm['mu_shape'].reshape(-1)/1000
        mu_shape_tensor = torch.from_numpy(mu_shape.astype(np.float32))
        w_shape = bfm['w_shape'].reshape(-1,3,199).reshape(-1,199).T
         # = torch.Tensor(bfm['mu_shape'].astype(float)/1000)
        w_tensor = torch.from_numpy(w_shape.astype(np.float32))
        index =  bfm['index']
        index = torch.from_numpy(index.astype('int')).long()
        #mean_albedo = torch.Tensor(np.load('../propressing/uv_albedo_map.npy'))
        # mask_albedo = torch.Tensor(np.load('../propressing/mask_uv.npy'))
        # self.mask_albedo_npixels = mask_albedo.sum()

        ms = sio.loadmat('../propressing/Model_Shape.mat')
        shape_ev_tensor = torch.from_numpy(ms['sigma'].reshape(-1).astype(np.float32)/1000)
        # print(shape_ev_tensor.shape)

        mu_tensor = mu_shape_tensor + mu_expression_tensor
        #mu_tensor = (mu_tensor.reshape(-1,3) -  mu_tensor.reshape(-1,3).mean(dim=0).reshape(1,3)).reshape(-1)
        exp_ev_tensor = torch.from_numpy(bfm['exp_ev'])


        #R, T = look_at_view_transform(200, 0, 0) 
 
        
        
        self.aflw_21_landmarks = np.array([37535, 4645, 38561, 38935, 11592, 39955, 2087, 4409, 5958, 10326, 11872, 14194, 19841, 6384, 8190, 9997, 33862, 5391, 8224, 11048, 43504])
    
        mean_para_exp = torch.zeros(29)
        mean_para_shape = torch.zeros(199)
        var_para_exp = torch.ones(29)
        var_para_shape = torch.ones(199)

        # for i in range((self.index.shape)[0]):
        #     self.index[i,:] = self.index[i,:] - 1
        
        '''
            register_buffer:
            3DMM information, 
        '''
        self.register_buffer('mu',mu_tensor)
        self.register_buffer('mu_shape',mu_shape_tensor)
        self.register_buffer('mu_exp',mu_expression_tensor)
        self.register_buffer('w_shape',w_tensor)
        self.register_buffer('w_exp', w_expression_tensor)
        self.register_buffer('e_shape',shape_ev_tensor)
        self.register_buffer('e_exp', exp_ev_tensor)
        self.register_buffer('tris', index-1)
        self.register_buffer('outer_landmark',outer_landmark_tensor)
        self.register_buffer('mp_exp',mean_para_exp)
        self.register_buffer('mp_shape',mean_para_shape)
        self.register_buffer('vp_exp',var_para_exp)
        self.register_buffer('vp_shape',var_para_shape)
        self.register_buffer('sym_map',sym_map)
        self.register_buffer('adj_matrix',adj_matrix)
        self.register_buffer('adj_mask',adj_mask)
        self.register_buffer('res_var',res_var)
        # self.register_buffer('H',torch.Tensor(H))
        # self.register_buffer('mean_albedo',torch.Tensor(mean_albedo))
        # self.register_buffer('mask_albedo',mask_albedo)

        #mean_albedo


        #print('mu:',self.mu)
        # self.aflw_21_landmarks


        # self.w_tensor = self.register_buffer('mu_shape',self.w_tensor)
        # self.


            
    def forward(self,shape_para, exp_para, camera_para=None,shape_flip=False):
        """
        Input:
            shape_para: batch * 99
            exp_rapa: batch * 29
            camera_para:batch * (4 + 2 + 1) R-4, T-2, S-1
        """
        batch_size = shape_para.shape[0]
        R = Q2R_batch(camera_para[:, 0:4])
        #print(self.mu)
        # if config.use_res_shape:
        #     res_shape = self.mlc(shape_para[:,:self.dim])
        #     face_shape_res = self.mu +  shape_para[:,:self.dim] @ self.w_shape + res_shape
        #     if shape_flip:
        #         face_shape_res = face_shape_res.reshape(batch_size, -1, 3)
        #         face_shape_res[:,:,0] = - face_shape_res[:,:,0]
        #         face_shape_res[:,self.sym_map[:,0],0] = face_shape_res[:,self.sym_map[:,1],0]
        #         face_shape_res = face_shape_res.reshape(batch_size,-1)
            
        #     face_vertex_tensor_res = face_shape_res + exp_para @ self.w_exp
        #     rotated_res = torch.bmm(R, face_vertex_tensor_res.reshape(batch_size, -1, 3).transpose(2,1)).reshape(batch_size, -1) 
        #     scaled_res = (-torch.abs(camera_para[:, 4].unsqueeze(1)) * rotated_res).reshape(batch_size, 3, -1)# 
        #     scaled_res[:, :2, :] = scaled_res[:, :2, :] + camera_para[:, 5:7].view(batch_size, 2, 1)

            # return face_vertex_tensor


        
        face_vertex_tensor = self.mu +  shape_para[:,:self.dim] @ self.w_shape  + exp_para @ self.w_exp
        #R = Q2R_batch(camera_para[:, 0:4])
        rotated = torch.bmm(R, face_vertex_tensor.reshape(batch_size, -1, 3).transpose(2,1)).reshape(batch_size, -1) 
        scaled = (-torch.abs(camera_para[:, 4].unsqueeze(1)) * rotated).reshape(batch_size, 3, -1)# 
        scaled[:, :2, :] = scaled[:, :2, :] + camera_para[:, 5:7].view(batch_size, 2, 1)
        if not config.use_res_shape:
            scaled_res = scaled



       # face_vertex_tensor = self.mu + (torch.mm(self.w_shape, shape_para.transpose(1, 0)) + torch.mm(self.w_exp, exp_para.transpose(1, 0))).transpose(1, 0)  
    #     print("F: ",face_vertex_tensor)
    #     print("S: ", shape_para @ self.w_shape)
    #     print("E: ",   exp_para @ self.w_exp)
    #     print("EP: ",exp_para )
    #     print("EW: ", self.w_exp)
    #    # print("SW: ", self.w_shape.t() @ self.w_shape)
    #     print("SW: ", self.w_shape @ self.w_shape.t())
        
        #print(face_vertex_tensor)

        # R = Q2R_batch(camera_para[:, 0:4])
        # rotated = torch.bmm(R, face_vertex_tensor.reshape(batch_size, -1, 3).transpose(2,1)).reshape(batch_size, -1) 
        # scaled = (-torch.abs(camera_para[:, 4].unsqueeze(1)) * rotated).reshape(batch_size, 3, -1)# 
        # scaled[:, :2, :] = scaled[:, :2, :] + camera_para[:, 5:7].view(batch_size, 2, 1)
        # face_vertex_tensor=0
        # rotated=0
        # scaled=0
        return scaled, scaled_res


    def get_shape(self,shape_para):
        if config.use_res_shape:
            face_vertex_tensor =  self.mu + shape_para[:,:self.dim] @ self.w_shape + self.mlc(shape_para[:,:self.dim]) 
        else:
            face_vertex_tensor = self.mu +  shape_para[:,:self.dim] @ self.w_shape  
        return face_vertex_tensor

    def shape_smooth_loss(self,shape_para):
        batch_size = shape_para.shape[0]
        shape_res = self.mlc(shape_para[:,:self.dim]).reshape(batch_size,-1,3)
        return ((shape_res - (shape_res[:,self.adj_matrix,:]*self.adj_mask.unsqueeze(0).unsqueeze(-1)).sum(2)/self.adj_mask.sum(1).reshape(1,-1,1))**2).mean()

    def shape_smooth_loss_m(self,shape_para):
        batch_size = shape_para.shape[0]
        shape_res = (self.mlc(shape_para[:,:self.dim])+shape_para[:,:self.dim] @ self.w_shape).reshape(batch_size,-1,3)
        return ((shape_res - (shape_res[:,self.adj_matrix,:]*self.adj_mask.unsqueeze(0).unsqueeze(-1)).sum(2)/self.adj_mask.sum(1).reshape(1,-1,1))**2).mean()

    def get_resloss(self,shape_para):
        log_var = ((shape_para[:,:self.dim]/self.e_shape).var(dim=0)+1e-8).log()
        mean = shape_para[:,:self.dim].mean()
        KLD = -0.5 * torch.sum((1 + log_var  - log_var.exp())*config.var_weight_shape-mean.pow(2))/shape_para.shape[0]
        #res = self.mlc(shape_para[:,self.dim:]) + shape_para[:,:self.dim] @ self.w_shape
        # mean_loss = (res.mean(dim=0)**2).mean()
        # loss = (((res**2).mean(dim=0)-self.res_var)**2).mean() + mean_loss
        return KLD

    def get_landmark_68(self,face_pixels):
        #face_vertex_tensor = self.mu +  shape_para[:,:self.dim] @ self.w_shape  + exp_para @ self.w_exp
        return face_pixels[ :, :2, self.outer_landmark]

    def get_3d_landmark_68(self, face_vertex):
        return face_vertex[ :, :, self.outer_landmark]
    
    def get_edge_landmark(self,landmarks):
        batch_size = landmarks.shape[0]  # batch_size, 2, 68
        landmarks =landmarks.transpose(2,1).unsqueeze(1).expand(batch_size,68,68,2)  # batch_size, 68, 68, 2
        edge_distance = (((landmarks-landmarks.transpose(2,1))**2).sum(dim=-1)+1e-12).sqrt() # batch_size, 68, 68
        return edge_distance

    def get_aflw_21_landmark(self,face_pixels):
        return face_pixels[ :, :, self.aflw_21_landmarks]

    def get_outer_moved_landmark(self, face_vertex, camera_para):
        outer_landmark_moved_idx =self._move_boundary_landmarks(face_vertex, camera_para)
        return face_pixels[ :, :, outer_landmark_moved_idx]
       

        '''
        # if self.camera_Para is not None:
        #     outer_landmark_moved = self.outer_landmark_tensor.repeat(batch_size, 1)
        #     #outer_landmark_moved = np.tile(self.outer_landmark, (batch_size, 1)) 
        #     outer_landmark_moved = self.__move_boundary_landmarks(face_vertex_tensor,camera_para,batch_size,outer_landmark_moved)
        #     #self.outerlandmark_vertex_tensor = (self.face_vertex_tensor.reshape(self.batch_size, -1, 3))[:, self.outer_landmark_moved].transpose(1, 2)
        #     #outer_landmark_moved = np.repeat(outer_landmark_moved, 3, 1)
        #     outer_landmark_moved = outer_landmark_moved.repeat(batch_size,3).reshape(batch_size,3,-1).transpose(1,2).reshape(batch_size,-1)
        #     outer_landmark_moved[:, 0::3] = outer_landmark_moved[:, 0::3] * 3
        #     outer_landmark_moved[:, 1::3] = outer_landmark_moved[:, 1::3] * 3 + 1
        #     outer_landmark_moved[:, 2::3] = outer_landmark_moved[:, 2::3] * 3 + 2
        #     #outer_landmark_moved_tensor = torch.tensor(outer_landmark_moved)

        #     if config.use_cuda:
        #         outer_landmark_moved = outer_landmark_moved.cuda()

        #     outerlandmark_vertex_tensor = torch.gather(face_vertex_tensor, 1, outer_landmark_moved).reshape(batch_size, -1, 3).transpose(1, 2)
            

        #     #aflw_21_landmark_idx = np.tile(self.aflw_21_landmarks, (batch_size, 1)) 
        #     aflw_21_landmark_idx = self.aflw_21_landmarks_tensor.repeat(batch_size,1)
        #     aflw_21_landmark_idx = aflw_21_landmark_idx.repeat(batch_size,3).reshape(batch_size,3,-1).transpose(1,2).reshape(batch_size,-1)

        #     #aflw_21_landmark_idx = np.repeat(aflw_21_landmark_idx, 3, 1)
        #     aflw_21_landmark_idx[:, 0::3] = aflw_21_landmark_idx[:, 0::3] * 3
        #     aflw_21_landmark_idx[:, 1::3] = aflw_21_landmark_idx[:, 1::3] * 3 + 1
        #     aflw_21_landmark_idx[:, 2::3] = aflw_21_landmark_idx[:, 2::3] * 3 + 2
        #     #aflw_21_landmark_idx = torch.tensor(aflw_21_landmark_idx)
        #     if config.use_cuda:
        #         aflw_21_landmark_idx = aflw_21_landmark_idx.cuda()

        #     aflw_21_landmark_vertex_tensor = torch.gather(self.face_vertex_tensor, 1, aflw_21_landmark_idx).reshape(batch_size, -1, 3).transpose(1, 2)
            
        #     outer_landmark_not_moved_tensor = face_vertex_tensor.reshape(self.batch_size,-1, 3)[:, self.outer_landmark_tensor] #(batch_size, 68, 3)

        #     if config.use_cuda:
        #         outerlandmark_vertex_tensor = outerlandmark_vertex_tensor.cuda()
        #         aflw_21_landmark_vertex_tensor = aflw_21_landmark_vertex_tensor.cuda()
        '''   


    def __move_boundary_landmarks(self,face_vertex_tensor, Camera_Para):
        batch_size = face_vertex_tensor.shape[0]
        outer_landmark_moved = self.outer_landmark_tensor.repeat(batch_size, 1)
        phi, theta , psi = Q2E_batch(Camera_Para[:, 0:4]) # Q -> E
        psi -= psi # set psi to 0
        R = Q2R_batch(E2Q_batch(phi, theta, psi)) # E -> Q -> R R shape = (batch_size, 3, 3)
        rotated_vertex = torch.bmm(R, face_vertex_tensor.reshape(batch_size, -1, 3).transpose(1, 2)).transpose(1, 2)
        
        lms = np.append(np.array(range(0, 6)),np.array(range(11, 17))) 
        mask = np.array([0, 0, 0, 0, 0, 0, -5, -5, -5, -5, -5, -5])
        
        landmark_vertex_cand = rotated_vertex[:, self.outer_landmark_index_cand[lms + mask]]
        
        selected_1 = torch.min(landmark_vertex_cand[:, :6], dim = 2)[1]
        selected_2 = torch.max(landmark_vertex_cand[:, 6:12], dim = 2)[1]
        selected = torch.cat([selected_1, selected_2], dim = 1)

        
        """
        landmark_vertex_cand = rotated_vertex[:, BFMA_batch.outer_landmark_index_cand[lms + mask]]
        
        landmark_vertex_cand[:, 6:12] = -landmark_vertex_cand[:, 6:12]
        selected = torch.min(landmark_vertex_cand, dim = 2)[1]
        """
        # if config.use_cuda:
        #     selected = selected.data.cpu().numpy()
        # else:
        #     selected = selected.data.numpy()
  
        outer_landmark_moved[:, lms] = self.outer_landmark_index_cand[lms + mask, selected[:, :, 0]]
        return outer_landmark_moved

        
    def get_tikhonov_regularization(self, Shape_Para, Exp_Para):
        """
        calculate the 3DMM parameters' tikhonov regularization loss
        Output:
            regularization_loss: shape = (batch_size)
        
        """
        
        a = torch.div(Shape_Para, self.shape_ev_tensor)
        
        b = torch.div(Exp_Para, self.exp_ev_tensor)
        return config.tik_shape_weight * torch.sum(a ** 2, dim = 1) + torch.sum(b ** 2, dim = 1)

    def get_batch_distribution_regular_shape(self, shape_para):
        """
        """

        a = torch.div(shape_para[:,:self.dim], self.e_shape)
         
        a = torch.sum(a**2,dim = 1).mean()

        # l_u_shape = torch.sum((torch.mean(a, dim = 0) - self.mp_shape) ** 2)
        # l_sigma_shape = torch.sum((torch.var(a, dim = 0) - self.vp_shape) ** 2)


        return a 

    def get_batch_distribution_regular_exp(self, exp_para):
        """
        """

        b = torch.div(exp_para, self.e_exp)

        
        l_u_exp = torch.sum((torch.mean(b, dim = 0) - self.mp_exp) ** 2)
        l_sigma_exp = torch.sum((torch.var(b, dim = 0) -  self.vp_exp) ** 2)

        return l_u_exp + l_sigma_exp

    def get_tikhonov_regularization_exp(self, exp_para):
        """
        """

        b = torch.div(exp_para, self.e_exp)
         
        b = torch.sum(b**2,dim = 1).mean()
        # l_u_exp = torch.sum((torch.mean(b, dim = 0) - self.mp_exp) ** 2)
        # l_sigma_exp = torch.sum((torch.var(b, dim = 0) -  self.vp_exp) ** 2)

        return b

    def get_2d_landmark_loss(self, gt_landmark):
        """
        calculate 2D-landmark loss
        mean( sum( delta_x ** 2 + delta_y ** 2) )
        Input:
            gt_landmark: 136 * 1
    
        Output:
            l2 loss: shape = ()
        """
        
        self.projected = self.project_outer_landmark() # get the projected 68 landmarks, shape = (batch_size, 2, 68)
        self.projected = self.projected.transpose(1, 2).reshape(self.batch_size, -1) # flatten it
        #print(self.projected)
        #print(gt_landmark)
        diff = self.projected - gt_landmark
        
        #print(diff[0])
        
        rst = torch.mean(diff ** 2, dim=1)
        #print(rst.shape)
        return rst
        

    
    def get_3D_loss_bfmz(self, gt_vertex_bfmz):
        """
        calculate VDC based on Zhu's face model vertices
        Input:
            gt_vertex_zhu_model: 159645 * 1
        
        Output:
            3d loss: shape = ()
        """
        
        trimed_gt = gt_vertex_bfmz[:, BFMA_batch.anh_zhu_index_map]
        diff = self.face_vertex_tensor - trimed_gt
        return torch.mean(diff ** 2)
        
        
    def project_outer_landmark(self, use_aflw = False, use_not_moved=False):
        """
        
        return :
            2d points array of the projected landmark, shape: 2, 68
        
        """
        
        
        R = Q2R_batch(self.Camera_Para[:, 0:4])
        
        #construct project matrix
        P = np.array([[1, 0, 0],[0, 1, 0]]).astype(np.float32)
        P = np.tile(P, (self.batch_size, 1)).reshape(self.batch_size, 2, 3)
        P = torch.from_numpy(P)
        
        # x' = s * R * x + T
        if config.use_cuda:
            P = P.cuda()
        if use_not_moved:
            rotated = torch.bmm(R, self.outer_landmark_not_moved_tensor.transpose(2,1)).reshape(self.batch_size, -1)
        else:
            rotated = torch.bmm(R, self.outerlandmark_vertex_tensor if not use_aflw else self.aflw_21_landmark_vertex_tensor).reshape(self.batch_size, -1)
        #print 'camera scale = ', torch.abs(self.Camera_Para[:, 4]).unsqueeze(1)
        scaled = (-torch.abs(self.Camera_Para[:, 4].unsqueeze(1)) * rotated).reshape(self.batch_size, 3, -1)
        projected = torch.bmm(P , scaled) 
        translated = projected + self.Camera_Para[:, 5:7].view(self.batch_size, 2, 1)

        return translated

    #def project_68_landmark(self,)
    
    def transform_face(self):
        '''
        transform the face using camera     
        '''
        R = Q2R_batch(self.Camera_Para[:, 0:4])
        rotated = torch.bmm(R, self.face_vertex_tensor.reshape(self.batch_size, -1, 3).transpose(2,1)).reshape(self.batch_size, -1) # batchszize * N
        scaled = (-torch.abs(self.Camera_Para[:, 4].unsqueeze(1)) * rotated).reshape(self.batch_size, 3, -1)# batchszize * 3 * N
        
        #print(rotated[:, 0, :].shape)
        #print(self.Camera_Para[:, 5].view(self.batch_size, 1, 1).shape)
        #rotated[:, 0, :] + self.Camera_Para[:, 5].view(self.batch_size, 1, 1)
        #rotated[:, 1, :] + self.Camera_Para[:, 6].view(self.batch_size, 1, 1)
        
        scaled[:, :2, :] = scaled[:, :2, :] + self.Camera_Para[:, 5:7].view(self.batch_size, 2, 1)
        
        return scaled




    
if __name__ == '__main__':
    BFMF_batch()
    