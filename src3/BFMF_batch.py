import numpy as np
import scipy.io    as sio
import torch
from torch.autograd import Variable
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt, pi
#from utils import *
import config
from misc import *
import torch.nn as nn

class BFMF_batch(nn.Module):
    def __init__(self):
        super(BFMF_batch, self).__init__()
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


        bfm=np.lopad("../propressing/bfmz.npz")
        w_shape = bfm['w_shape'][:,0:99]
        mu_shape = bfm['mu_shape'].reshape(-1)
        f = open(imagefile,'r')
        w_expression = bfm['w_expression'][:,0:29]
        mu_expression =torch.bfm['mu_expression'].reshape(-1)
        print(w_shape.shape)
        # for param in params:
        #     print(param)
        #     face_shape = (mu_shape + w_shape @ param).reshape(-1,3)/1000
        #     filepath = f.readline().replace('\n',"")
        #     identify = int(filepath.split('/')[-2])
        #     filename = filepath.split('/')[-1][:-3]+"obj"
        #     outdir = os.path.join(out_root,str(identify))
        #     if not os.path.exists(outdir):
        #         os.mkdir(outdir)
        #     filename = os.path.join(outdir,filename)
        #     print(face_shape)
        #     self.write_obj(filename,face_shape)
        
        #########################################################################
        f_model = np.load(config.f_model_path)
        x = f_model['x']
        mu = f_model['mu']
        w_shape = f_model['w_shape']
        index = f_model['faces']




        exit()


        #########################################################################

        #1. construct  the mean face and the base from local .mat files
        #1.1 load the data from .mat files

        # data_anh_model = (sio.loadmat(config.data_anh_model_mat_path, squeeze_me=True,struct_as_record=False))['BFM']
        #1.2 load the data from the dicts

        # mu_shape = data_anh_model.shapeMU
        # gl coor-sys --> cv coor-sys
        # (x, y, z) --> (x, -y, -z)
        # mu_shape[1::3] = -mu_shape[1::3]
        # mu_shape[2::3] = -mu_shape[2::3]
        # self.mu_shape_tensor = torch.from_numpy(mu_shape.astype(np.float32)) # to calculate loss and bp

        w = data_anh_model.shapePC
        w[1::3, :] = -w[1::3, :]
        w[2::3, :] = -w[2::3, :]
        self.w_tensor = torch.from_numpy(w.astype(np.float32)) # to calculate loss and bp , shape:(N, 99)
        shape_ev = data_anh_model.shapeEV
        self.shape_ev_tensor = torch.from_numpy(shape_ev.astype(np.float32)) #shape: (99,)
        
        index = data_anh_model.faces
        self.index = torch.from_numpy(index.astype('int')-1)
        self.inner_landmark = data_anh_model.innerLandmarkIndex.astype('int') - 1
        self.outer_landmark = data_anh_model.outerLandmarkIndex.astype('int') - 1
        self.outer_landmark_tensor = torch.from_numpy(self.outer_landmark).long()
        #w_expression is the exp base
        w_expression = data_anh_model.expPC
        w_expression[1::3, :] = -w_expression[1::3, :]
        w_expression[2::3, :] = -w_expression[2::3, :]
        self.w_expression_tensor = torch.from_numpy(w_expression.astype(np.float32))# to calculate loss and bp

        mu_expression = data_anh_model.expMU
        mu_expression[1::3] = -mu_expression[1::3]
        mu_expression[2::3] = -mu_expression[2::3]
        self.mu_expression_tensor =torch.from_numpy(mu_expression.astype(np.float32))# to calculate loss and bp
        
        exp_ev = data_anh_model.expEV
        self.exp_ev_tensor = torch.from_numpy(exp_ev.astype(np.float32))
        mu = mu_shape + mu_expression
        #mu = mu.reshape(-1, 1)
        self.mu_tensor = torch.from_numpy(mu.astype(np.float32))# to calculate loss and bp
        
        
        self.mu_vertex = mu.reshape(-1, 3)
        
        self.anh_zhu_index_map = np.load(config.anh_zhu_index_map_path)
        
        self.outer_landmark_index_cand = np.load(config.outer_landmark_index_cand_path)
        
        
        self.aflw_21_landmarks = np.array([37535, 4645, 38561, 38935, 11592, 39955, 2087, 4409, 5958, 10326, 11872, 14194, 19841, 6384, 8190, 9997, 33862, 5391, 8224, 11048, 43504])
    
        mean_para_exp = torch.zeros(29)
        mean_para_shape = torch.zeros(99)
        var_para_exp = torch.ones(29)
        var_para_shape = torch.ones(99)

        # for i in range((self.index.shape)[0]):
        #     self.index[i,:] = self.index[i,:] - 1
        
        '''
            register_buffer:
            3DMM information, 
        '''
        self.register_buffer('mu',self.mu_tensor)
        self.register_buffer('mu_shape',self.mu_shape_tensor)
        self.register_buffer('mu_exp',self.mu_expression_tensor)
        self.register_buffer('w_shape',self.w_tensor)
        self.register_buffer('w_exp', self.w_expression_tensor)
        self.register_buffer('e_shape', self.shape_ev_tensor)
        self.register_buffer('e_exp', self.exp_ev_tensor)
        self.register_buffer('tris', self.index)
        self.register_buffer('out',self.outer_landmark_tensor)
        self.register_buffer('mp_exp',mean_para_exp)
        self.register_buffer('mp_shape',mean_para_shape)
        self.register_buffer('vp_exp',var_para_exp)
        self.register_buffer('vp_shape',var_para_shape)


        #print('mu:',self.mu)
        # self.aflw_21_landmarks


        # self.w_tensor = self.register_buffer('mu_shape',self.w_tensor)
        # self.


            
    def forward(self,shape_para, exp_para, camera_para=None):
        """
        Input:
            shape_para: batch * 99
            exp_rapa: batch * 29
            camera_para:batch * (4 + 2 + 1) R-4, T-2, S-1
        """
        batch_size = shape_para.shape[0]
        #print(self.mu)
        face_vertex_tensor = self.mu + (torch.mm(self.w_shape, shape_para.transpose(1, 0)) + torch.mm(self.w_exp, exp_para.transpose(1, 0))).transpose(1, 0)  
        
        
        #print(face_vertex_tensor)

        R = Q2R_batch(camera_para[:, 0:4])
        rotated = torch.bmm(R, face_vertex_tensor.reshape(batch_size, -1, 3).transpose(2,1)).reshape(batch_size, -1) 
        scaled = (-torch.abs(camera_para[:, 4].unsqueeze(1)) * rotated).reshape(batch_size, 3, -1)# 
        scaled[:, :2, :] = scaled[:, :2, :] + camera_para[:, 5:7].view(batch_size, 2, 1)
        # face_vertex_tensor=0
        # rotated=0
        # scaled=0
        return face_vertex_tensor, rotated, scaled

    def get_landmark_68(self,face_pixels):
        return face_pixels[ :, :2, self.outer_landmark]

    def get_3d_landmark_68(self, face_vertex):
        return face_vertex[ :, :, self.outer_landmark]

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

        a = torch.div(shape_para, self.e_shape)

        l_u_shape = torch.sum((torch.mean(a, dim = 0) - self.mp_shape) ** 2)
        l_sigma_shape = torch.sum((torch.var(a, dim = 0) - self.vp_shape) ** 2)


        return l_u_shape + l_sigma_shape 

    def get_batch_distribution_regular_exp(self, exp_para):
        """
        """

        b = torch.div(exp_para, self.e_exp)

        
        #l_u_exp = torch.sum((torch.mean(b, dim = 0) - self.mp_exp) ** 2)
        #l_sigma_exp = torch.sum((torch.var(b, dim = 0) -  self.vp_exp) ** 2)

        return torch.sum(b ** 2, dim = 1).mean() 

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
    