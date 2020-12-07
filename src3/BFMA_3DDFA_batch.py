import numpy as np
import scipy.io    as sio
import torch
from torch.autograd import Variable
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt, pi
from utils import *
import config
from misc import *
import torch.nn as nn
"""
 coor-sys means coordinate system!

"""

class BFMA_3DDFA_batch(nn.Module):
    def __init__(self):
        super(BFMA_3DDFA_batch, self).__init__()
        """    
        A batch version for BFMA_3DDFA
        
        A 3DMM class similar to the BFM proposed by Anh
        A face model can be presented as:
            face_mesh  = mu + w * Shape_para + w_expression * Exp_para
            where mu is the mean face, w is the shape basis, w_expression is the expression basis
        
        According to 3DDFA, the shape basis and expression basis come from BFM and FaceWarehouse respectively
        
        Atttibutes:
            mu:     A 1-D array of the mean face mesh vertex, 140970 * 1, [x0, y0, z0, x1, y1, z1, ..., xn-1, yn-1, zn-1]
            index:    A 2-D array of the face mesh triangle index, 93323 * 3, [[i00, i10, i20], [i01, i11, i21], ..., [i0n-1, i1n-1, i2n-1]]
            
            w: A 1-D array of the shape_base, 140970 * 199
            w_expression:  1-D array of the expression base, 140970 * 29
            
            inner_landmark: A 1-d array of the 68 inner landmark indices
            outer_landmark: : A 1-d array of the 68 outter landmark indices
        
        """
        bfm=np.load("../propressing/bfmz.npz")
        w_expression= bfm['w_expression'].reshape(-1,3,29).reshape(-1,29).T/1000
        mu_expression=bfm['mu_expression'].reshape(-1,3).reshape(-1)/1000
        w_shape = bfm['w_shape'][:,0:99].reshape(-1,3,99).reshape(-1,99).T
        mu_shape = bfm['mu_shape'].reshape(-1)/1000
        index =  bfm['index']
        mu = mu_shape + mu_expression
        shape_ev = bfm['shape_ev']
        exp_ev = bfm['exp_ev']
        g_model = np.load(config.g_model_path)
        #print(mu_shape)
        outer_landmark = g_model['landmarks'].astype(int)

        # print(mu_shape)
        # print(mu_expression_tensor)
        
        # exit()
        
        #1. construct  the mean face and the base from local .mat files
        
        #1.1 load the data from .mat files
        # data_anh_model =  (sio.loadmat(config.data_anh_model_mat_path, squeeze_me=True,struct_as_record=False))['BFM']
        # #1.2 load the data from the dicts
        # #w[1::3, :] = -w[1::3, :]
        # #w[2::3, :] = -w[2::3, :]
        # mu_shape = data_anh_model.shapeMU
        # w = data_anh_model.shapePC
        # shape_ev = data_anh_model.shapeEV
        # index = data_anh_model.faces
        # for i in range((index.shape)[0]):
        #     index[i,:] = index[i,:] - 1
        # inner_landmark = data_anh_model.innerLandmarkIndex.astype('int') - 1
        # outer_landmark = data_anh_model.outerLandmarkIndex.astype('int') - 1
        # print(outer_landmark)
        # w_expression = data_anh_model.expPC * 1000
        # mu_expression = data_anh_model.expMU
        # exp_ev = data_anh_model.expEV
        # mu = mu_shape + mu_expression
        # mu_vertex = mu.reshape(-1, 3)
        # anh_zhu_index_map = np.load(config.anh_zhu_index_map_path)
        # outer_landmark_index_cand = np.load(config.outer_landmark_index_cand_path)
        
        
        aflw_21_landmarks = np.array([37535, 4645, 38561, 38935, 11592, 39955, 2087, 4409, 5958, 10326, 11872, 14194, 19841, 6384, 8190, 9997, 33862, 5391, 8224, 11048, 43504])
        # gl coor-sys --> cv coor-sys
        # (x, y, z) --> (x, -y, -z)
        #mu_shape[1::3] = -mu_shape[1::3]
        #mu_shape[2::3] = -mu_shape[2::3]
        self.mu_shape_tensor = torch.from_numpy(mu_shape.astype(np.float32)) # to calculate loss and bp
        self.w_tensor = torch.from_numpy(w_shape.astype(np.float32)) # to calculate loss and bp , shape:(N, 99)
        self.shape_ev_tensor = torch.from_numpy(shape_ev.astype(np.float32)) #shape: (99,)
        self.index_tensor = torch.from_numpy(index.astype('int')).long()
        #self.inner_landmark_tensor = torch.from_numpy(inner_landmark).long()
        self.outer_landmark_tensor = torch.from_numpy(outer_landmark).long()
        self.w_expression_tensor = torch.from_numpy(w_expression.astype(np.float32))
        self.mu_expression_tensor =torch.from_numpy(mu_expression.astype(np.float32))
        self.exp_ev_tensor = torch.from_numpy(exp_ev.astype(np.float32))
        self.mu_tensor = torch.from_numpy(mu.astype(np.float32))
        self.aflw_21_landmarks_tensor = torch.from_numpy(aflw_21_landmarks).long()
        # self.mu_tensor = self.mu_tensor * 1000
        
        self.mu_tensor[0::3] = self.mu_tensor[0::3] - 0.231
        #print("111",self.mu_shape_tensor)
        

        #print(outer_landmark)
        landmark_map = np.load('../propressing/landmark.npy')
        landmark_tensor = torch.Tensor(outer_landmark[landmark_map!=-1]).long()
        #print(landmark_tensor)

        self.register_buffer('mu',self.mu_tensor)
        self.register_buffer('mu_shape',self.mu_shape_tensor)
        self.register_buffer('mu_exp',self.mu_expression_tensor)
        self.register_buffer('w_shape',self.w_tensor)
        self.register_buffer('w_exp', self.w_expression_tensor)
        self.register_buffer('e_shape', self.shape_ev_tensor)
        self.register_buffer('e_exp', self.exp_ev_tensor)
        self.register_buffer('tris', self.index_tensor)
        self.register_buffer('outer_landmark',self.outer_landmark_tensor)
        self.register_buffer('landmark',landmark_tensor)


            
        
    def forward(self, shape_para, exp_para, rotate, translate):
        """
        Input:
            shape_para: batch * 99
            exp_rapa: batch * 29
            camera_para:batch * (4 + 3 + 1) R-4, T-3, S-1
        """
        
        batch_size = shape_para.shape[0]
        face_vertex_tensor = self.mu +  shape_para @ self.w_shape/1000 + exp_para @ self.w_exp
        # print("f: ", face_vertex_tensor)
        # print(translate/1000)
        # print(rotate*1000)
        # print("s: ",shape_para @ self.w_shape/1000)
        # print("e: ",exp_para @ self.w_exp)
        # print("F: ",face_vertex_tensor)
        # print("S: ", torch.mm(self.w_shape, shape_para.transpose(1, 0)).transpose(1, 0))
        # print("E: ",  torch.mm(self.w_exp, exp_para.transpose(1, 0)).transpose(1, 0))
        # print("EP: ",exp_para )
        # print("EW: ", self.w_exp)
        # print("SW: ", self.w_shape.t() @ self.w_shape)
        #print("SW: ", self.w_shape @ self.w_shape.t())
        '''
        transform the face using given rotate matrix and translate vector
        Input:
            rotate:     BatchSize * 3 * 3
            translate:  BatchSize * 3 * 1
        '''
        # print(rotate)
        # print(face_vertex_tensor)
        rotated = torch.bmm(rotate*1000, face_vertex_tensor.reshape(batch_size, -1, 3).transpose(2,1)) 
        #print("r: ", rotated)
        '''

        print(res.shape)
        lms = res[1, :2, [1827, 14452, 8190, 5391, 10919]]
        lms[1] = 120 - lms[1]
        print(lms.transpose(1, 0))
        print(res[1, :, BFMA_3DDFA_batch.outer_landmark])
        
        from PIL import Image, ImageDraw
        
        projected = res[1, :, BFMA_3DDFA_batch.outer_landmark].data.cpu().detach().numpy().transpose(1,0)[:, :2]
        img = Image.open('/ssd-data/lmd/train_aug_120x120/LFPWFlip_LFPW_image_train_0859_12_2.jpg')
        drawObject = ImageDraw.Draw(img)
        print(projected.shape)
        # draw predicted landmarks
        for i in range(68):
            pred_point = projected[i]
            pred_point[1] = 120 - pred_point[1] 
            drawObject.ellipse((pred_point[0]-1,pred_point[1]-1,pred_point[0]+1,pred_point[1]+1),fill = "red")
        img.save('imgs/test.jpg')
        exit()
        '''
        res = rotated + translate
        res[:, [1, 2], :] = -res[:, [1, 2], :]
        '''
        res = (res - 60) * 112. / 120_
        res[:,0,:] = res[:,0,:] + 96 / 2
        res[:,1,:] = res[:,1,:] + 112 / 2
        '''

        res[:, 1, :] = 112 + res[:, 1, :]

        return  res # batch size * 3 * N

    def get_landmark(self,face_pixels):
        return face_pixels[ :, :2, self.landmark]

    def get_landmark_68(self,face_pixels):
        return face_pixels[ :, :2, self.outer_landmark]

    def get_edge_landmark(self,face_pixels):
        batch_size = face_pixels.shape[0]
        landmarks = face_pixels[:,:2,self.outer_landmark].transpose(2,1)
        landmarks = landmarks.transpose(0,1).expand(68,68,batch_size,2).transpose(0,2)
        edge_distance = (((landmarks-landmarks.transpose(2,1))**2).sum(dim=-1)+1e-12).sqrt()
        return edge_distance


    def get_3d_landmark_68(self, face_vertex):
        return face_vertex[ :, :, self.outer_landmark]

    def get_aflw_21_landmark(self,face_pixels):
        return face_pixels[ :, :, self.aflw_21_landmarks]

    def get_outer_moved_landmark(self, face_vertex, camera_para):
        outer_landmark_moved_idx =__move_boundary_landmarks(face_vertex, camera_para)
        return face_pixels[ :, :, outer_landmark_moved_idx]


    def __move_boundary_landmarks(self,face_vertex_tensor, Camera_Para):
        phi, theta , psi = Q2E_batch(Camera_Para[:, 0:4]) # Q -> E
        psi -= psi # set psi to 0
        R = Q2R_batch(E2Q_batch(phi, theta, psi)) # E -> Q -> R R shape = (batch_size, 3, 3)
        rotated_vertex = torch.bmm(R, self.face_vertex_tensor.reshape(batch_size, -1, 3).transpose(1, 2)).transpose(1, 2)
        
        lms = np.append(np.array(range(0, 6)),np.array(range(11, 17))) 
        mask = np.array([0, 0, 0, 0, 0, 0, -5, -5, -5, -5, -5, -5])
        
        landmark_vertex_cand = rotated_vertex[:, BFMA_3DDFA_batch.outer_landmark_index_cand[lms + mask]]
        
        selected_1 = torch.min(landmark_vertex_cand[:, :6], dim = 2)[1]
        selected_2 = torch.max(landmark_vertex_cand[:, 6:12], dim = 2)[1]
        selected = torch.cat([selected_1, selected_2], dim = 1)

        
        """
        landmark_vertex_cand = rotated_vertex[:, BFMA_3DDFA_batch.outer_landmark_index_cand[lms + mask]]
        
        landmark_vertex_cand[:, 6:12] = -landmark_vertex_cand[:, 6:12]
        selected = torch.min(landmark_vertex_cand, dim = 2)[1]
        """
        if config.use_cuda:
            selected = selected.data.cpu().numpy()
        else:
            selected = selected.data.numpy()
            
        self.outer_landmark_moved[:, lms] = BFMA_3DDFA_batch.outer_landmark_index_cand[lms + mask, selected[:, :, 0]]

        
    def get_tikhonov_regularization(self):
        """
        calculate the 3DMM parameters' tikhonov regularization loss
        Output:
            regularization_loss: shape = (batch_size)
        
        """
        
        a = torch.div(self.Shape_Para, BFMA_3DDFA_batch.shape_ev_tensor)
        b = torch.div(self.Exp_Para, BFMA_3DDFA_batch.exp_ev_tensor)
        return config.tik_shape_weight * torch.sum(a ** 2, dim = 1, keepdim = True) + torch.sum(b ** 2, dim = 1, keepdim = True)
    
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
        
        trimed_gt = gt_vertex_bfmz[:, BFMA_3DDFA_batch.anh_zhu_index_map]
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
            P = P.to(config.device)
        if use_not_moved:
            rotated = torch.bmm(R, self.outer_landmark_not_moved_tensor.transpose(2,1)).reshape(self.batch_size, -1)
        else:
            rotated = torch.bmm(R, self.outerlandmark_vertex_tensor if not use_aflw else self.aflw_21_landmark_vertex_tensor).reshape(self.batch_size, -1)
        #print 'camera scale = ', torch.abs(self.Camera_Para[:, 4]).unsqueeze(1)
        scaled = (self.Camera_Para[:, 4].unsqueeze(1) * rotated).reshape(self.batch_size, 3, -1)
        projected = torch.bmm(P , scaled) 
        translated = projected + self.Camera_Para[:, 5:7].view(self.batch_size, 2, 1)

        return translated
    
    def transform_face(self):
        '''
        transform the face using camera     
        '''
        R = Q2R_batch(self.Camera_Para[:, 0:4])
        rotated = torch.bmm(R, self.face_vertex_tensor.reshape(self.batch_size, -1, 3).transpose(2,1)) # batchszize * 3 * N
        #print(rotated[:, 0, :].shape)
        #print(self.Camera_Para[:, 5].view(self.batch_size, 1, 1).shape)
        #rotated[:, 0, :] + self.Camera_Para[:, 5].view(self.batch_size, 1, 1)
        #rotated[:, 1, :] + self.Camera_Para[:, 6].view(self.batch_size, 1, 1)
        rotated[:, :2, :] = rotated[:, :2, :] + self.Camera_Para[:, 5:7].view(self.batch_size, 2, 1)
        return rotated
    
    def transform_face_vertices_manual(self, rotate, translate):
        '''
        transform the face using given rotate matrix and translate vector
        Input:
            rotate:     BatchSize * 3 * 3
            translate:  BatchSize * 3 * 1
        '''
        
        res = (torch.bmm(rotate, self.face_vertex_tensor.reshape(self.batch_size, -1, 3).transpose(2,1)) + translate)
        '''
        print(res.shape)
        lms = res[1, :2, [1827, 14452, 8190, 5391, 10919]]
        lms[1] = 120 - lms[1]
        print(lms.transpose(1, 0))
        print(res[1, :, .outer_landmark])
        
        from PIL import Image, ImageDraw
        
        projected = res[1, :, BFMA_3DDFA_batch.outer_landmark].data.cpu().detach().numpy().transpose(1,0)[:, :2]
        img = Image.open('/ssd-data/lmd/train_aug_120x120/LFPWFlip_LFPW_image_train_0859_12_2.jpg')
        drawObject = ImageDraw.Draw(img)
        print(projected.shape)
        # draw predicted landmarks
        for i in range(68):
            pred_point = projected[i]
            pred_point[1] = 120 - pred_point[1] 
            drawObject.ellipse((pred_point[0]-1,pred_point[1]-1,pred_point[0]+1,pred_point[1]+1),fill = "red")
        img.save('imgs/test.jpg')
        exit()
        '''
        res[:, [1, 2], :] = -res[:, [1, 2], :]
        '''
        res = (res - 60) * 112. / 120
        res[:,0,:] = res[:,0,:] + 96 / 2
        res[:,1,:] = res[:,1,:] + 112 / 2
        '''
        #print(res[0, :, 0::3000])
        res[:, 1, :] = 112 + res[:, 1, :]
        return  res# batch size * 3 * N
        
