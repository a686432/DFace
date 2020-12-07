import torch
import torch.nn as nn
import torch.nn.functional as func
import BFMA_batch, BFMA_3DDFA_batch, BFMF_batch, BFMG_batch, BFMN_batch
import config
import sys
from utils import * 
from fr_loss import CosLoss, RingLoss
import time
from FLAME import FLAME
import BFMP_batch 
def print_grad():
    def hook(grad):
        print(grad)
    return hook

class mixed_loss_batch(nn.Module):
    def __init__(self):
        super(mixed_loss_batch, self).__init__()
        
    def forward(self, pred_camera_exp, pred_shape, gt_lable):
        batch_size = pred_camera_exp.shape[0]
        
        shape_para = pred_shape[:, 0:99]
        exp_para = pred_camera_exp[:, 7:36]
        camera_para = pred_camera_exp[:, 0:7]
        gt_2d_landmark = gt_lable['lm']
        gt_shape = gt_lable['shape']
        #gt_expr = gt_lable['expr']
        
        face = BFMA_batch.BFMA_batch(shape_para, exp_para, camera_para)
        #gt_face_bfmz = BFMZ_batch.BFMZ_batch(gt_shape, gt_expr)
        
        loss_2d_landmark = face.get_2d_landmark_loss(gt_2d_landmark)
        
        #loss_3d_bfmz = face.get_3D_loss_bfmz(gt_face_bfmz.get_face_vertex())
        tikhonov_regularization = face.get_tikhonov_regularization()
        
        mixed_loss = loss_2d_landmark.reshape(-1, 1) + config.loss_weight_tik * tikhonov_regularization #+ config.loss_weight_bfmz_3d * loss_3d_bfmz
        #
        #print(loss_2d_landmark.reshape(-1, 1).shape)
        #print(tikhonov_regularization.shape)
        #exit()
        loss = torch.mean(mixed_loss)
        
        return loss

def _parse_param_batch(param,device):
        """Work for both numpy and tensor"""
        param = param.float()
        N = param.shape[0]
        p_ = param[:, :12].view(N, 3, -1)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].view(N, 3, 1)
        alpha_shp = torch.zeros(N * 99).float().reshape(N, 99)
        alpha_exp = torch.zeros(N * 29).float().reshape(N, 29)
        if config.use_cuda:
            alpha_shp, alpha_exp = alpha_shp.to(device), alpha_exp.to(device)
        
        alpha_shp[:, :40] = param[:, 12:52]
        alpha_exp[:, :10] = param[:, 52:]
        return p, offset, alpha_shp, alpha_exp

class loss_recon_decoder(nn.Module):
    def __init__(self,decode):
        pass
        #self.decoder = Linear_batch()
        # self.sr_part = Decoder.sr_part()
        # self.bfma_3ddfa = BFMA_3DDFA_batch()

class loss_vdc_3ddfa(nn.Module):
    def __init__(self):
        super(loss_vdc_3ddfa, self).__init__()
        self.bfma_3ddfa =  BFMA_3DDFA_batch.BFMA_3DDFA_batch()
        from photometric import Pixel_loss
        from net import PerceptualLoss
        self.pho_loss = Pixel_loss()
        if config.use_perceptual_loss:
            self.PerceptualLoss = PerceptualLoss(requires_grad=False)
        if 'nonlinear' in config.mode: 
            print("Using mlc layer")
            self.bfmf = BFMN_batch.BFMN_batch()
        elif 'Flame' in config.mode:
            self.bfmf = FLAME()
        elif 'bfm' in config.mode:
            
            self.bfmf = BFMP_batch.BFMG_batch()
        else:
            self.bfmf = BFMG_batch.BFMG_batch()
        mean_albedo = torch.Tensor(np.load('../propressing/uv_albedo_map.npy'))
        self.register_buffer('mean_albedo',mean_albedo)
        mask_albedo = torch.Tensor(np.load('../propressing/mask.npy'))
        self.mask_albedo_npixels = mask_albedo.sum()
        self.register_buffer('mask_albedo',mask_albedo)
        kernel_low = torch.FloatTensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        kernel_high = torch.FloatTensor([[1.0/16,1.0/8,1./16],[1./8,1./4,1./8],[1./16,1./8,1./16]])
        #kernel = torch.FloatTensor(kernel)

        self.kernel_low = nn.Parameter(data=kernel_low,requires_grad=False) 
        self.kernel_high = nn.Parameter(data=kernel_high,requires_grad=False) 


        #self.weights_landmarks = torch.ones(68)
        #self.register_buffer('pi',torch.Tensor(np.pi))
        
        #self.weights_landmarks[17:68] = self.weights_landmarks[17:68]*10
        #self.register_buffer('weights_landmark',self.weights_landmarks)

    def reset_albedo(self):
        mean_albedo = torch.Tensor(np.load('../propressing/uv_albedo_map.npy'))
        self.mean_albedo = mean_albedo
        mask_albedo = torch.Tensor(np.load('../propressing/mask.npy'))
        self.mask_albedo_npixels = mask_albedo.sum()
        self.mask_albedo = mask_albedo

    def BFM_forward(self,shape_para, exp_para, camera_para):
        face_proj = self.bfmf(shape_para, exp_para, camera_para)
        return face_proj

    def smooth_loss(self,tensor_image,mask):
        tensor_image = tensor_image.permute(0,3,1,2)
        mask = mask.permute(0,3,1,2)
        kernel = self.kernel_low.expand(tensor_image.shape[1],tensor_image.shape[1],3,3)
        loss = ((func.conv2d(tensor_image,kernel,stride=1,padding=1)*mask)**2).sum()/self.mask_albedo_npixels/tensor_image.shape[0]
        return loss
    

    def sharp_loss(self,tensor_image,mask):
        tensor_image = tensor_image.permute(0,3,1,2)
        mask = mask.permute(0,3,1,2)
        kernel = self.kernel_high.expand(tensor_image.shape[1],tensor_image.shape[1],3,3)
        loss = ((func.conv2d(tensor_image,kernel,stride=1,padding=1)*mask)**2).sum()/self.mask_albedo_npixels/tensor_image.shape[0]
        return loss
    
        

    def forward(self, pred_camera_exp, pred_shape, target,albedo=None,ori_img=None, conf =None):
        # st = time.time()
        metrics = {}
        train_result = {}
        metrics['vdc_landmark'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['vdc_landmark_res'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['shape_smooth_loss'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_edge'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['regular_loss_shape'] =torch.Tensor([0]).to(pred_shape.device)
        metrics['regular_loss_exp'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_perc_im'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['pixel_loss'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['pixel_loss_all'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_3d'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_3d_raw'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['pixel_loss_all'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_albedo_reg'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['albedo_smooth_loss'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['albedo_res_loss'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_fr'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_center'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['weighted_center_loss'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['loss_fr_all'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['albedo_sharp_loss'] = torch.Tensor([0]).to(pred_shape.device)
        metrics['res_shape_loss'] = torch.Tensor([0]).to(pred_shape.device)
        #metrics['center_loss'] = torch.Tensor([0]).to(pred_shape.device)

       # metrics['loss_perc_im'] = loss_perc_im
        train_result['images_s'] = None
        train_result['illum_images_s'] = None
        train_result['ver_color_s'] = None
        train_result['center_norm'] = torch.Tensor([0]).to(pred_shape.device)
        


        batch_size = pred_camera_exp.shape[0]
        gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(target,pred_shape.device)
        
        shape_para = pred_shape
        exp_para = pred_camera_exp[:, 7:36]
        camera_para = pred_camera_exp[:, 0:7]
        illum = pred_camera_exp[:,36:63]

        if config.use_flip and torch.randn(1)>config.filp_threshold:
            face_proj,face_proj_res = self.bfmf(shape_para, exp_para, camera_para,shape_flip=True)
        #print(shape_para.device, exp_para.device, camera_para.device, target.device)
        else:
            face_proj,face_proj_res = self.bfmf(shape_para, exp_para, camera_para)

        gt_face_proj = self.bfma_3ddfa(gt_shape, gt_exp , gt_rotate, gt_offset)


        pred_landmark =  self.bfmf.get_landmark_68(face_proj)
        g_landmark = self.bfma_3ddfa.get_landmark_68(gt_face_proj)
        pred_landmark_res =  self.bfmf.get_landmark_68(face_proj_res)


        #g_landmark_res = self.bfma_3ddfa.get_landmark_68(gt_face_proj)
        vdc_landmark = ((g_landmark - pred_landmark)**2).reshape(batch_size,-1).mean(1)
        metrics['vdc_landmark'] = vdc_landmark.mean()


        if config.use_face_recognition_constraint:
            if config.use_res_shape:
                res_shape_loss = self.bfmf.get_resloss(shape_para) + self.bfmf.get_resloss2(shape_para)
            else:
                res_shape_loss = self.bfmf.get_resloss(shape_para)
            metrics['res_shape_loss'] = res_shape_loss

        if config.use_res_shape:
            vdc_landmark_res = ((g_landmark - pred_landmark_res)**2).reshape(batch_size,-1).mean(1)
            metrics['vdc_landmark_res'] = vdc_landmark_res.mean()
            loss =  config.mix_loss_weight_3d_vdc_lm * vdc_landmark + config.weight_res_landmark*vdc_landmark_res
            
            #metrics['loss_3d_raw'] = loss
        
                
                #loss += shape_smooth_loss*config.weight_shape_smooth_loss
        
        else:
            loss = config.mix_loss_weight_3d_vdc_lm * vdc_landmark

        if config.use_shape_smooth:
           # shape_smooth_loss = ((face_proj_res - (face_proj_res[:,self.bfmf.adj_matrix,:]*self.adj_mask.unsqueeze(0).unsqueeze(-1)).sum(2)/self.adj_mask.sum(1).reshape(1,-1,1))**2).mean()
            shape_smooth_loss = self.bfmf.shape_smooth_loss_m(shape_para)
            metrics['shape_smooth_loss'] = shape_smooth_loss
        
        metrics['loss_3d_raw'] = loss
        metrics['loss_3d'] = loss.mean() + metrics['shape_smooth_loss']*config.weight_shape_smooth_loss


        # if config.use_edge_lm:
        #     edge_landmark = self.bfma_3ddfa.get_edge_landmark(g_landmark)-self.bfmf.get_edge_landmark(pred_landmark)
        #     loss_edge = (edge_landmark**2).mean()
        #     metrics['loss_edge'] = loss_edge


        # edge_landmark = torch.norm(edge_landmark.reshape(batch_size, -1),p=2,dim=-1)
        # loss_edge = torch.mean(edge_landmark.reshape(batch_size, -1), dim = 1)
        # loss_edge = 0



        regular_loss_shape = self.bfmf.get_batch_distribution_regular_shape(shape_para)
        regular_loss_exp = self.bfmf.get_tikhonov_regularization_exp(exp_para)
        metrics['regular_loss_shape'] = regular_loss_shape
        metrics['regular_loss_exp'] = regular_loss_exp
        #if config.use_res_shape:
            

        # st2 = time.time()-st-st1
        # print('vdc time:',st2)
        
        #print(config.weight_pixel_loss)
        if not config.weight_pixel_loss==0:
            albedo_a, albedo_s = albedo


            albedo_s = albedo_s.permute(0,2,3,1)
            albedo_a = albedo_a.permute(0,2,3,1)
            # albedo_a = albedo_a - albedo_a.reshape(batch_size,-1,3).mean(dim=1).reshape(batch_size,1,1,3).expand(batch_size,albedo_a.shape[1],albedo_a.shape[2],albedo_a.shape[3])


            mean_albedo = self.mean_albedo.expand(batch_size,albedo_a.shape[1],albedo_a.shape[2],albedo_a.shape[3])
            train_result['albedo_s'] = mean_albedo+albedo_s
            train_result['albedo_a'] = mean_albedo+albedo_a
            
            

            #loss_albedo_reg = ((albedo_a+albedo_s).mean())**2
            loss_albedo_reg = albedo_a.mean()**2 + albedo_s.mean()**2
            mask_albedo = self.mask_albedo.view(1,self.mask_albedo.shape[0],self.mask_albedo.shape[1],1).expand(batch_size,self.mask_albedo.shape[0],self.mask_albedo.shape[1],3)
            #face_albedo = mask_albedo*albedo
            #mean_face_albedo = (face_albedo.view(batch_size,-1,3).sum(1) / self.mask_albedo_npixels).view(batch_size,1,1,3)
            
            #texels = interpolate_texture_map(fragments, meshes)

            metrics['loss_albedo_reg'] = loss_albedo_reg
                #albedo, albedo_s = albedo
            if config.use_flip and torch.randn(1)>config.filp_threshold:
                albedo_a = torch.flip(albedo_a,[2])*mask_albedo +  (1-mask_albedo)*albedo_a
                albedo_flip_flag = True
            else: 
                albedo_flip_flag = False
            
                #albedo_s = torch.flip(albedo_s,[2])
            if config.use_albedo_res:
                albedo_res_loss = (torch.abs(albedo_s+albedo_a).mean())**2
                metrics['albedo_res_loss'] = albedo_res_loss
                albedo_res = albedo_a + albedo_s 
            else:
                albedo_res = albedo_a
            
            albedo_smooth_loss = self.smooth_loss(albedo_a,mask_albedo)
            metrics['albedo_smooth_loss'] = albedo_smooth_loss
            albedo_sharp_loss = self.sharp_loss(albedo_s,mask_albedo)
            metrics['albedo_sharp_loss'] = albedo_sharp_loss



            # face_diff = torch.zeros_like(albedo_res).to(albedo_res.device)
            # face_albedo_l = (albedo_res[:,:,:-1,:] - albedo_res[:,:,1:,:])**2
            # face_albedo_r = (albedo_res[:,:,1:,:] - albedo_res[:,:,:-1,:])**2
            # face_albedo_u = (albedo_res[:,:-1,:,:] - albedo_res[:,1:,:,:])**2
            # face_albedo_d = (albedo_res[:,1:,:,:] - albedo_res[:,:-1,:,:])**2
            # face_diff[:,:,:-1,:] += face_albedo_l
            # face_diff[:,:,1:,:] += face_albedo_r
            # face_diff[:,:-1,:,:] += face_albedo_u
            # face_diff[:,1:,:,:] += face_albedo_d
            
            # albedo_smooth_loss = (face_diff/4 * mask_albedo).sum() / self.mask_albedo_npixels
            
            # if config.use_proxy:
            #     pixel_loss_r, images_r,illum_images_r,ver_color_r = self.pho_loss(ori_img,face_proj_res,albedo,illum)
            #     pixel_loss_s, images_s,illum_images_s,ver_color_s = self.pho_loss(ori_img,face_proj_res,albedo_s,illum,proxy=True)
            #     pixel_loss = (pixel_loss_r+pixel_loss_s)/2
            #     proxy_loss =  torch.abs(albedo.detach()-albedo_s).mean()
            #     pixel_loss += config.proxy_loss_weight *proxy_loss
            albedo = mean_albedo + albedo_res
            
            if config.use_confidence_map:

                conf_a, _ = conf
                if albedo_flip_flag:
                    conf_a = conf_a[:,1:]
                else:
                    conf_a = conf_a[:,:1]
                
                pixel_loss, images_s,illum_images_s,ver_color_s = self.pho_loss(ori_img,face_proj_res,albedo,illum,conf=conf_a)

                
            else:
                pixel_loss, images_s,illum_images_s,ver_color_s = self.pho_loss(ori_img,face_proj_res,albedo,illum)
            metrics['pixel_loss'] = pixel_loss


            train_result['images_s'] = images_s
            train_result['illum_images_s'] = illum_images_s
            train_result['ver_color_s'] = ver_color_s
            train_result['albedo'] = albedo

            if not config.weight_loss_perc_im == 0:
                if config.use_perceptual_loss:
                    if config.use_confidence_map:
                        _, conf_lm = conf
                        if albedo_flip_flag:
                            conf_lm = conf_lm[:,1:]
                        else:
                            conf_lm = conf_lm[:,:1]
                        loss_perc_im = self.PerceptualLoss(images_s, ori_img,conf_lm)
                    else:
                        loss_perc_im = self.PerceptualLoss(images_s, ori_img)
                    metrics['loss_perc_im'] = loss_perc_im
                    #pixel_loss += config.weight_loss_perc_im * loss_perc_im
            
            metrics['pixel_loss_all'] = pixel_loss

        # st3 = time.time()-st-st1-st2
        # print('pho time:',st3)

        #metrics = {'res_shape_loss': res_shape_loss}
        # metrics['res_shape_loss'] = res_shape_loss
        # metrics['regular_loss_shape'] = regular_loss_shape
        # metrics['regular_loss_exp'] = regular_loss_exp
        # metrics['loss_edge'] = loss_edge
        return metrics,train_result

 
    
class mixed_loss_FR_batch(nn.Module):
    def __init__(self, fr_ip=None, fr_loss=None, fr_loss_sup = None, d=None):
        super(mixed_loss_FR_batch, self).__init__()
        self.fr_ip = fr_ip
        self.fr_loss = fr_loss
        self.loss_3d_func = loss_vdc_3ddfa()
        self.fr_loss_sup = fr_loss_sup
        self.D = d
        self.dim = 199
        #self.bfma =BFMA_batch.BFMA_batch()
        #RingLoss(loss_weight=0.01)


    def forward(self, pred_camera_exp, pred_shape, gt_label,albedo=None,ori_img=None,conf=None):
        batch_size = pred_camera_exp.shape[0]
        

           
        gt_3d = gt_label['3d']
        metrics,train_result =  self.loss_3d_func(pred_camera_exp, pred_shape, gt_3d,albedo,ori_img,conf)


        ######################################################################################
        if not self.fr_loss is None:
            fr_embedding = pred_shape[:,:self.dim]  
            gt_id = gt_label['id'].long()
            loss_fr = gt_label['ind_id'].float() * self.fr_loss(self.fr_ip(fr_embedding), gt_id)
            loss_fr = torch.sum(loss_fr) / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_fr*0) 
            metrics['loss_fr'] = loss_fr
            if not self.fr_loss_sup is None:

                #weighted_center_loss = None

                weights = get_weighted(pred_camera_exp[:, 7:36],pred_camera_exp[:, 0:7]).detach()
                #center_embedding = torch.div(fr_embedding,self.loss_3d_func.bfmf.e_shape)

                loss_center, centers, weighted_center_loss=self.fr_loss_sup(gt_id, fr_embedding, gt_label['ind_id'].float()) if self.fr_loss_sup is not None else 0
                #gan_loss = self.D()
                #weighted_center_loss = loss_center*weights.reshape(batch_size,1)
                loss_center = torch.sum(loss_center) / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_center * 0)
                
                #weighted_center_loss = torch.sum(weighted_center_loss) / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(weighted_center_loss * 0)
                #weighted_center_loss = weighted_center_loss*config.mix_loss_weight_fr_center_loss*config.mix_loss_weight_fr 
                metrics['loss_center'] = loss_center
                metrics['weighted_center_loss'] = weights
                
                c_norm = centers.norm(dim=1).mean()
                train_result['center_norm'] = c_norm
            else: 
                loss_center = torch.Tensor([0]).to(pred_shape.device)

            
            #loss_center = loss_center / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_center * 0)


            loss_fr_all =  loss_center*config.mix_loss_weight_fr_center_loss + loss_fr
            metrics['loss_fr_all'] = loss_fr_all
        
        #loss_fr = feature_fr
        #print(gt_label['ind_id'].float())

        
        


        loss_3d = torch.sum(gt_label['ind_3d'].float() * metrics['loss_3d_raw']) / torch.sum(gt_label['ind_3d'].float()) if torch.sum(gt_label['ind_3d']) >= 1 else torch.sum(metrics['loss_3d_raw'] * 0)
        metrics['loss_3d'] = loss_3d +  metrics['shape_smooth_loss']*config.weight_shape_smooth_loss
        #loss = loss_3d + config.mix_loss_weight_fr * loss_fr_all + regular_loss_exp*config.loss_weight_tik_exp
        #print(loss)
        #print(loss.shape, loss_3d.shape, loss_2d_landmark.shape)
        #print "%f %f %f %f\r" %  (torch.mean(loss_2d_landmark).data.cpu().numpy(), torch.mean(loss_fr).data.cpu().numpy(),torch.mean(loss_3d).data.cpu().numpy(), loss.data.cpu().numpy())
        #loss = loss_fr
        
        return metrics,train_result

class mixed_loss_data_batch(nn.Module):
    def __init__(self, fr_ip, fr_loss, fr_loss_sup = None, d=None):
        super(mixed_loss_data_batch, self).__init__()
        self.fr_ip = fr_ip
        self.fr_loss = fr_loss
        self.loss_3d_func = loss_vdc_3ddfa()
        self.fr_loss_sup = fr_loss_sup
        self.D = d
        self.dim = 199
        #self.bfma =BFMA_batch.BFMA_batch()
        #RingLoss(loss_weight=0.01)


    def forward(self, pred_camera_exp, pred_shape, gt_label,albedo=None,ori_img=None):
        batch_size = pred_camera_exp.shape[0]
        
    
        fr_embedding = pred_shape[:,:self.dim]     
        gt_3d = gt_label['3d']
        metrics,train_result =  self.loss_3d_func(pred_camera_exp, pred_shape, gt_3d,albedo,ori_img)


        ######################################################################################

        gt_id = gt_label['id'].long()
        loss_fr = gt_label['ind_id'].float() * self.fr_loss(self.fr_ip(fr_embedding), gt_id)
        loss_fr = torch.sum(loss_fr) / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_fr*0) 
        metrics['loss_fr'] = loss_fr
        if not self.fr_loss_sup is None:

            #weighted_center_loss = None
            weights = get_weighted(pred_camera_exp[:, 7:36],pred_camera_exp[:, 0:7])

            loss_center, centers, weighted_center_loss=self.fr_loss_sup(gt_id, fr_embedding, gt_label['ind_id'].float(),weights) if self.fr_loss_sup is not None else 0
            #gan_loss = self.D()
            #weighted_center_loss = loss_center*weights.reshape(batch_size,1)
            loss_center = torch.sum(loss_center) / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_center * 0)
            
            weighted_center_loss = torch.sum(weighted_center_loss) / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(weighted_center_loss * 0)
            weighted_center_loss = weighted_center_loss*config.mix_loss_weight_fr_center_loss*config.mix_loss_weight_fr 
            metrics['loss_center'] = loss_center
            metrics['weighted_center_loss'] = weighted_center_loss
        else: 
            loss_center = torch.Tensor([0]).to(pred_shape.device)

        
        #loss_center = loss_center / torch.sum(gt_label['ind_id'].float()) if torch.sum(gt_label['ind_id']) >= 1 else torch.sum(loss_center * 0)


        loss_fr_all =  loss_center*config.mix_loss_weight_fr_center_loss + loss_fr
        metrics['loss_fr_all'] = loss_fr_all
        
        #loss_fr = feature_fr
        #print(gt_label['ind_id'].float())

        
        


        loss_3d = torch.sum(gt_label['ind_3d'].float() * metrics['loss_3d_raw']) / torch.sum(gt_label['ind_3d'].float()) if torch.sum(gt_label['ind_3d']) >= 1 else torch.sum(metrics['loss_3d_raw'] * 0)
        metrics['loss_3d'] = loss_3d +  metrics['shape_smooth_loss']*config.weight_shape_smooth_loss
        #loss = loss_3d + config.mix_loss_weight_fr * loss_fr_all + regular_loss_exp*config.loss_weight_tik_exp
        #print(loss)
        #print(loss.shape, loss_3d.shape, loss_2d_landmark.shape)
        #print "%f %f %f %f\r" %  (torch.mean(loss_2d_landmark).data.cpu().numpy(), torch.mean(loss_fr).data.cpu().numpy(),torch.mean(loss_3d).data.cpu().numpy(), loss.data.cpu().numpy())
        #loss = loss_fr
        
        return metrics,train_result


def get_weighted(exp_para,pose_para):
    '''
        get pose_texture 
        get exp_norm
    '''
    R = Q2R_batch(pose_para[:, 0:4])
    x,y,z = matrix2angle(R)
    #print(x,y,z)
    #print((x.cos()+1),(y.cos()+1),(z.cos()+1),(-1.0/3*torch.norm(exp_para,dim=1)).exp())
    return (x.cos().clamp(min=0))*(y.cos().clamp(min=0))*(z.cos().clamp(min=0))*(-1.0/3*torch.norm(exp_para,dim=1)).exp()


def asymmetric_euclidean_loss(pred,target):
        l1, l2 = 1.0, 3.0
        gamma_plus = torch.abs(target)
        gamma_pred_plus = torch.sign(target) * pred
        gamma_max = torch.max(gamma_plus, gamma_pred_plus)
        return torch.mean(l1 * torch.sum((gamma_plus - gamma_max) ** 2, 1) + l2 * torch.sum((gamma_pred_plus - gamma_max) ** 2, 1))



            

