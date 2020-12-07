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
import pickle
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler
import config
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
EPS = 1e-7
class Pixel_loss(nn.Module):
    def __init__(self):
        super(Pixel_loss, self).__init__()
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

        # with open(config.flame_model_path, 'rb') as f:
        #     self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        #     self.NECK_IDX = 1
        #     self.batch_size = config.batch_size
        #     self.dtype = torch.float32
        #     self.use_face_contour = config.use_face_contour
        #     self.faces = self.flame_model.f
        #     self.register_buffer('tris',
        #                         to_tensor(to_np(self.faces, dtype=np.int64),
        #                                 dtype=torch.long))


        #g_model = np.load(config.g_model_path)

        #outer_landmark_tensor = torch.Tensor(g_model['landmarks'].astype(int)).long()
        #index = torch.Tensor(g_model['faces'])
        g_model = np.load(config.g_model_path)
        index = torch.Tensor(g_model['faces'])
        #mean_albedo = torch.Tensor(np.load('../propressing/uv_albedo_map.npy'))


        # R, T = look_at_view_transform(200, 0, 0) 
        #self.cameras = SfMOrthographicCameras(focal_length=2.0/config.img_h,R=R, T=T)
        self.raster_settings = RasterizationSettings(
            image_size=112, 
            blur_radius=0.0, 
            faces_per_pixel=8, 
            bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None  # this setting is for coarse rasterization
        )

        la = [np.pi, 2*np.pi/3.0, 0.25*np.pi]
        H = np.zeros(9)
        H[0] = la[0] * (1 / torch.sqrt(4 * torch.Tensor([np.pi])))
        H[1] = la[1] * np.sqrt(3 / (4 * np.pi)) 
        H[2] = la[1] * np.sqrt(3 / (4 * np.pi)) 
        H[3] = la[1] * np.sqrt(3 / (4 * np.pi)) 
        H[4] = la[2] * 0.5 * np.sqrt(5 / (4 * np.pi)) 
        H[5] = la[2] * 3 * np.sqrt(5 / (12 * np.pi)) 
        H[6] = la[2] * 3 * np.sqrt(5 / (12 * np.pi)) 
        H[7] = la[2] * 1.5 * np.sqrt(5 / (12 * np.pi)) 
        H[8] = la[2] * 3 * np.sqrt(5 / (12 * np.pi)) 



        self.register_buffer('H',torch.Tensor(H))
        #self.register_buffer('mean_albedo',mean_albedo)
        #self.register_buffer('mask_albedo',mask_albedo)
        self.register_buffer('tris', index-1)

            
    def forward(self, ori_images,proj_face,albedo, illum,proxy = False, conf=None):
        # albedo, albedo_s = albedo
        batch_size = ori_images.shape[0]         
        ori_images=ori_images.permute(0,2,3,1).view(batch_size,config.img_h,config.img_w,3) #(batch, 3, 112, 96) -> (batch, 112*96, 3)
        verts = proj_face*1
        verts = verts.reshape(batch_size,3, -1).transpose(1,2)
        verts[:,:,0] = verts[:,:,0]-config.img_h/2
        verts[:,:,1] = config.img_h/2-verts[:,:,1]        
        if config.use_ConvTex:
            #self.mean_albedo = self.mean_albedo.permute(1,2,0)
            #albedo = albedo.permute(0,2,3,1)
            #print(albedo.shape)
            textures = Textures(maps = albedo,verts_uvs=torch.Tensor(config.uv_coords).expand(verts.shape[0], config.uv_coords.shape[0], config.uv_coords.shape[1]),faces_uvs=self.tris.long().expand(verts.shape[0], self.tris.shape[0], self.tris.shape[1])).to(verts.device)
            meshes =  Meshes(verts=verts, faces=self.tris.expand(verts.shape[0],  self.tris.shape[0], self.tris.shape[1]),textures=textures)

            # if not proxy:
                #mean_albedo = self.mean_albedo.expand(batch_size,albedo.shape[1],albedo.shape[2],albedo.shape[3])
                #loss_reg = config.weight_albedo_reg *  ((albedo-mean_albedo)**2).mean()
                # mask_albedo = self.mask_albedo.view(1,self.mask_albedo.shape[0],self.mask_albedo.shape[1],1).expand(batch_size,self.mask_albedo.shape[0],self.mask_albedo.shape[1],3)
                # #face_albedo = mask_albedo*albedo
                # #mean_face_albedo = (face_albedo.view(batch_size,-1,3).sum(1) / self.mask_albedo_npixels).view(batch_size,1,1,3)
                
                # #texels = interpolate_texture_map(fragments, meshes)
                # face_diff = torch.zeros_like(albedo).to(albedo.device)
                # face_albedo_l = (albedo[:,:,:-1,:] - albedo[:,:,1:,:])**2
                # face_albedo_r = (albedo[:,:,1:,:] - albedo[:,:,:-1,:])**2
                # face_albedo_u = (albedo[:,:-1,:,:] - albedo[:,1:,:,:])**2
                # face_albedo_d = (albedo[:,1:,:,:] - albedo[:,:-1,:,:])**2
                # face_diff[:,:,:-1,:] += face_albedo_l
                # face_diff[:,:,1:,:] += face_albedo_r
                # face_diff[:,:-1,:,:] += face_albedo_u
                # face_diff[:,1:,:,:] += face_albedo_d
                
                # smooth_loss = config.weight_albedo_smooth*(face_diff/4 * mask_albedo).sum() / self.mask_albedo_npixels
                
                #smooth_loss = config.weight_albedo_smooth*((face_albedo - mean_face_albedo)**2 * mask_albedo).sum() / self.mask_albedo_npixels
                # if not config.use_flip:
                #     flip_loss = config.weight_albedo_flip*(torch.abs(torch.flip(albedo,[2]) - albedo)* mask_albedo).sum() / self.mask_albedo_npixels
                
            
        else:
            mu_tex = torch.Tensor(config.colors).expand(batch_size,albedo.shape[1],3).to(verts.device)
            albedo = albedo.reshape(batch_size,-1, 3)
            meshes =  Meshes(verts=verts, faces=self.tris.expand(verts.shape[0],  self.tris.shape[0], self.tris.shape[1]),textures=Textures(verts_rgb=albedo))
            #loss_reg = config.weight_albedo_reg *  ((albedo- mu_tex)**2).mean()

        R, T = look_at_view_transform(200, 0, 0)
        cameras = SfMOrthographicCameras(focal_length=2.0/config.img_h,R=R, T=T).to(meshes.device)
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=self.raster_settings
        )


        fragments = rasterizer(meshes)
        
        vis_mask = self._vis_mask(fragments, meshes)[:, :config.img_h,:config.img_w].detach() 
        if config.use_ConvTex:
            colors,illum_colors,ver_color = self._shader_tex(fragments, meshes,illum)
            #ver_color = softmax_rgb_blend(ver_color, fragments,BlendParams(),zfar=500)
        else:
            colors,illum_colors,ver_color = self._shader(fragments, meshes,illum)

        images = softmax_rgb_blend(colors, fragments,BlendParams(),zfar=500).clamp(min=0,max=1)#.sigmoid()
        illum_images = softmax_rgb_blend(illum_colors, fragments,BlendParams(),zfar=500)

        #images[...,:3] = torch.sigmoid(images[...,:3])
        images = images[:, :config.img_h,:config.img_w , :]
        illum_images = illum_images[:, :config.img_h,:config.img_w , :]
        images[:, :config.img_h,:config.img_w , 3] = vis_mask.int()
        #mask = images[...,3].unsqueeze(-1)
        if config.use_confidence_map:
            loss = self.photometric_loss(images[...,:3],ori_images[...,:3],conf_sigma=conf, mask=vis_mask) 
        else:
            loss = self.photometric_loss(images[...,:3],ori_images[...,:3], mask=vis_mask) 

        # mse = torch.abs(images[...,:3]- ori_images[...,:3])

        # loss = torch.where(vis_mask.unsqueeze(-1), mse, ori_images[...,:3]*0).sum() /(vis_mask.sum()+1e-12)
        #if not proxy:
            #loss += smooth_loss

        return  loss, images,illum_images,ver_color

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            conf_sigma = conf_sigma.permute(0,2,3,1)
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def _shader_tex(self,fragments,meshes,illum=None):
        texels = interpolate_texture_map(fragments, meshes)
        
        #texels = torch.sigmoid(texels)
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )
        N, H, W, K, C = pixel_normals.shape
        pixel_normals = pixel_normals.view(N*H*W*K,C)
        pn = N*H*W*K
        illum = illum.view(-1,9,3).unsqueeze(1).expand(N, int(pn/N), 9, 3).reshape(-1,9,3)
        pixel_normals = pixel_normals*1
        h1 = self.H[0].expand(pn).to(vertex_normals.device)
        h2 = self.H[1] * pixel_normals[:, 2]
        h3 = self.H[2] * pixel_normals[:, 0]
        h4 = self.H[3] * pixel_normals[:, 1]
        h5 = self.H[4] * (3 * (pixel_normals[:, 2] ** 2) - 1)
        h6 = self.H[5] * (pixel_normals[:, 0] * pixel_normals[:, 2])
        h7 = self.H[6] * (pixel_normals[:, 1] * pixel_normals[:, 2])
        h8 = self.H[7] * (pixel_normals[:, 0] **2 -  pixel_normals[:, 1] ** 2)
        h9 = self.H[8] * (pixel_normals[:, 0] * pixel_normals[:, 1])
        spherical_har = torch.stack([h1, h2, h3, h4, h5, h6, h7, h8, h9], dim = 1) # [batch, vn, 9]

        spherical_har = spherical_har.unsqueeze(2).expand(pn, 9, 3)
        illum = illum * spherical_har #pn, 9, 3 
        illum = illum.sum(dim = 1) #  pn, 3 

        pix_color =(texels*illum.view(N,H,W,K,C))
        illum_colors = illum.view(N,H,W,K,C)
        return pix_color,illum_colors, None

    

    def _shader(self, fragments, meshes, illum=None):
        #st = time.time()
           
        '''
        v_norm [batch, vn, 3]
        illum [batch, 27]
        
        '''
        # if illum == None:
        #     illum = torch.ones((1,27)).to(config.device)
        faces = meshes.faces_packed() # (F, 3)
        #verts = meshes.verts_packed()
        v_norm = meshes.verts_normals_packed() # (V, 3)
        #print('Cal norm:',time.time() - st)
        batch_size = illum.shape[0]
        
        vertex_colors = meshes.textures.verts_rgb_packed()
        #vert_to_mesh_idx = meshes.verts_packed_to_mesh_idx()
        # print(v_norm.shape)
        #st1 = time.time() - st
        #print('Cal norm:',time.time() - st)
        batch_size = illum.shape[0]
        vn = v_norm.shape[0]
        #illum = illum.view(-1,9,3)
        illum = illum.view(-1,9,3).unsqueeze(1).expand(batch_size, int(vn/batch_size), 9, 3).reshape(-1,9,3)
        # pi = torch.tensor(3.141592653589).float()
        # pi = pi.cuda() if config.use_cuda else pi
        # la = [np.pi, 2*np.pi/3.0, 0.25*np.pi]	
        # 1
        h1 = self.H[0].expand(vn).to(v_norm.device)
        # 2
        h2 = self.H[1] * v_norm[:, 2]
        # 3
        h3 = self.H[2] * v_norm[:, 0]
        # 4
        h4 = self.H[3] * v_norm[:, 1]
        # 5
        h5 = self.H[4] * (3 * (v_norm[:, 2] ** 2) - 1)
        # 6
        h6 = self.H[5] * (v_norm[:, 0] * v_norm[:, 2])
        # 7
        h7 = self.H[6] * (v_norm[:, 1] * v_norm[:, 2])
        # 8d
        h8 = self.H[7] * (v_norm[:, 0] **2 -  v_norm[:, 1] ** 2)
        # 9
        h9 = self.H[8] * (v_norm[:, 0] * v_norm[:, 1])
        spherical_har = torch.stack([h1, h2, h3, h4, h5, h6, h7, h8, h9], dim = 1) # [batch, vn, 9]
        spherical_har = spherical_har.unsqueeze(2).expand(vn, 9, 3)
        #print(illum.shape, spherical_har.shape)
        illum = illum * spherical_har #batch_size, vn, 9, 3 
        #print(illum.shape, albedo.shape)
        illum = illum.sum(dim = 1) #  (V, 3)
        #print(illum.shape, albedo.shape)
        # print(vertex_colors,illum)
        # print(vertex_colors.shape,illum.shape)
        ver_color =(vertex_colors*illum).squeeze()
        #ver_color =vertex_colors.squeeze()
        

        #faces = faces
        #verts_colors_shaded = ver_color
        # print(ver_color.shape)
        # print(faces.shape)
        # ver_color = ver_color.cpu()
        illum_colors = illum[faces]
        face_colors = ver_color[faces]
       
        #st1 = time.time() - st
        #print('Apply Lighting:',st1)
        # print(face_colors)
        colors = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_colors).clamp(min=0,max=1)
        illum_colors = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, illum_colors).clamp(min=0,max=1)

        #st2 = time.time() - st - st1
        #print('Frame Time:',st2)
        # print(colors.shape)
        # print(colors)
        return colors,illum_colors,ver_color

    def _vis_mask(self,fragments,meshes):
        N, H, W, K = fragments.pix_to_face.shape

        pix_to_face = fragments.pix_to_face.clone()
        mask = pix_to_face == -1

        pix_to_face[mask] = 0
        f_norm = meshes.faces_normals_packed()
        
        vis_faces = f_norm[:,2]>=0
        idx = pix_to_face.view(N * H * W * K).expand(N * H * W * K)
        pixel_face_vals = vis_faces.gather(0, idx).view(N, H, W, K)
        pixel_face_vals[mask] = False
        pixel_face_vals = pixel_face_vals.any(-1)
        return pixel_face_vals

    # def get_pho_loss(self, ori_images,proj_face,albedo, illum):
    #     #st = time.time()
        
    
if __name__ == '__main__':
    BFMF_batch()
    