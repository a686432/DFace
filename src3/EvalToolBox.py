import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import sys
from torchvision import datasets, transforms
import net
import torch
import argparse
import data_loader
#from getDepth import get_depth_rmse
from PIL import Image
import config
from glob import glob
from BFMN_batch import BFMN_batch
from BFMG_batch import BFMG_batch
import matplotlib.pyplot as plt

# Data structures and functions for rendering
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
    HardPhongShader
)

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))




class EvalToolBox(object):

    def __init__(self):
    
        self.evalset_micc = data_loader.MICCDataSet(root=config.micc_image_root, filelist=config.micc_filelist, transform=config.transform_eval_fs)
        #self.evalset_bu3def = data_loader.BU3DEFDataSet(img_root=config.bu_image_root, target_root=config.bu_obj_root, filelist=config.bu_filelist, transform=config.transform_eval_fs)
        self.eval_loader_micc = torch.utils.data.DataLoader(self.evalset_micc, batch_size=40, shuffle=False, num_workers=1)
        #self.eval_loader_bu3def  = torch.utils.data.DataLoader(self.evalset_bu3def, batch_size=1, shuffle=False, num_workers=1)
        self.errors = []
        self.bfm=np.load("../propressing/bfmz.npz")
        self.w_shape = self.bfm['w_shape'][:,0:199]
        self.mu_shape = self.bfm['mu_shape'].reshape(-1)
        #self.bfma = BFMF_batch().cuda()


    def save_mesh(self,mesh_vertices, mesh_face, output_name):
        num_vertices = mesh_vertices.shape[0]
        num_face = mesh_face.shape[0]
        f = open(output_name, "w")
        f.write("# {} vertices, {} faces\n".format(num_vertices, num_face))
        for i in range(num_vertices):
            f.write("v {} {} {}\n".format(mesh_vertices[i][0], mesh_vertices[i][1], mesh_vertices[i][2]))
        for i in range(num_face):
            f.write("f {} {} {}\n".format(mesh_face[i][0], mesh_face[i][1], mesh_face[i][2]))
        f.close()

    def read_mesh(self,file_name):
        f = open(file_name, "r") 
        lines=f.readlines()
        f.close()
        vertices=[]
        faces=[]
        for line in lines:
            words = line.split(" ")
            if words[0]=="v":
                ver=np.zeros(3)
                ver[:]=float(words[1]),float(words[2]),float(words[3])
                vertices.append(ver)
            if words[0]=="f":
                face=np.zeros(3,dtype=int)
                face[:]=int(words[1].split("/")[0]),int(words[2].split("/")[0]),int(words[3].split("/")[0])
                faces.append(face)
        vertices=np.array(vertices)
        faces=np.array(faces)
        return vertices,faces

    def crop_radius(self,mesh_vertices, mesh_faces):
  
        def radius(v1,v2):
    	    return np.sqrt(((v1[0:3]-v2[0:3])**2).sum())

        zmax_index=np.argmax(mesh_vertices[:,2])

        vertices=np.zeros((mesh_vertices.shape[0],5))
        vertices[:,0:3]=mesh_vertices
        vertices[:,3] = 1
        vertices[:,4] = -1
        faces = np.zeros((mesh_faces.shape[0],4),dtype=int)
        faces[:,0:3] = mesh_faces
        faces[:,3]= 1

        mesh_vertices = vertices
        mesh_faces = faces

        origin=mesh_vertices[zmax_index]
        target_vertices=[]
        target_faces=[]
        #face=np.zeros(mesh_vertices.shape())

        for idx,vertex in enumerate(mesh_vertices):
            if radius(vertex,origin)>95: 
                mesh_vertices[idx,3] = 0
            else:
                target_vertices.append(mesh_vertices[idx,0:3])
        
        i=0
        for idx,vertex in enumerate(mesh_vertices):
            if mesh_vertices[idx,3] == 1:
                i+=1
                mesh_vertices[idx,4] = i
        for idx,face in enumerate(mesh_faces):
            #print(face[0:3])
            if (mesh_vertices[face[0:3]-1] == 0).any():
                mesh_faces[idx,3]=0
            else:
                mesh_faces[idx,0:3]=mesh_vertices[mesh_faces[idx,0:3]-1,4]
                target_faces.append(mesh_faces[idx,0:3])
        target_vertices=np.array(target_vertices)
        target_faces=np.array(target_faces)

        return target_vertices, target_faces
    

    def simple_crop(self,v):
        def radius(v1,v2):
    	    return np.sqrt(((v1[0:3]-v2[0:3])**2).sum())

        zmax_index=np.argmax(v[:,2])
        origin=v[zmax_index]

        target_vertices=[]
        for idx,vertex in enumerate(v):
            if radius(vertex,origin)<=95: 
                target_vertices.append(vertex)
        
        target_vertices=np.array(target_vertices)
        return target_vertices


        # save_mesh(target_vertices, target_faces, output_name)    
    def vertices2obj(self,vertices_group, labels=None, imname=None):
        for idx, vertices in enumerate(vertices_group):
            #print(imname)
            imname=os.path.splitext(imname[0].split('/')[-1])[0]
    
            #print(imname)
            num_vertex = vertices.shape[0]
            outdir = os.path.join(config.save_tmp_dir,str(labels[idx]))
            filename = os.path.join(outdir,imname+'.obj') 
            if not os.path.exists(os.path.split(filename)[0]):
                os.mkdir(os.path.split(filename)[0])
            fobj=open(filename,'w+')
            fobj.write("# {} vertices, {} faces\n".format(num_vertex, config.num_index))
            #print(vertices[0])
            for i in range(num_vertex):
                fobj.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
            for i in range(config.num_index):
                fobj.write("f {} {} {}\n".format(config.index[i][0]+1, config.index[i][2]+1, config.index[i][1]+1))
            fobj.close()

    
    def write_obj(self, filename, vertices):
        outdir = os.path.dirname(filename)
        # if not os.path.exists(outdir):
        #     os.mkdir(outdir)
        num_vertex = vertices.shape[0]
        fobj=open(filename,'w+')
        fobj.write("# {} vertices, {} faces\n".format(num_vertex, config.num_index))
        #print(vertices[0])
        for i in range(num_vertex):
            fobj.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(config.num_index):
            fobj.write("f {} {} {}\n".format(config.index[i][0]+1, config.index[i][2]+1, config.index[i][1]+1))
        fobj.close()



    def get_mesh_from_model(self, model, imgname):
        print("evaluting "+str(imgname) + "...")
        model.eval()
        model.cuda()
        #total_loss=0
        norm=0
        img=Image.open(imgname)
        
        #label = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = self.transform_eval_fs(img)
        img = torch.unsqueeze(img, 0).cuda()
        feat , (param, vertex,  vertex2) = model(img)
        norm= torch.norm(param,dim=1).mean().item()
            #print(norm/(batch_idx+1))
            # print(para) 
        print(vertex.shape)
        self.write_obj("image00320.obj",vertex[0].data.cpu().numpy().reshape(-1,3))
        self.write_obj("image00320_e.obj",vertex2[0].data.cpu().numpy().reshape(-1,3))
            #total_loss+=loss
        #     sys.stdout.write("%d/%d | Loss: %.5f | norm: %.5f\r" 
        #         % (batch_idx+1,len(self.eval_loader_micc),total_loss/(batch_idx+1),norm/(batch_idx+1)))
        # np.save("result.npy",np.array(self.errors))
        # sys.stdout.write("\n")
        # print("Fanished!")


    def get_bu3def_rmse(self, src_root, filelists):
        errors = []
        d_errors = []
        error =0
        d_error = 0
        num = 0
        error_num =0

        with open(filelist,'r') as f:
            files = f.readlines()
        for idx,filename in enumerate(files):

            filename = filename.replace("\n","")
            v1, f1 = self.read_mesh(os.path.join(src_root,filename))


            #label = int(filename.split("/")[0])
            v2, f2 = v1, f1 #self.crop_radius(v1,f1)
            bu_name = os.path.join(self.bu_obj_root,filename).replace("F2D","F3D")
            #print(v2)
            #print(bu_name)
            try:
                v1, f1 = self.read_mesh(bu_name)
                err, v3,f3= self._icp(v1,v2,f2)
                if err>8.0:
                    print("Error Too Large: ", )
                    print(filename)
                    continue
                #self.save_mesh(v3,f3,os.path.join(self.save_tmp_dir,filename))
                #d_err = get_depth_rmse(v1,f1,v3,f3,size=(224,224))
                d_err = 0
                error += err
                d_error += d_err

                num+=1
                if err>4.5:
                    print(filename)
                errors.append(err)
                d_errors.append(d_err)
            except:   
                print(e) 
                print(filename)
                error_num+=1
        
            sys.stdout.write("%d/%d | Error: %.5f | Depth Error: %.5f\r" 
            % (idx+1,len(files),1.0* error/num,1.0* d_error/num))
        sys.stdout.write("\n")
        np.savez(os.path.join(root,'rmse.npz'), error=errors, derror=d_errors, err_num=error_num)
        print("Error_num: ",error_num)
        print('mean:',np.mean(errors))
        print('std:',np.std(errors))
        pass

    
    def get_mesh_from_param(self, model,param_file=config.save_error_npy, imagefile=config.save_error_file, out_root=config.save_tmp_dir):
        params = np.load(param_file)
        f = open(imagefile,'r')
        #######################################################
        # g_model = np.load(config.g_model_path)
        # #x = torch.Tensor(f_model['x'])
        # #print(x.norm(dim=1).shape)
        # mu_shape = g_model['mu']
        # w_shape = g_model['w_shape']
        # index = g_model['faces']
        # print(mu_shape.shape)
        # print()
        # mu_tensor = mu_shape_tensor + mu_expression_tensor
        # exp_ev_tensor = torch.Tensor(bfm['exp_ev'])
        ################################################

        # bfm=np.load("../propressing/bfmz.npz")
        # w_shape = bfm['w_shape'][:,0:99]
        # mu_shape = bfm['mu_shape'].reshape(-1)
        
        # w_expression = bfm['w_expression'][:,0:29]))
        # mu_expression =torch.Tensorbfm['mu_expression'].reshape(-1))
        
        for param in params:
            param = torch.Tensor(param[0:199])
            face_shape = model.module.get_shape_obj(param).reshape(-1,3)
            # face_shape = (mu_shape + param @ w_shape ).reshape(-1,3)
            filepath = f.readline().replace('\n',"")
            identify = int(filepath.split('/')[-2])
            filename = filepath.split('/')[-1][:-3]+"obj"
            outdir = os.path.join(out_root,str(identify))
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            filename = os.path.join(outdir,filename)
            #print(face_shape)
            self.write_obj(filename,face_shape)
        ################################################################

   
    def syn_mesh(self):
        import pickle
        f_model = np.load(config.f_model_path)
        #x = torch.Tensor(f_model['x'])
        #print(x.norm(dim=1).shape)
        mu_shape = f_model['mu']
        w_shape = f_model['w_shape']
        index = f_model['faces']
        # print(mu_shape.shape)
        # print()
        # mu_tensor = mu_shape_tensor + mu_expression_tensor
        # exp_ev_tensor = torch.Tensor(bfm['exp_ev'])
        ################################################

        # bfm=np.load("../propressing/bfmz.npz")
        # w_shape = bfm['w_shape'][:,0:99]
        # mu_shape = bfm['mu_shape'].reshape(-1)
        # f = open(imagefile,'r')
        # w_expression = bfm['w_expression'][:,0:29]))
        # mu_expression =torch.Tensorbfm['mu_expression'].reshape(-1))
        gmm = pickle.load(open('../propressing/gmm.pkl','rb'))
        #gmm_data = np.load('../propressing/gmm.npy')
        params,y = gmm.sample(1000)
        for x in params:
            print(x @ w_shape)
            face_shape = (mu_shape + x @ w_shape).reshape(-1,3)
            #print(face_shape)
            # filepath = f.readline().replace('\n',"")
            # identify = int(filepath.split('/')[-2])
            # filename = filepath.split('/')[-1][:-3]+"obj"
            # outdir = os.path.join(out_root,str(identify))
            # if not os.path.exists(outdir):
            #     os.mkdir(outdir)
            # filename = os.path.join(outdir,filename)
            # #print(face_shape)
            # self.write_obj(filename,face_shape)


    def get_mesh_from_model_single(self, model,basis,image_folder=config.test_align,outdir = config.test_obj):
        print("Extract param...")
        # f_model = np.load(config.f_model_path)
        # mu_shape = f_model['mu']
        # w_shape = f_model['w_shape']
        # index = f_model['faces']
            
        types = ('*.jpg', '*.png')
        image_path_list= []
        for files in types:
            image_path_list.extend(glob(os.path.join(image_folder, files)))
        total_num = len(image_path_list)
        print(total_num)
        # total = 0
        # count = np.zeros(1000)
        #landmarks_map = np.load('landmark.npy')
        for i, image_path in enumerate(image_path_list):
            image = Image.open(image_path)
            data = config.transform_eval_fs(image)
            data = data.unsqueeze(0).cuda()
            _, pred_shape, _, pred_camera_exp = model(data)
            param = torch.cat((pred_shape, pred_camera_exp),dim=1)
            shape_para = pred_shape[:, 0:199]
            exp_para = pred_camera_exp[:, 7:36]
            camera_para = pred_camera_exp[:, 0:7]
            # print(pred_camera_exp[:, 4:7])
            exp_face, rotated, scaled = basis.module(pred_shape,exp_para,camera_para)
            # print(scaled)
            scaled_raw = scaled.clone().reshape(3, -1).t()
            scaled_raw[:,0] = scaled[:,0]-56
            scaled_raw[:,1] = 56-scaled[:,1]
            # print(scaled_raw)


            pred_face = basis.module.get_shape_obj(shape_para)
            #exp_face = basis.module.get_exp_obj(shape_para,exp_para)
            
            face_shape = pred_face.squeeze().reshape(-1,3)#.data.cpu().numpy()
            exp_face = exp_face.squeeze().reshape(-1,3).data.cpu().numpy()
            rotated = rotated.squeeze().reshape(3, -1).t()
            # scaled_mesh = scaled.squeeze().t()
            scaled = scaled.squeeze()[:2,:].t().unsqueeze(0)
            scaled[:,:,0] = scaled[:,:,0]/96
            scaled[:,:,1] = 1-scaled[:,:,1]/112
            faces = torch.Tensor(config.index.astype(int))
            tex_rgb = torch.ones_like(rotated).unsqueeze(0)
            texture_img = config.transform_raw_img(image).unsqueeze(0).transpose(1,3).transpose(2,1)
            #print(scaled_raw)
            textures = Textures(maps = texture_img,verts_uvs=scaled,faces_uvs=faces.long().unsqueeze(0))
            mesh =  Meshes(verts=[face_shape], faces=[faces], textures = textures).to(config.device)
            normals = mesh.verts_normals_padded()
            print(normals)

            ##########################################################################
            # Initialize an OpenGL perspective camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
            #print(rotated)
            R, T = look_at_view_transform(100, 0, 0) 
            #print(R,T)
            # principal_point = torch.Tensor[1.0/56,1.0/56]
            cameras = SfMOrthographicCameras(device=config.device,focal_length=1.0/56,R=R, T=T)
            #cameras = SfMOrthographicCameras(device=config.device,focal_length=1.0/56,R=R, T=T)
            # Define the settings for rasterization and shading. Here we set the output image to be of size
            # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
            # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
            # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
            # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
            # the difference between naive and coarse-to-fine rasterization. 
            raster_settings = RasterizationSettings(
                image_size=112, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
                bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
                max_faces_per_bin = None  # this setting is for coarse rasterization
            )

            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
            # -z direction. 
            lights = PointLights(device=config.device, location=[[0.0, 0.0, 3.0]])

            # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
            # apply the Phong lighting model
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=TexturedSoftPhongShader(
                    device=config.device, 
                    cameras=cameras,
                    lights=None
                )
            )


            # ## 3. Render the mesh

            # The light is in front of the object so it is bright and the image has specular highlights.


           
            ##########################################################################

            # pose_exp = pose_exp.squeeze().data.cpu().numpy()
            # R = Q2R(pose_exp[0:4])
            # res.append((param.data.cpu().numpy(),imname))
            # norm.append(torch.norm(param,dim=1).mean().item())
            
            # self.vertices2obj(vertex2.data.cpu().numpy().reshape(len(vertex2),-1,3),target.data.cpu().numpy(), imname=imname)
            # face_shape = (mu_shape + param @ w_shape).reshape(-1,3)
            #filepath = image_path.replace('\n',"")
            #identify = int(filepath.split('/')[-2])
            rotated = rotated.data.cpu().numpy()
            filename = image_path.split('/')[-1][:-3]+"obj"
            filename_exp = image_path.split('/')[-1][:-4]+"_e.obj"
            filename_all = image_path.split('/')[-1][:-4]+"_a.obj"
            filename_render = image_path.split('/')[-1][:-4]+".png"
            #outdir = os.path.join(out_root,str(identify))
            # if not os.path.exists(outdir):
            #     os.mkdir(outdir)
            filename = os.path.join(outdir,filename)
            filename_exp = os.path.join(outdir,filename_exp)
            filename_all = os.path.join(outdir,filename_all)
            filename_render = os.path.join(outdir,filename_render)


            images = renderer(mesh)
            #print(images)
            images = images[0, :112,:96 , :]
            mask = images[...,3].unsqueeze(-1)
            #mask
            out_img = torch.where(mask==0, texture_img[0].to(config.device), images[...,:3]) 
            
            #print(texture_img,out_img.shape)
            plt.figure(figsize=(10, 10))
            #print("saving figure!")
            plt.imsave(filename_render,out_img[ ..., :3].data.cpu().numpy())
            plt.grid("off")
            plt.axis("off")
            #print(face_shape)
            # self.write_obj(filename,face_shape)
            # self.write_obj(filename_exp,exp_face)
            self.write_obj(filename_all,rotated)


    def get_rmse_from_param(self, param_file= config.save_name_obj, imagefile=config.save_error_file, out_root=config.save_tmp_dir):
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures import as_completed
        import threading
        import tqdm

        

        objs = np.load(param_file)
        
        f = open(imagefile,'r')
        files = f.readlines()
        f.close()
        # # w_expression = bfm['w_expression'][:,0:29]))
        # # mu_expression =torch.Tensorbfm['mu_expression'].reshape(-1))
        results = []
        print(files)
        executor = ProcessPoolExecutor(max_workers=4)  
        for idx,param in tqdm.tqdm(enumerate(objs),total=len(objs)):
            filepath =files[idx].replace('\n',"")
            results.append(executor.submit(func,param,filepath))
            #print(executor.submit(func,param,filepath).result())
        errors = []
        files = []
        for future in as_completed(results):
            errors.append(future.result()[0])
            #print(future.result()[0])
            files.append(future.result()[1])
        errors = np.array(errors)
        print(errors.shape)
        
        file_err = os.path.join(config.save_tmp_dir,'errors.npy')
        print(file_err)
        file_f = os.path.join(config.save_tmp_dir,'file.txt')
        np.save(file_err,errors)
        f = open(file_f,'w')

        for file in files:
            f.write(file)
            f.write('\n')
        f.close()



    def get_rmse_from_obj(self, objfile=config.micc_filelist, out_root=config.save_tmp_dir):
        #params = np.load(param_file)

        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures import as_completed
        import threading
        import tqdm

        
        f = open(objfile,'r')
        files = f.readlines()
        f.close()
        root = "/home/jdq/github/RingNet/RingNet_output/neutral_mesh"
        # # w_expression = bfm['w_expression'][:,0:29]))
        # # mu_expression =torch.Tensorbfm['mu_expression'].reshape(-1))
        results = []
        executor = ProcessPoolExecutor(max_workers=16)  
        
        for idx,param in tqdm.tqdm(enumerate(files),total=len(files)):
            filepath =files[idx].replace('\n',"")[:-4]+".obj"
            filepath = os.path.join(root, filepath)
            results.append(executor.submit(func2,filepath))
            #print(executor.submit(func2,filepath).result())
        # exit()
        errors = []
        files = []
        for future in as_completed(results):
            errors.append(future.result()[0])
            files.append(future.result()[1])
        errors = np.array(errors)
        file_err = os.path.join(config.save_tmp_dir,'errors.npy')
        file_f = os.path.join(config.save_tmp_dir,'file.txt')
        np.save(file_err,errors)
        f = open(file_f,'w')

        for file in files:
            f.write(file)
            f.write('\n')
        f.close()


    def lfw_acc_param():
        pass

        
    def get_param_from_model(self,model,basis,save_name = config.save_error_npy, file_index = config.save_error_file):
        print("Extract param...")
        
        outdir = os.path.dirname(save_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        model.eval()
        # total_loss=0
        res = []
        norm=[]
        objs = []
        params = np.zeros(shape=(0,235)) 
        mesh = np.zeros(shape=(0,53215*3))
        for batch_idx, (data,target,imname) in enumerate(self.eval_loader_micc):
            data,target =  data.cuda(), target.cuda()
            #print(imname)
            #feat , param, vertex, _ = model(data)
            _, pred_shape, _, pred_camera_exp = model(data)
            param = torch.cat((pred_shape, pred_camera_exp),dim=1)
            shape_para = pred_shape[:, 0:199]
            exp_para = pred_camera_exp[:, 7:36]
            camera_para = pred_camera_exp[:, 0:7]
            pred_face = basis.module.get_shape_obj(shape_para)
            objs.append(pred_face.data.cpu().numpy())
            res.append((param.data.cpu().numpy(),imname))
            norm.append(torch.norm(pred_shape,dim=1).mean().item())

            sys.stdout.write("%d/%d  | norm: %.5f\r" 
                % (batch_idx+1,len(self.eval_loader_micc),sum(norm)/len(norm)))
        print('finish regressing')
        f = open(file_index,'w+')
        for param, imname in res:
            params = np.concatenate((params, param),axis=0)
            for im in imname:
                f.write(im+'\n')
        f.close() 
        for idx,obj in enumerate(objs):
            mesh = np.concatenate((mesh, obj),axis=0)
            sys.stdout.write("%d/%d \r" % (idx+1,len(objs)))
        sys.stdout.write("\n")
        norm = np.array(norm)
        # params = np.array(params)
        np.save(save_name,params) 
        np.save(config.save_name_obj,mesh)
        print("mean:",norm.mean(),"std:",norm.std())
        print("Fanished!")

    def get_bu3def_obj(self,model, save_tmp = False):
        print("evaluting BU3DEF...")
        model.eval()
        total_loss=0
        norm=[]
        for batch_idx, (data,target,imname) in enumerate(self.eval_loader_bu3def):
            data,target =  data.cuda(), target.cuda()
            feat , (param, vertex, vertex2) = model(data)
            norm.append(torch.norm(param,dim=1).mean().item())
            self.vertices2obj(vertex2.data.cpu().numpy().reshape(len(vertex2),-1,3),target.data.cpu().numpy(), imname=imname)
            sys.stdout.write("%d/%d  | norm: %.5f\r" 
                % (batch_idx+1,len(self.eval_loader_micc),sum(norm)/len(norm)))
        #np.save("result.npy",np.array(self.errors))
        sys.stdout.write("\n")
        norm = np.array(norm)
        print("mean:",norm.mean(),"std:",norm.std())
        print("Fanished!")

    def get_micc_obj(self, model, save_tmp = False):


        print("Get MICC obj...")
        model.eval()
        total_loss=0
        norm=[]
        # ver
        for batch_idx, (data, target, image_name) in enumerate(self.eval_loader_micc):
            data ,target=  data.cuda(),target.cuda()
            feat , param, vertex, _ = model(data)
            #print(torch.norm(param,dim=1).mean().item())
            norm.append(torch.norm(param,dim=1).mean().item())
            # ver = vertex.data.cpu().numpy()

            self.vertices2obj(vertex.data.cpu().numpy().reshape(len(vertex),-1,3),target.data.cpu().numpy(),image_name)
            sys.stdout.write("%d/%d  norm: %.5f\r" 
                % (batch_idx+1,len(self.eval_loader_micc),sum(norm)/len(norm)))
        np.save("result.npy",np.array(self.errors))
        sys.stdout.write("\n")
        norm = np.array(norm)
        print("mean:",norm.mean(),"std:",norm.std())
        print("Fanished!")

    def get_micc_rmse_from_obj(self,root,filelist):
        errors = []
        d_errors = []
        error =0
        d_error = 0
        num = 0
        error_num =0
        with open(filelist,'r') as f:
            files = f.readlines()
        #print(files)
        for idx,filename in enumerate(files):
            #print(filename)
            filename = filename[:-1]
            v1, f1 = self.read_mesh(os.path.join(root,filename))
            
            label = int(filename.split("/")[0])
            v2, f2 = v1, f1 #self.crop_radius(v1,f1)
            micc_name = os.path.join(config.micc_obj_root,filename.split("/")[0].zfill(2)+'.obj')
            #print(v2)
            #print(micc_name)
            try:
                
                v1, f1 = self.read_mesh(micc_name)
                err, v3,f3= self._icp(v1,v2,f2)
                # self.save_mesh(v3,f3,"../1.obj")
                # self.save_mesh(v1,f1,"../2.obj")
                if err>8.0:
                    print("Error Too Large: ", )
                    print(filename)
                #self.save_mesh(v3,f3,os.path.join(self.save_tmp_dir,filename))
                #d_err = get_depth_rmse(v1,f1,v3,f3,size=(224,224))
                d_err = 0
                error += err
                d_error += d_err

                num+=1
                if err>4.5:
                    print(filename)
                    #print(err)
                errors.append(err)
                d_errors.append(d_err)
            except:   
                print(e) 
                print(filename)
                error_num+=1
        
            sys.stdout.write("%d/%d | Error: %.5f | Depth Error: %.5f\r" 
            % (idx+1,len(files),1.0* error/num,1.0* d_error/num))
        sys.stdout.write("\n")
        np.savez(os.path.join(root,'rmse.npz'), error=errors, derror=d_errors, err_num=error_num)
        print("Error_num: ",error_num)
        print('mean:',np.mean(errors))
        print('std:',np.std(errors))
        #np.save()

    def pooling_rmse(self,root,filelist):

        errors = []
        d_errors = []
        error =0
        d_error = 0
        num = 0
        error_num =0


        with open(filelist,'r') as f:
            filelists = f.readlines()
        iddirs = os.listdir(root)
        #print(iddirs)
        for idx, iddir in enumerate(iddirs):
            c_dir = os.path.join(root,iddir)
            if os.path.isdir(c_dir):
                label = int(iddir)
                
                objs = os.listdir(c_dir)
                vertices = []
                for idx2, obj in enumerate(objs):
                    objname = os.path.join(iddir,obj)
                    if objname+"\n" in filelists:
                        v ,f2 = self.read_mesh(os.path.join(root,objname))
                        vertices.append(v)
                        sys.stdout.write("%d/%d \r" 
                        % (idx2+1,len(objs)))
                sys.stdout.write("\n")
                v2 = np.array(vertices).mean(axis=0)
                micc_name = os.path.join(config.micc_obj_root,str(iddir).zfill(2)+'.obj')
                print("Finish Pooling: " + str(iddir) )
                try:
                    v1, f1 = self.read_mesh(micc_name)
                    err, v3,f3= self._icp(v1,v2,f2)
                    #self.save_mesh(v3,f3,os.path.join(self.save_tmp_dir,filename))
                    

                    # d_err = get_depth_rmse(v1,f1,v3,f3,size=(224,224))
                    error += err
                    # d_error += d_err
                    num+=1
                    if err>4.5:
                        print(iddir)
                        #print(err)
                    errors.append(err)
                    d_errors.append(d_err)
                except:   
                    print(iddir)
                    error_num+=1
                sys.stdout.write("%d/%d | Error: %.5f | Depth Error: %.5f\r" 
                % (idx+1,len(iddirs),1.0* error/num,1.0* d_error/num))
                sys.stdout.write("\n")
        np.savez(os.path.join(root,'pool_rmse.npz'), error=errors, derror=d_errors, err_num=error_num)
        print("Error_num: ",error_num)
        print('mean:',np.mean(errors))
        print('std:',np.std(errors))

    def get_mesh_from_npy(self,obj_file=config.save_name_obj,imagefile=config.save_error_file):
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures import as_completed
        import threading
        import tqdm
        print("stage3")
        objs = np.load(obj_file)
        
        f = open(imagefile,'r')
        files = f.readlines()
        f.close()
        # # w_expression = bfm['w_expression'][:,0:29]))
        # # mu_expression =torch.Tensorbfm['mu_expression'].reshape(-1))
        results = []
        #print(files)
        executor = ProcessPoolExecutor(max_workers=4)  
        for idx,obj in tqdm.tqdm(enumerate(objs),total=len(objs)):
            filepath =files[idx].replace('\n',"")
            results.append(executor.submit(func_obj,obj,filepath))
            #print(executor.submit(func,param,filepath).result())
        # errors = []
        # files = []
        # for future in as_completed(results):
        #     errors.append(future.result()[0])
        #     #print(future.result()[0])
        #     files.append(future.result()[1])
        # errors = np.array(errors)
        # print(errors.shape)
        
        # file_err = os.path.join(config.save_tmp_dir,'errors.npy')
        # print(file_err)
        # file_f = os.path.join(config.save_tmp_dir,'file.txt')
        # np.save(file_err,errors)
        # f = open(file_f,'w')

        # for file in files:
        #     f.write(file)
        #     f.write('\n')
        # f.close()


                    
                        

                  

def icp(v1, v2):
    
    def P2sRt(P):
        ''' decompositing camera matrix P. 
        Args: 
            P: (3, 4). Affine Camera Matrix.
        Returns:
            s: scale factor.
            R: (3, 3). rotation matrix.
            t3d: (3,). 3d translation. 
        '''
        T = P[:3, 3]
        R1 = P[0:1, :3]
        R2 = P[1:2, :3]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
        r1 = R1/np.linalg.norm(R1)
        r2 = R2/np.linalg.norm(R2)
        r3 = np.cross(r1, r2)
        R = np.concatenate((r1, r2, r3), 0)
        return s, R, T


    def compute_similarity_transform(points_to_transform,points_static):
        p0 = np.copy(points_static).T
        p1 = np.copy(points_to_transform).T
        t0 = -np.mean(p0, axis=1).reshape(3,1)
        t1 = -np.mean(p1, axis=1).reshape(3,1)
        #t_final = t1 -t0
        
        p0c = p0+t0
        p1c = p1+t1

        covariance_matrix = p0c.dot(p1c.T)
        U,S,V = np.linalg.svd(covariance_matrix)
        R = U.dot(V)
        if np.linalg.det(R) < 0:
            R[:,2] *= -1

        rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
        rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

        ## ours 
        # rms_d0 = np.mean(np.linalg.norm(p0c, axis=0))
        # rms_d1 = np.mean(np.linalg.norm(p1c, axis=0))
            
        t_final = -t0 - np.dot(R,-t1)
        s = (rms_d0/rms_d1)
        # P = np.c_[s*np.eye(3).dot(R), t_final]
        # print(P)
        return s,R,t_final
    
    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        # assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    vertices1 = v1
    vertices2 = v2

    zmax_index1=np.argmax(vertices1[:,2]) 
    origin1=vertices1[zmax_index1]

    zmax_index2=np.argmax(vertices2[:,2]) 
    origin2=vertices2[zmax_index2]
    
    vertices1=vertices1[:,0:3]
    vertices2=vertices2[:,0:3]-origin2[0:3]+origin1[0:3]
    
    prev_error = 0

    
    # neigh = NearestNeighbors(n_neighbors=1)

    for i in range(1000):
        #print('aaaaa')
        distances2, indices2=nearest_neighbor(vertices1,vertices2)
        #print('bbbbb')
        s,R,t2 = compute_similarity_transform(vertices2[indices2], vertices1)
        #print('ccccc')
        #s,R,t2 = P2sRt(P)
        centroid = np.mean(vertices2, axis=0)
        vertices2 = (np.dot(R,vertices2.T)*s+t2).T
        # check error
        mean_error = np.sqrt(np.mean(distances2**2))
        #print('ME:',mean_error)
        if np.abs(prev_error - mean_error) < 0.001:
            break
        if i>10 and prev_error < mean_error:
            break
        prev_error = mean_error
    distances, indices = nearest_neighbor(vertices1, vertices2)
    mean_error = np.sqrt(np.mean(distances**2))
    return mean_error, vertices2

    def test_icp(self,src_file,dst_file):
        v1,f1= self.read_mesh(src_file)
        v2,f2= self.read_mesh(dst_file)
        err,v2,f2 = self._icp(v1, v2,f2)
        self.save_mesh(v2,f2,"../output.obj")


def read_mesh(file_name):
        f = open(file_name, "r") 
        lines=f.readlines()
        f.close()
        vertices=[]
        faces=[]
        for line in lines:
            words = line.split(" ")
            if words[0]=="v":
                ver=np.zeros(3)
                ver[:]=float(words[1]),float(words[2]),float(words[3])
                vertices.append(ver)
            if words[0]=="f":
                face=np.zeros(3,dtype=int)
                face[:]=int(words[1].split("/")[0]),int(words[2].split("/")[0]),int(words[3].split("/")[0])
                faces.append(face)
        vertices=np.array(vertices)
        faces=np.array(faces)
        return vertices,faces
    
def func(pred_face,filepath):
    # shape_para = param[:, 0:199]
    # exp_para = param[:, 199+7:199+36]
    # camera_para = param[:, 199+0:199+7]
    # pred_face, face_rotated, face_proj = model(shape_para, exp_para, camera_para)
    #print(1)
    #import time
    #t_begin = time.time()
    #filepath =files[idx].replace('\n',"")
    # face_shape = 
    #face_shape = (mu_shape + w_shape @ (param*1000)).reshape(-1,3)/1000
    pred_face = pred_face.reshape(-1,3)
    #print(pred_face)
    label = int(filepath.split('/')[-2])
    v2, f2 = pred_face, config.index #self.crop_radius(v1,f1)
    micc_name = os.path.join(config.micc_obj_root,str(label).zfill(2)+'.obj')   
    
    v1, f1 = read_mesh(micc_name)
    #print('ssssss')
    #print(v1)
    err, v3= icp(v1,v2)
    if err>8.0:
        print("Error Too Large: ", )
        print(filepath)
    #self.save_mesh(v3,f3,os.path.join(self.save_tmp_dir,filename))
    #d_err = get_depth_rmse(v1,f1,v3,f3,size=(224,224))
    # d_err = 0
    # error += err
    # d_error += d_err

    # num+=1
    if err>4.5:
        print(filepath)
    print(err)
        #print(err)
    # errors.append(err)
    # d_errors.append(d_err)
    # except:   
    #     print(str(e)) 
    #     print(filepath)
    #     #error_num+=1

    #print(time.time() - t_begin)
    return err,filepath      

def func_obj(pred_face,filepath):
    pred_face = pred_face.reshape(-1,3)
    #print(filepath)
    # f = open(filepath,'r')
    # filepath = f.readline().replace('\n',"")
    identify = int(filepath.split('/')[-2])
    filename = filepath.split('/')[-1][:-3]+"obj"
    outdir = os.path.join(config.save_tmp_dir,str(identify))
    #print(outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # count += 1
    #print(count)
    filename = os.path.join(outdir,filename)
    print(filename)
    write_obj(filename,pred_face)

def write_obj(filename, vertices):
    outdir = os.path.dirname(filename)
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    num_vertex = vertices.shape[0]
    fobj=open(filename,'w+')
    fobj.write("# {} vertices, {} faces\n".format(num_vertex, config.num_index))
    #print(vertices[0])
    for i in range(num_vertex):
        fobj.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
    for i in range(config.num_index):
        fobj.write("f {} {} {}\n".format(config.index[i][0]+1, config.index[i][2]+1, config.index[i][1]+1))
    fobj.close()

def save_model(model, path):
    print("Saving...")
    torch.save(model.state_dict(), path)
def load_model(model, path):
    model.load_state_dict(torch.load(path,map_location='cuda:0'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face recognition with CenterLoss')
    parser.add_argument('--gpu', default="0", help="select the gpu")
    parser.add_argument('--stage', default=1, help="select the stage")
    parser.add_argument('--mode', default='micc', help="select the stage")
    args = parser.parse_args()
    count = 0
    # g_model = np.load(config.g_model_path)
    # mu_shape = g_model['mu'].reshape(-1)
    # w_shape = g_model['w_shape']
    # index = g_model['faces']
    e=EvalToolBox()
    print(args.stage)
    if args.stage == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        num_gpus=len(args.gpu.split(","))
        gpu_ids = range(num_gpus)
        print('num of GPU is ' + str(num_gpus))
        print('GPU is ' + str(gpu_ids))
        use_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.set_device(gpu_ids[0])
        config.use_cuda = use_cuda
        config.device = device
        config.device_ids = gpu_ids
        model = net.sphere64a(pretrained=False, stage = 3).to(device)
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        model = model.to(device)
        load_model(model,config.dict_file)
        if 'nonlinear' in config.dict_file:
            bfmn = BFMN_batch()
        else:
            bfmn = BFMG_batch()
        bfmn = torch.nn.DataParallel(bfmn,device_ids=gpu_ids).to(device)
        bfmn = bfmn.to(device)
        if args.mode == 'micc':
            e.get_param_from_model(model,bfmn)
        elif args.mode == 'single':
            e.get_mesh_from_model_single(model,bfmn)

    elif args.stage == 2 or  args.stage =='2':
        e.get_rmse_from_param()
    elif args.stage == 3 or args.stage =='3':
        e.get_mesh_from_npy()
    

   # e.get_mesh_from_param(bfmn)
    
    
    # e.get_rmse_from_param()
 
