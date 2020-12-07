import numpy as np
import config
import matplotlib.pyplot as plt
from numpy.random import randn
# mairezhineng
import os
from PIL import Image
import torch
# from renderer import Renderer
# import neural_renderer as nr
import tqdm
import imageio
#from scipy.interpolate import spline
class VisToolBox(object):
    
    def __init__(self):
        pass


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


    def Q2R(self,Q):
        """
        Quaternion to Rotation Matrix
        
        Q: Tensor (4, )
        return :
            R: 3 * 3 Rotation Matrix
        """
        #print 'Q=', Q
        len = np.sqrt(np.dot(Q, Q))
        
        
        
        if len == 0:
            x = Q[0] * len
            y = Q[1] * len
            z = Q[2] * len
            s = Q[3] * len
        else:
            x = Q[0] / len
            y = Q[1] / len
            z = Q[2] / len
            s = Q[3] / len
        
        #print 'len = ' ,len
        #R.index_put_((torch.from_numpy(np.array([0, 0])), ), 1-2*(y*y+z*z))
        #print R
        
        #R2 = R.sum()
        #R2.backward()
        w1=1-2*(y*y+z*z)
        w2=2*x*y-2*s*z
        w3=2*s*y+2*x*z
        w4=2*x*y+2*s*z
        w5=1-2*(x*x+z*z)
        w6=-2*s*x+2*y*z
        w7=-2*s*y+2*x*z
        w8=2*s*x+2*y*z
        w9=1-2*(x*x+y*y)
        #print w1
        #print w1.view(1)
        
        
        R = np.array([w1, w2, w3, w4, w5, w6, w7, w8, w9]).reshape(3, 3)

        return R

    def Q2R_batch(self,Q):
        """
        batch_version
        Quaternion to Rotation Matrix
        
        Q: Tensor (batch_size, 4)
        return :
            R: 3 * 3 Rotation Matrix
        """
        batch_size = Q.shape[0]
        len = torch.sqrt(torch.sum(Q * Q, dim = 1, keepdim = True)).squeeze(1)
        
        x = Q[:, 0] / len
        y = Q[:, 1] / len
        z = Q[:, 2] / len
        s = Q[:, 3] / len
        
        w1=1-2*(y*y+z*z)
        w2=2*x*y-2*s*z
        w3=2*s*y+2*x*z
        w4=2*x*y+2*s*z
        w5=1-2*(x*x+z*z)
        w6=-2*s*x+2*y*z
        w7=-2*s*y+2*x*z
        w8=2*s*x+2*y*z
        w9=1-2*(x*x+y*y)
        #print w1
        #print w1.view(1)
        R = torch.stack([w1, w2, w3, w4, w5, w6, w7, w8, w9]).transpose(1, 0).reshape(batch_size, 3, 3)
        return R
    def draw_curve_ced(self,datas):
        # data1 = np.load(data_file)
        # data2 = np.load('../data/errors10.npy')
        # datas = [data1,data2]
        #print(data2-data)
        # print("Error Mean: ", data.mean())
        # print("Error Std:", data.std())
        
        #data = randn(75)
        for data_name in datas:
            data = np.load(data_name)
            # print(data)
            print(data_name)
            print("Error Mean: ", data.mean())
            print("Error Std:", data.std())
           
            y,x = np.histogram(data,bins=40,normed=False)
            # print(x)
            y = np.insert(y,0,0)
            # x_new = np.linspace(x[0],x[-1],1000)
            y = np.cumsum(y)/data.shape[0]
            # y_s = spline(x,y,x_new)
            plt.plot(x,y*100,label=data_name[8:-4])
        #print(y_s)
        plt.title("Cumulative Error Curve on Florence",fontsize=12)
        plt.ylabel("Number of Images(%)",fontsize=10)
        plt.xlabel("Root Mean Square Error(mm)",fontsize=10)
        plt.xlim(1,4)
        plt.ylim(0,100)
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        # plt.hist(data)
        # print(data.mean())
    def extract_param(self,model,imgname):
        print("evaluting "+str(imgname) + "...")
        model.eval()
        model.cuda()
        #total_loss=0
        #norm=0
        img=Image.open(imgname)
        
        #label = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = config.transform_eval_fs(img)
        img = torch.unsqueeze(img, 0).cuda()
        _ , param,  _,  up = model(img)
        print(param)
        param_all = torch.cat((param,up),1)
        return param_all.data.cpu().numpy()[0]
        #norm= torch.norm(param,dim=1).mean().item()
            #print(norm/(batch_idx+1))
            # print(para) 
        #print(vertex.shape)
        #self.write_obj("image00320.obj",vertex[0].data.cpu().numpy().reshape(-1,3))
        #self.write_obj("image00320_e.obj",vertex2[0].data.cpu().numpy().reshape(-1,3))


    def draw_mesh_on_image(self,model,image):
        params_all = self.extract_param(model,image)
        param_shape = params_all[0:99]
        param_exp = params_all[99:128]
        param_camera = params_all[128:135]
        R = self.Q2R(param_camera[0:4])
        print(param_camera[0:4])
        face_shape = (mu_shape + w_shape @ (param_shape*1000) ).reshape(-1,3)/1000
        #print(face_shape)
        face_geometric =  (mu_shape +  w_shape @ (param_shape*1000)).reshape(-1,3)/1000
        face_projected =  -np.abs(param_camera[4]) * (face_geometric.T) 
        face_projected[:2,:] = face_projected[:2,:] + param_camera[5:7].reshape(2,1)
        face_projected = face_projected.T
        print(face_projected)

        # face_geometric =  (np.abs(param_camera[4]) * (mu_shape +  w_shape @ (param_shape*1000)).reshape(-1,3)/1000).T
        # face_geometric[:2,:] = face_geometric[:2,:] + param_camera[5:7].reshape(2,1)
        print(param_camera[5:7])
        print(param_camera[4])
        self.write_obj('../1.obj',face_projected)
        self.rander_face('../1.obj',image)
        pass

    
    def rander_face(self,obj_name, image_name):

        texture_size = 4
        image_size = 112
        # load .obj
        vertices, faces = nr.load_obj(obj_name, texture_size=texture_size)
        vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        print(vertices.shape)
        vertices[:, :, 0] +=  image_size /2
        vertices[:, :, 1] -=  image_size /2
        vertices[:, :, 1] = -vertices[:, :, 1]
        vertices[:, :, 2] = -vertices[:, :, 2]
        vertices /= image_size/2
        # vertices[:, :, 1] = -vertices[:, :, 1]
        # vertices[:, :, 0] = -vertices[:, :, 0]
        faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
        #textures = textures[None, :, :,:,:,:]
        textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
        #textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        renderer = Renderer(camera_mode='look_at',image_size = 112)
        print(vertices)
    
    # draw object
        loop = tqdm.tqdm(range(0, 360, 360))
   # writer = imageio.get_writer(args.filename_output, mode='I')
        for num, azimuth in enumerate(loop):
            loop.set_description('Drawing')
            azimuth = 180
            #renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)


            images,depth, alpha = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
            image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
            alpha = alpha.detach().cpu().numpy()[0] 
            image = image[:,8:104]
            alpha = alpha[:,8:104].reshape(112,96,1)
            image_b = imageio.imread(image_name)
            image_b = (1-alpha)  * image_b + image*255
            imageio.imwrite('../data/1.png',(255*image).astype(np.uint8))
            imageio.imwrite('../data/2.png',(image_b).astype(np.uint8))
        

        


if __name__ == '__main__':
    # bfm=np.load("../propressing/bfmz.npz")
    # w_shape = bfm['w_shape'][:,0:99]
    # mu_shape = bfm['mu_shape'].reshape(-1)
    # w_exp = bfm['w_expression'][:,0:29]
    # mu_exp = bfm['mu_expression'].reshape(-1)
    # e_exp = bfm['exp_ev'].reshape(1,29)

    # a = np.linalg.norm(np.random.randn(100000,29)*e_exp,axis=1).mean()
    # print(a)
    # exit(0)
    # ###########################################################
    # print("loading model...")
    # dict_file = config.dict_file
    # model = net3.sphere64a(pretrained=True,model_root=dict_file).cuda()
    # model = model.cuda()
    # gpu_ids = range(1)
    # print("evaluting...")
    # model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    

    

    
    ###########################################################
    # parser = argparse.ArgumentParser(description='EvalToolBox')
    # parser.add_argument('--list', '-l' , default= "./files.txt", help="model parameter filename")
    # # print("**************************")
    
    # args = parser.parse_args()
    # root = config.save_tmp_dir
    # filelist = args.list
    # print(args)
    print("**************************")
    print("evaluting...")
    # e=VisToolBox()
    # e.get_param_from_model(model)
    V  = VisToolBox()
    npylist = []
    # file_err = os.path.join(config.save_tmp_dir,'errors.npy')
    # npylist.append(file_err)
    # npylist.append('/data0/jdq/pca/errors.npy')
    # npylist.append('/data0/jdq/C_nonlinear_001/errors.npy')
    # npylist.append('/data0/jdq/C_nonlinear_002/errors.npy')
    # npylist.append('/data0/jdq/VC_nonlinear_001/errors.npy')

    npylist.append('/data/jdq/Clinear_001/errors.npy')


    # npylist.append('../data/errors_vae_s_0.0004.npy')
    # npylist.append('../data/errors_vae_s_0.01.npy')
    # npylist.append('../data/errors_vae_s2_0.001.npy')
    # npylist.append('../data/errors_vae_s2_0.01.npy')
    # npylist.append('../data/errors_vae_sc_0.001.npy')
    # npylist.append('../data/errors_vae_sc3_0.01.npy')
    # npylist.append('../data/errors_vae_sc3_0.001.npy')
    # npylist.append('../data/errors_vae_sc4_0.001.npy')
    
    # npylist.append('../data/errors_pca_0.01.npy')
    # #npylist.append('../data/vae3.npy')
    # npylist.append('../data/errors_vae_f.npy')
    # npylist.append('../data/errors_vae_f_0.01.npy')
    # npylist.append('../data/errors_vae_f_0.05.npy')
    # #npylist.append('../data/errors_vae2.npy')
    # npylist.append('../data/errors_vae.npy')
    # #npylist.append('../data/vae_p.npy')
    # npylist.append('../data/errors_ae_f.npy')
    # npylist.append('../data/Ours.npy')
    # V.draw_mesh_on_image(model,'../data/image00013.jpg')
    V.draw_curve_ced(npylist)
    