import os, shutil, sys
import numpy as np
import torch, torch.nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import scipy.io as scio
from pathlib import Path
from utils import *
from glob import glob

import config

def _parse_param_batch(param):
        """Work for both numpy and tensor"""
        N = param.shape[0]
        p_ = param[:, :12].view(N, 3, -1)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].view(N, 3, 1)
        alpha_shp = torch.zeros(N * 99).float().reshape(N, 99)
        alpha_exp = torch.zeros(N * 29).float().reshape(N, 29)
        # if config.use_cuda:
        #     alpha_shp, alpha_exp = alpha_shp.to(config.device), alpha_exp.to(config.device)
        
        alpha_shp[:, :40] = param[:, 12:52]
        alpha_exp[:, :10] = param[:, 52:]
        return p, offset, alpha_shp, alpha_exp


class MyDataSet(Dataset):
    def __init__(self, max_number_class=100000,root="../", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.catalog = root 

        if not os.path.exists(self.catalog):
            raise ValueError("Cannot find the data!")
        self.shape_mat = dict(scio.loadmat("/data2/lmd2/imgc/shape.mat"))
        self.transform_with_lms = transforms.Compose([
            #transforms.RandomResizedCrop(112,scale=(0.9,1),ratio=(1, 1)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomApply([transforms.RandomRotation(10)]),
            transforms.RandomCrop((112,96))        
        ])
            

        file_list_path = root  + 'file_path_list_imgc2.txt'
        count=0
        with open(file_list_path) as ins:
            for ori_img_path in ins:
                ori_img_path = ori_img_path[0:-1]
                ori_mat_path = ori_img_path[:-3] + 'mat'
                id = os.path.split(os.path.split(ori_img_path)[0])[-1]
                if str(id) in self.shape_mat and os.path.exists(ori_mat_path):
                    self.data.append([ori_img_path, id, ori_mat_path])
                    count += 1
                    print("loaded:%d\r" % count, 
                    sys.stdout.flush())
                    if count == 128 * 40:
                        return 
                        pass


    def __getitem__(self, index):
        #index = 0
        imgname, label, mat_path = self.data[index]
        img=Image.open(imgname)
        
        mat = scio.loadmat(mat_path)
        lms = mat['lm'].reshape(-1)
        
        img, lms = self.transform_with_lms(img, lms)
        
        lms = torch.from_numpy(lms)
        shape = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = self.transform(img)
        return img, shape, lms

    def __len__(self):
        return len(self.data)

class AFLW2000DataSet(Dataset):
    def __init__(self, max_number_class=100000,root="../", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.catalog = root 
        self.transform = transform
        if not os.path.exists(self.catalog):
            raise ValueError("Cannot find the data!")
            
        types = ('*.jpg', '*.png')
        image_path_list= []
        for files in types:
            image_path_list.extend(glob(os.path.join(root, files)))
        #file_list_path = root  + '../file_path_list_AFLW2000_align.txt'
        print(len(image_path_list))
        #count=0
        
        for ori_img_path in image_path_list:
            #ori_img_path = ori_img_path[0:-1]
            ori_mat_path = ori_img_path[:-3] + 'mat'
            #print(ori_img_path)
            self.data.append([ori_img_path, ori_mat_path])
            #count += 1
            #print("loaded:%d\r" % count, 
            #sys.stdout.flush())
        # imgname, mat_path = self.data[0]
        # img=cv2.imread(imgname)
        # mat = scio.loadmat(mat_path)
        # lms = mat['pt2d_68'].transpose(1, 0).astype('int')
        # point_size = 1
        # point_color = (0, 0, 255)
        # thickness = 1 

    
        # for point in lms:
        #     print(point)
        #     cv2.circle(img, tuple(point), point_size, point_color, thickness)
        #         # img[int(mse[:,0]),int(mse[:,1])] = [1,0,0]
        # cv2.imwrite('1234.jpg',img)
        # exit()
      

    def __getitem__(self, index):
        #index = 0
        imgname, mat_path = self.data[index]
        img=Image.open(imgname)
        mat = scio.loadmat(mat_path)
        
        lms = mat['pt2d_68'].transpose(1, 0).reshape(-1)
        img = self.transform(img)
        return img, lms 

    def __len__(self):
        return len(self.data)
class LfwDataSet(Dataset):
    def __init__(self, root="../", pairfile="../pair.txt", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.pairfile = pairfile 

        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        with open('../pairs.txt') as f:
            pairs_lines = f.readlines()[1:]
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')
            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            if 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

            imgname1 = self.root + name1
            imgname2 = self.root + name2
            #print(imgname1,imgname2,sameflag)
            self.data.append([imgname1,imgname2,sameflag])

class FSDataSet(Dataset):
    def __init__(self, root="", filelist="../pair.txt", transform=torchvision.transforms.ToTensor()):
        self.root = root
        self.data = []
        self.transform = transform
        self.filelist =  filelist
        self.shape_mat = dict(scio.loadmat("/data2/lmd2/imgc/shape.mat"))
        # if not os.path.exists(self.root):
        #     raise ValueError("Cannot find the data!")
        label=0
        with open(self.filelist) as f:
            imgnames = f.readlines()
            for imgname in imgnames:
                # print(imgname)
                # print(imgname.split('/'))
                # label = int(imgname.split('/')[0])
                # print(label,str(label) in self.shape_mat )   
                # if str(label) in self.shape_mat:  
                
                self.data.append([self.root+imgname[:-1], label])
                # readlines = f.readlines()
            
    def __getitem__(self, index):
        print(self.data)
        imgname, label = self.data[index]
        img=Image.open(imgname)
        label = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
        
class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DDFADataset(Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        with open(filelists) as f:
            self.lines = f.readlines()#[:4000]
        from utils import _load
        #from pytorch_3DMM import BFMA_3DDFA_batch
       
        self.params = torch.from_numpy(_load(param_fp))
        meta = _load('./train.configs/param_whitening_2.pkl')
        self.param_mean = torch.from_numpy(meta.get('param_mean'))
        self.param_std = torch.from_numpy(meta.get('param_std'))
        
        self.params = self.params * self.param_std + self.param_mean
        '''
        img_idx =100
        self.bfmz = BFMA_3DDFA_batch.BFMA_3DDFA_batch()
        path = os.path.join(self.root, self.lines[img_idx][:-1])
        img = cv2.imread(path)
        gt_rotate, gt_offset, gt_shape, gt_exp = _parse_param_batch(self.params[img_idx:img_idx+1])
        gt_face, gt_face_rotated, gt_face_proj = self.bfmz(gt_shape, gt_exp , gt_rotate, gt_offset)
        mse = self.bfmz.get_landmark_68(gt_face_proj).transpose(2,1)[0]
        # print(mse[:,0])
        # print(img[int(mse)])
        point_size = 1
        point_color = (0, 0, 255)
        thickness = 1 

    
        for point in mse:
            print(point)
            cv2.circle(img, tuple(point), point_size, point_color, thickness)
                # img[int(mse[:,0]),int(mse[:,1])] = [1,0,0]

        cv2.imwrite("123.jpg",img)
        '''
        # print(mse)
        #exit()
        #self.params[:, :11] = self.params[:, :11] * 112. / 120
        #self.params[:, 3] = self.params[:, 3] - 8 # 112 -> 96
        
        #print(self.params[0, 12:52])
        #self.params[:, 12:52] = self.params[:, 12:52].div(BFMA_batch.BFMA_batch.shape_ev_tensor[:40].to('cpu'))
        #self.params[:, 12:] = self.params[:, 12:]# / 1000.
        
        #print(BFMA_batch.BFMA_batch.shape_ev_tensor[:40])
        #print(self.params[0, 12:52])
        #exit()
        
    def _target_loader(self, index): 
        target = self.params[index]

        return target

    def __getitem__(self, index):
        path = os.path.join(self.root, self.lines[index][:-1])
        img = Image.open(path)
        target = self._target_loader(index)# * self.param_std + self.param_mean
        
        #target[:11] = target[:11] * 112. / 120
        #target[3] = target[3] - 8
        raw_img = config.transform_raw(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, raw_img 

    def __len__(self):
        return len(self.lines)

class VGG2MixDataset(Dataset):
    def __init__(self, max_number_class=8631,indexfile="../", transform=torchvision.transforms.ToTensor(), ddfa_root = '../', ddfa_filelists = '../', ddfa_param_fp = '../', mix = True):
        self.indexfile = indexfile
        self.data = []
        self.transform = transform
        self.mix = mix
        self.ddfa_dataset = DDFADataset(ddfa_root, ddfa_filelists, ddfa_param_fp,transform)

        if not os.path.exists(self.indexfile):
            raise ValueError("Cannot find the index file!")


        with open(indexfile) as f:
            lines = f.readlines()

        for idx,line in enumerate(lines): 
            #sys.stdout.write(str(idx+1)+'/'+str(len(lines))+'\r')
            line=line.replace('\n', '')
            line=line.replace('ssd-', '')
            #print(line)
            words = line.split('/')
            label=int(words[-2])
            if label<max_number_class:         
                self.data.append([line, label])
        print(len(self.data))
        
        self.sample_weight = ([len(self.ddfa_dataset)] * len(self.data)) + ([len(self.data)]*len(self.ddfa_dataset))
    def __getitem__(self, index):
        label = {}
#        print(index)
        #if False:
        if index < len(self.data):
            imgname, id = self.data[index]
            img=Image.open(imgname)
            raw_img = config.transform_raw(img)
            img = self.transform(img)
            
            
            mat_path = imgname.replace('.jpg', '.mat')
            
            label['ind_id'] = 1
            label['id'] = id
            if os.path.exists(mat_path):
                mat_obj = scio.loadmat(mat_path)
                lms = mat_obj['lm'].reshape(-1)
                label['ind_lm'] = 1
                label['lm'] = lms.astype(np.float)
            else:
                label['ind_lm'] = 0
                label['lm'] = np.zeros(68 * 2).astype(np.float)
            
            label['ind_3d'] = 0
            label['3d'] = torch.from_numpy(np.zeros(62).astype(np.float).reshape(-1))
            
        else:
            #img, target_3d, _ = self.ddfa_dataset.__getitem__(index)
            img, target_3d, raw_img = self.ddfa_dataset.__getitem__(index - len(self.data))
            label['ind_id'] = 0
            label['id'] = 0
            label['ind_lm'] = 0
            label['lm'] = np.zeros(68 * 2).astype(np.float).reshape(-1)
            label['ind_3d'] = 1
            label['3d'] = torch.from_numpy(target_3d.data.numpy().astype(np.float))
        #print(img.shape)
        return img, label ,raw_img

    def __len__(self):
        return self.ddfa_dataset.__len__() + len(self.data) if self.mix else len(self.data)

    def get_weight(self):
        return self.sample_weight

class FeatureDataset(Dataset):
    def __init__(self, root, filelist, transform=torchvision.transforms.ToTensor(), target_transform=None, resample = 1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")

        filelist= os.path.join(self.root,filelist)
        with open(filelist,'r') as f:
            imgs=f.readlines()
            for img in imgs:  
                imgname = os.path.join(self.root,img)  
                imgname = imgname.replace("\n",'')
                #print(imgname)
                if '.jpg' in imgname or '.png' in imgname: 
                    #img = Image.open(imgname)
                    self.data.append(imgname)
        self.data = self.data[::resample]
    def __getitem__(self, index):
        imgname = self.data[index]
        img = Image.open(imgname)
        # img = self.transform(img)
        img_o = self.transform(img)
        img_f = self.target_transform(img)
        return  imgname, img_o,img_f

    def __len__(self):
        return len(self.data)

class LfwDataSet(Dataset):
    def __init__(self, root="../", pairfile="../pair.txt", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.pairfile = pairfile 

        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        with open('../pairs.txt') as f:
            pairs_lines = f.readlines()[1:]
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')
            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            if 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

            imgname1 = self.root + name1
            imgname2 = self.root + name2
            #print(imgname1,imgname2,sameflag)
            self.data.append([imgname1,imgname2,sameflag])
           

                
    def __getitem__(self, index):
        imgname1, imgname2, label = self.data[index]
        img1 = Image.open(imgname1)
        img2 = Image.open(imgname2)
        img1_o = self.transform(img1)
        img1_f = self.target_transform(img1)
        img2_o = self.transform(img2)
        img2_f = self.target_transform(img2)
        return imgname1, imgname2,img1_o,img2_o,img1_f,img2_f,label

    def __len__(self):
        return len(self.data)

def Test():
    print("For test.\n")
    '''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomResizedCrop(112,scale=(0.8,1.2)),
        torchvision.transforms.ToTensor()
    ])
    '''
    transform = torchvision.transforms.ToTensor()   
    #dataset = VGG2MixDataset(transform=transform, indexfile="/ssd-data/jdq/dbs/faces_vgg_112x96/file_path_list_vgg2.txt")
    #dataset = VGG2MixDataset(
    #    max_number_class=8631,
    #    indexfile="/ssd-data/jdq/dbs/faces_vgg_112x96/file_path_list_vgg2.txt",
    #    transform = transform,
    #    ddfa_root="/ssd-data/lmd/train_aug_120x120_aligned",
    #    ddfa_filelists='./train.configs/train_aug_120x120.list.train',
    #    ddfa_param_fp = './train.configs/param_aligned.pkl'
    #)
    dataset = MSCelebShapeDataSet(transform=transform, indexfile="/data1/jdq/imgc2/file_path_list_imgc2.txt", max_number_class=79077 )
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=32)
    inputdata, target = dataset.__getitem__(3300000)
    img=inputdata.data.cpu().numpy()
    #cv2.imwrite("1.jpg",img)
    print(img)
    print(target)


class MICCDataSet(Dataset):
    def __init__(self, root="../", filelist="../files.txt", transform=torchvision.transforms.ToTensor()):
        self.root = root
        self.data = []
        self.transform = transform
        self.filelist =  filelist
        # self.shape_mat = dict(scio.loadmat("/data2/lmd2/imgc/shape.mat"))
        # if not os.path.exists(self.root):
        #     raise ValueError("Cannot find the data!")
        label=0
        with open(self.filelist) as f:
            imgnames = f.readlines()
            for imgname in imgnames:
                # print(imgname)
                # print(imgname.split('/'))
                label = int(imgname.split("/")[0])
                imgname = os.path.join(root,imgname)
                imgname = imgname.replace("\n","")
                # print(label,str(label) in self.shape_mat )   
                # if str(label) in self.shape_mat:  
                
                self.data.append([imgname , label])
            
    def __getitem__(self, index):
        #print(self.data)
        imgname, label = self.data[index]
        img=Image.open(imgname)
        #label = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = self.transform(img)
        return img, label, imgname

    def __len__(self):
        return len(self.data) 

class MSCelebShapeDataSet(Dataset):
    def __init__(self, max_number_class=100000,indexfile="../", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.indexfile = indexfile
        self.data = []
        self.transform = transform

        if not os.path.exists(self.indexfile):
            raise ValueError("Cannot find the data!")
        self.shape_mat = dict(scio.loadmat("/data/jdq/imgc2/shape.mat"))
       
        with open(self.indexfile) as f:
            lines = f.readlines()
            for idx,line in enumerate(lines): 
                line=line.replace('\n', '')
                words = line.split('/')
                label=int(words[4])
                if label<max_number_class:    
                    if str(label) in self.shape_mat:     
                        self.data.append([line, label])
                        print("loaded:%d\r" % len(self.data), 
                        sys.stdout.flush())
                
    def __getitem__(self, index):
        imgname, label = self.data[index]
        img=Image.open(imgname)
        label = torch.Tensor(self.shape_mat[str(label)].reshape(-1))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    Test()
