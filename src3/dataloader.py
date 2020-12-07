import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as scio
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
import argparse
from scipy import stats
#from scipy.interpolate import spline
# data = np.load('frgc2.npz')
# vertices = data['data']
# x = vertices.reshape(vertices.shape[0],-1)
# x_mean = x.mean(axis=0)
# x = x - x_mean.reshape(1,-1)
# y = data['label']
# clf = LinearDiscriminantAnalysis(n_components=299)
# clf.fit(x,y)
# x_init = clf.transform(x)
# print(x_init.shape)
# print(clf.coef_.shape)
# np.savez('init.npz', ceof = clf.coef_, x= x_init, x_mean = x_mean)
# np.save('lda_init.npy',clf.coef_)
# np.save('x_init.npy',x_init)
# print(x)
# print(y) 

def readobj2(file_name):
    with open(file_name, "r") as f:
        lines=f.readlines()
        vertices = []
        faces = []
        for line in lines:
            words = line.split(" ")
            if words[0]=="v":
                ver=np.zeros(3)
                ver[:]=float(words[1]),float(words[2]),float(words[3])
                vertices.append(ver)
            if words[0]=="f":
                face=np.zeros(3,dtype = int)
                face[:]=float(words[1]),float(words[2]),float(words[3])
                faces.append(face)
        vertices=np.array(vertices)
        faces=np.array(faces)
    return vertices, faces

def write_obj(filename,vertices,index):
        outdir = os.path.dirname(filename)
        # if not os.path.exists(outdir):
        #     os.mkdir(outdir)
        num_vertex = vertices.shape[0]
        num_index = index.shape[0]
        fobj=open(filename,'w+')
        fobj.write("# {} vertices, {} faces\n".format(num_vertex, num_index))
        #print(vertices[0])
        for i in range(num_vertex):
            fobj.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(num_index):
            fobj.write("f {} {} {}\n".format(index[i][0], index[i][2], index[i][1]))
        fobj.close()


def draw_curve_ced(datas):
        # data1 = np.load(data_file)
        # data2 = np.load('../data/errors10.npy')
        # datas = [data1,data2]
        #print(data2-data)
        # print("Error Mean: ", data.mean())
        # print("Error Std:", data.std())
        
        #data = randn(75)
        # for data_name in datas:
        #data = np.load(data_name)
        #data = data[:,1]
        for i in range(1):
            print(i)
            data = datas.reshape(-1)
            data = data[np.abs(data)<200]
            print("Error Mean: ", data.mean())
            print("Error Std:", data.std())
            print(data.shape)
            
            data2=np.random.normal(data.mean(),data.std(),4950*99)#+np.random.normal(data.mean(),data.std(),4950*909)
            print(data2.shape)
            data = datas.reshape(-1)

            y2,x2 = np.histogram(data2,bins=1000)
            y2 = y2/data2.shape
            y,x = np.histogram(data,bins=1000)
            y = y/data.shape
          
            y = np.insert(y,0,0)
            y2 = np.insert(y2,0,0)

            x_new = np.linspace(-2000,2000,5000)
            y_s = spline(x,y,x_new)
            y_s /= np.sum(y_s)
           


            x_new2 = np.linspace(-2000,2000,5000)
            y_s2 = spline(x2,y2,x_new2)
            y_s2 /= np.sum(y_s2)
            print(np.sum(y_s))
            print(np.sum(y_s2))
            plt.plot(x_new,y_s*100,label='all')
            plt.plot(x_new2,y_s2*100,label='normal')
        #print(y_s)
        plt.title("Probability Density Function",fontsize=12)
        plt.ylabel("Probability(%)",fontsize=10)
        plt.xlabel("Coordinate Value",fontsize=10)
        plt.xlim(-500,500)
        # plt.ylim(0,100)
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()





class BasicBlock(nn.Module):
    
    def __init__(self,planes,expansion=8):
        super(BasicBlock,self).__init__()
        m = OrderedDict()
        m['fc1'] = nn.Linear(planes, planes*expansion)
        m['relu1'] = nn.PReLU(planes*expansion)
        # m['fc2'] = nn.Linear(planes*expansion, planes*expansion)
        # m['relu2'] = nn.PReLU(planes*expansion)
        m['fc3'] = nn.Linear(planes*expansion, 53215*3)
        # m['relu2'] = nn.PReLU(planes)
        self.planes = planes
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        out = self.group1(x)
        return out

class model_non_vae(nn.Module):
    def __init__(self, dim=199,layer=2,width=199, vae=False, use_cuda=True, center=False):
        super(model_non_vae, self).__init__()
        self.width = width
        self.dim = dim
        self.vae = vae
        #data = np.load('/data0/jdq/model/frgc2.npz')
        data = np.load('frgc2.npz')
        x = data['data']
        y = data['label']-1
        self.num_cls = y.max()+1
        self.centers = torch.nn.Parameter(torch.randn(self.num_cls,self.width))
        self.x_mean = torch.nn.Parameter(torch.randn(x.shape[0],self.width))
        self.x_logvar =  torch.nn.Parameter(torch.randn(x.shape[0],self.width))
        # self.s = torch.nn.Parameter(torch.randn(1))
        # self.s.data.fill_(1)
        self.mlc= BasicBlock(self.width)
        if use_cuda:
            self.labels = torch.from_numpy(y).long().cuda()
        else:
            self.labels = torch.from_numpy(y).long()
        self.center = center     
        self.mu = x.mean(axis=0).reshape(-1)
        self.B = x.reshape(x.shape[0],-1) - self.mu
        if use_cuda:
            self.B = torch.from_numpy(self.B.astype(np.float32)).cuda()
        else:
            self.B = torch.from_numpy(self.B.astype(np.float32))
        self.num_sample = x.shape[0]
        #####################
        self.E = torch.eye(self.width).cuda()
        #self.ip = nn.Linear(self.width,self.num_cls)
        self.params = nn.ParameterList()
        for param in self.parameters():
            self.params.append(param)
        print(self.params)
        self.batch = 296
        ##########################
        # if with_data:
        #     data = np.load('/data0/jdq/test.npz')
        #     x_data = data['x']
        #     labels = data['y']
        #     self.B_test = torch.from_numpy((x_data.reshape(x_data.shape[0],-1) - self.mu).astype(np.float32)).cuda()
        #     self.labels_test = torch.from_numpy(labels.astype(np.float32)).cuda().long()
        #     self.num_cls_test = max(labels)
        #     self.num_sample_test = x_data.shape[0]
        
    def load_data(self,x,y):
        num = x.shape[0]
        x = torch.from_numpy(x.astype(np.float32).reshape(num,-1)- self.mu).float()
        #x = x - x.mean(dim=0) 
        #print(x)
        y = torch.from_numpy(y.astype('int')).reshape(num)
        self.num_cls = max(y)+1
        self.B = x.cuda()
        self.labels = y.long().cuda()
        self.batch = num 
        self.x_mean.data = torch.randn(self.B.shape[0],self.dim).cuda()
        self.x_logvar.data = torch.randn(self.B.shape[0],self.dim).cuda()
        self.centers.data = torch.randn(self.num_cls,self.dim).cuda()
        #self.x_mean.data = torch.randn(self.B.shape[0],self.dim).cuda()
        #self.x = self.x[:num]
        # self.B[num:] = 0
        self.num_sample = num
        # self.labels[:num]=-1
        # self.test_ex = True
        #self.batch = num
        #num_y = y.shape

    def center_loss(self,y,hidden):
        #batch_size = hidden.size(0)
        
        expanded_centers = self.centers.index_select(dim=0,index=y)

        # intre_diff = self.centers.expand(self.num_cls,self.num_cls,self.width)
        # intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()

        
        
        intra_distances = torch.sum((hidden - expanded_centers)**2, dim=1).mean()
        # re = intra_distances/intre_diff
        #print("cl",expanded_centers.norm(dim=1).shape)
        #cmd = intra_distances / (expanded_centers.norm(dim=1).mean())

        return intra_distances



    def sampling(self,args):
        z_mean, z_log_var = args
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        if use_cuda:
            epsilon = torch.randn(batch_size,dim).cuda()
        else:
            epsilon = torch.randn(batch_size,dim)
        return z_mean + torch.exp(0.5*z_log_var) * epsilon

    def cal_centers(self):
        max_n = 0
        data = torch.zeros(self.num_cls,30,self.width).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        for idx,label in enumerate(self.labels):
            data[label,count[label],:] = self.x[idx]
            count[label]+=1
        centers = torch.sum(data,dim=1) / ((count+1e-8).reshape(-1,1).float())
        inner_diff = ((self.x - centers.index_select(dim=0, index = self.labels))**2).mean()
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.width)
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        re = inner_diff/intre_diff
        return re

    def cal_error(self):
        # sum_err=0
        idx = 0
        loss = 0
        with torch.no_grad():
            while idx + self.batch <= self.num_sample:
                loss+=(((self.mlc(self.x_mean[idx:idx + self.batch])  - self.B[idx:idx + self.batch])**2).reshape(self.batch,-1,3).sum(dim=-1)**0.5).mean(dim=1).sum()
                idx += self.batch
            if idx < self.num_sample:
                loss+=(((self.mlc(self.x_mean[idx:])  - self.B[idx:])**2).reshape(self.num_sample-idx,-1,3).sum(dim=-1)**0.5).mean(dim=1).sum()
        return loss/self.num_sample

    # def cal_error_test(self):
    #     for idx in range(0,self.num_sample_test):
    #         loss=(((self.mlc(self.x_mean[idx])  - self.B[idx])**2).reshape(-1,3).sum(dim=-1)**0.5).mean()
    #         sum_err += loss
    #     return sum_err/self.num_sample


    def get_single_NMCD(self,ix,iy):
        data = torch.zeros(self.num_cls,self.dim).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x_mean[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())
        
        inner_diff = ((self.x_mean[ix]-self.x_mean[iy])**2).mean()
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.dim)
        
        #print(intre_diff)
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
            #print(intre_diff/(self.x.norm(dim=1).mean()**2))
        re = inner_diff/intre_diff
        return re

    def cal_NMCD(self):
        data = torch.zeros(self.num_cls,self.dim).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        #x_mean = self.x_mean
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x_mean[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())
        #print(centers)
        inner_diff = ((self.x_mean[:self.num_sample] - centers.index_select(dim=0, index = self.labels))**2).mean()
        print(inner_diff/(centers**2).mean())
        #print(centers.norm(dim=1).mean())
        #intre_diff = centers.norm(dim=1).mean()
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.dim)
        #print((intre_diff- intre_diff.transpose(0,1)).norm(dim=-1).mean())
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        print(intre_diff/(centers**2).mean())
            #print(intre_diff/(self.x_mean.norm(dim=1).mean()**2))
        re = inner_diff/intre_diff
        return re

    def getmesh(self,idx):
        err = (((self.mlc(self.x_mean[idx])  - self.B[idx])**2).reshape(-1,3).sum(dim=-1)**0.5).mean()
        print(err)
        return self.mu+(self.mlc(self.x_mean[idx])).data.cpu().numpy(), err.data.cpu().numpy()
    def get_data(self):
        return self.x_mean.data.cpu().numpy(),self.labels.cpu().numpy()
    

    def ground_truth_mesh(self,idx):
        return self.mu+self.B[idx].data.cpu().numpy()

    def forward(self):
        
        idx = random.sample(range(0,self.num_sample),256) 
        if self.vae:
            x = self.sampling([self.x_mean[idx],self.x_logvar[idx]]).cuda()
            c = self.mlc(x)
        else:
            c = self.mlc(self.x_mean[idx])
        if self.center:
        
            cl = self.center_loss(self.labels[idx],self.x_mean[idx])
        else:
            cl =0

        #pred = self.ip(x) 
     #   x = self.s*x
        
        #print(pred.shape)
        #crx = self.cel_loss(pred,self.label[idx])
        KL_loss =   1 + (self.x_logvar[idx]) - (self.x_mean[idx])**2 - torch.exp(self.x_logvar[idx])
        KL_loss = (-0.5*torch.sum(KL_loss,dim=-1)).mean()
        #rex = (c**2).sum(dim=-1).mean()
        #rex = 0
        #d =  ((c - self.B[idx])**2).mean()
        d =  ((c - self.B[idx])**2).mean()
        #r = ((self.A @ self.A.t() - self.E)**2).mean()
        #r = 0
        #print("AE:", ((x @ self.A)**2).mean() )
        #print("s:",self.s)
        # self.count = 0
        #re = ((self.mlc(self.x_mean[idx]) - self.B[idx])**2).mean()
        
        # centers = self.cal_centers()
        # inner_diff = ((self.x[idx] - centers.index_select(dim=0, index = self.labels[idx]))**2).mean()
        # #print(inner_diff)
        # intre_diff = centers.expand(self.num_cls,self.num_cls,self.width)
        # #print(intre_diff)
         # intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        # #print(intre_diff/(self.x.norm(dim=1).mean()**2))
        # re = inner_diff/intre_diff
    
            #re = ((self.x_mean[idx]*self.s @ self.A - self.B[idx])**2).mean() 
        #print('re:',re)
        # print(self.A @ self.A.T)
        # g = 
        # print('d',d)
        # print('r',r)
        # print('re',re)
        # y=  d + r
        return d,  KL_loss, cl

    def discrimination_metric(self):
        X = self.x_mean.data.cpu().numpy()
        labels = self.labels.data.cpu().numpy()
        #print(X.shape,labels.shape)
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
        davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
        return silhouette_score,calinski_harabasz_score,davies_bouldin_score


class model_vae(nn.Module):
    def __init__(self, n_components=299):
        super(model_vae, self).__init__()

        data = np.load('/data0/jdq/model/frgc2.npz')
        x = data['data']
        y = data['label']
        # x = np.load('pca.npy')

        self.x_mean = torch.nn.Parameter(torch.randn(x.shape[0],99))
        self.x_logvar =  torch.nn.Parameter(torch.randn(x.shape[0],99))
        self.s = torch.nn.Parameter(torch.randn(1))
        self.s.data.fill_(1)
        #self.label = torch.Tensor(range(x.shape[0])).long().cuda()
        if use_cuda:
            self.label = torch.from_numpy(y).long().cuda()
        else:
            self.label = torch.from_numpy(y).long()
        

        
        
        self.mu = x.mean(axis=0).reshape(-1)
        self.B = x.reshape(x.shape[0],-1) - self.mu
        
        # print(std)
        # exit()
        if use_cuda:
            self.B = torch.from_numpy(self.B.astype(np.float32)).cuda()
        else:
            self.B = torch.from_numpy(self.B.astype(np.float32))
        # print((self.B**2).sum(dim=1).mean())
        # print((self.B**2).reshape(-1,3).sum(-1).pow(0.5).mean())
        # print(2.7433**2*53215)
        # exit()
        self.A = torch.nn.Parameter(torch.randn(99,53215*3))
        #self.x = torch.nn.Parameter(torch.randn(x.shape[0],99))
        self.num_sample = x.shape[0]
        self.num_cls = 600
        self.cel_loss = torch.nn.CrossEntropyLoss()
        ####################
        


        #####################
        self.E = torch.eye(99).cuda()
        self.ip = nn.Linear(99,self.num_cls)
        self.centers = torch.nn.Parameter(torch.randn(self.num_cls,99))
        self.params = nn.ParameterList()
        for param in self.parameters():
            self.params.append(param)
        print(self.params)
        

    def center_loss(self,y,hidden):
        #batch_size = hidden.size(0)
        expanded_centers = self.centers.index_select(dim=0,index=y)
        intra_distances = torch.sum((hidden - expanded_centers)**2, dim=1).pow(0.5).mean()

        return intra_distances



    def sampling(self,args):
        z_mean, z_log_var = args
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        if use_cuda:
            epsilon = torch.randn(batch_size,dim).cuda()
        else:
            epsilon = torch.randn(batch_size,dim)
        return z_mean + torch.exp(0.5*z_log_var) * epsilon

    def cal_centers(self,dim=199):
        max_n = 0
        data = torch.zeros(self.num_cls,dim)
        count = torch.zeros(self.num_cls).long()
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())

        inner_diff = ((self.x - centers.index_select(dim=0, index = self.labels))**2).mean()
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.width)
        #print(intre_diff)
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        #print(intre_diff/(self.x.norm(dim=1).mean()**2))
        re = inner_diff/intre_diff
        return centers

        

    def forward(self):
        
        idx = random.sample(range(0,self.num_sample),1500) 
        x = self.sampling([self.x_mean[idx],self.x_logvar[idx]]).cuda()
        cl = self.center_loss(self.label[idx],self.x_mean[idx])
        pred = self.ip(x)
        #x = self.s*x
        
        #print(pred.shape)
        crx = self.cel_loss(pred,self.label[idx])
        KL_loss =   1 + (self.x_logvar[idx]) - (self.x_mean[idx])**2 - torch.exp(self.x_logvar[idx])
        KL_loss = (-0.5*torch.sum(KL_loss,dim=-1)).mean()
        rex = (x**2).sum(dim=-1).mean()
        d =  ((x @ self.A - self.B[idx])**2).mean()
        r = ((self.A @ self.A.t() - self.E)**2).mean()
        #print("AE:", ((x @ self.A)**2).mean() )
        #print("s:",self.s)
        # self.count = 0
        #re = ((self.x_mean[idx]*self.s @ self.A - self.B[idx])**2).reshape(1500,-1,3).sum(-1).pow(0.5).mean()
        re = 0
        # centers = self.cal_centers()
        # inner_diff = ((self.x[idx] - centers.index_select(dim=0, index = self.labels[idx]))**2).mean()
        # #print(inner_diff)
        # intre_diff = centers.expand(self.num_cls,self.num_cls,99)
        # #print(intre_diff)
         # intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        # #print(intre_diff/(self.x.norm(dim=1).mean()**2))
        # re = inner_diff/intre_diff
    
        #re = ((self.x_mean[idx]*self.s @ self.A - self.B[idx])**2).mean() 
        #print('re:',re)
        # print(self.A @ self.A.T)
        # g = 
        # print('d',d)
        # print('r',r)
        # print('re',re)
        # y=  d + r
        return d, r , crx, rex,KL_loss,cl,re

class pca_model(nn.Module):
    def __init__(self,server='lab',dim=199):
        super(pca_model,self).__init__()
        self.test_ex = False
        self.dim = dim
        model = np.load('p'+str(self.dim)+'_model.npz')
        if server == 'lab':
            data = np.load('/data0/jdq/model/frgc2.npz')
        else:
            data = np.load('/data/jdq/frgc2.npz')
        # x = data['data']
        y = data['label']-1
        self.num_cls = max(y)+1
        self.sample = y.shape[0]
        #print(min(y))
        self.labels = torch.from_numpy(y.astype(np.float32)).cuda().long()
        
        self.mu = torch.from_numpy(model['mu'].astype(np.float32)).cuda()
        self.A = torch.from_numpy(model['w_shape'].astype(np.float32)).cuda()
        self.B =  torch.from_numpy(data['data'].reshape(data['data'].shape[0],-1).astype(np.float32)).cuda()-self.mu
        self.x = torch.nn.Parameter(torch.randn(self.B.shape[0],self.dim))
        self.batch = 600


    def forward(self):
        idx = random.sample(range(0,self.B.shape[0]),self.batch) 
        loss=((self.x[idx] @ self.A - self.B[idx])**2).mean()
        return loss

    def cal_error(self):
        sum_err=0
        for idx in range(0,self.B.shape[0]):
            loss=(((self.x[idx] @ self.A - self.B[idx])**2).reshape(-1,3).sum(dim=-1)**0.5).mean()
            sum_err += loss
        return sum_err/self.B.shape[0]
    def ground_truth_mesh(self,idx):
        return (self.mu+self.B[idx]).data.cpu().numpy()

    def getmesh(self,idx):
        loss=(((self.x[idx] @ self.A - self.B[idx])**2).reshape(-1,3).sum(dim=-1)**0.5).mean()
        print(loss)
        return (self.mu+self.x[idx] @ self.A).data.cpu().numpy()
    def get_data(self):
        return self.x.data.cpu().numpy(),self.labels.cpu().numpy()

    def getlabel(self,idx):
        return self.labels[idx].cpu().numpy()

    def discrimination_metric(self):
        X = self.x.data.cpu().numpy()
        labels = self.labels.data.cpu().numpy()
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
        davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
        return silhouette_score,calinski_harabasz_score,davies_bouldin_score

    def load_data(self,x,y):
        num = x.shape[0]
        x = torch.from_numpy(x.astype(np.float32)).reshape(num,-1)
        #x = x - x.mean(dim=0) 
        print(x)
        y = torch.from_numpy(y.astype('int')).reshape(num)
        self.num_cls = max(y)+1
        self.B = x.cuda()- self.mu
        self.labels = y.long().cuda()
        self.batch = num 
        self.x.data = torch.randn(self.B.shape[0],self.dim).cuda()
        #self.x = self.x[:num]
        # self.B[num:] = 0
        self.sample = num
        # self.labels[:num]=-1
        # self.test_ex = True
        
        #num_y = y.shape


    def get_single_NMCD(self,ix,iy):
        data = torch.zeros(self.num_cls,self.dim).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())
        
        inner_diff = ((self.x[ix]-self.x[iy])**2).mean()
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.dim)
        
        #print(intre_diff)
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
            #print(intre_diff/(self.x.norm(dim=1).mean()**2))
        re = inner_diff/intre_diff
        return re


    def cal_NMCD(self):
        max_n = 0
        data = torch.zeros(self.num_cls,self.dim).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())
        
        inner_diff = ((self.x[:self.sample] - centers.index_select(dim=0, index = self.labels))**2).mean()
        #print((self.x[:self.sample] - centers.index_select(dim=0, index = self.labels)).norm(dim=-1).mean())
        print(inner_diff/(centers**2).mean())
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.dim)
        #print((intre_diff- intre_diff.transpose(0,1)).norm(dim=-1).mean())
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        
        print(intre_diff/(centers**2).mean())
        re = inner_diff/intre_diff
        return re


class linear_model(nn.Module):
    def __init__(self,server='lab',dim=199, vae=False, with_data=True, center=False):
        super(linear_model,self).__init__()
        #model = np.load('p399_model.npz')
        self.center = center
        self.vae = vae
        self.dim  = dim

        if server == 'lab':
            data = np.load('/data0/jdq/model/frgc2.npz')
        else:
            data = np.load('/data/jdq/frgc2.npz')

        x = data['data']
        y = data['label']-1
        self.num_cls = max(y)+1
        self.num_sample = x.shape[0]
        
        #print(min(y))
        self.labels = torch.from_numpy(y.astype(np.float32)).cuda().long()
        self.mu = x.mean(axis=0).reshape(-1)
        self.B = torch.from_numpy((x.reshape(x.shape[0],-1) - self.mu).astype(np.float32)).cuda()
        self.E = torch.eye(self.dim).cuda()

        if with_data:
            data = np.load('/data0/jdq/test.npz')
            x_data = data['x']
            labels = data['y']
            self.B_test = torch.from_numpy((x_data.reshape(x_data.shape[0],-1) - self.mu).astype(np.float32)).cuda()
            self.labels_test = torch.from_numpy(labels.astype(np.float32)).cuda().long()
            self.num_cls_test = max(labels)
            self.num_sample_test = x_data.shape[0]
        

        
        #self.B =  torch.from_numpy(data['data'].reshape(data['data'].shape[0],-1).astype(np.float32)).cuda()-self.mu

        # self.x_mean = torch.nn.Parameter(torch.randn(x.shape[0],99))
        # self.x_logvar =  torch.nn.Parameter(torch.randn(x.shape[0],99))
        # self.s = torch.nn.Parameter(torch.randn(1))
        # self.s.data.fill_(1)

        self.x_mean = torch.nn.Parameter(torch.randn(self.num_sample,self.dim))
        self.x_logvar =  torch.nn.Parameter(torch.randn(self.num_sample,self.dim))
        self.s = torch.nn.Parameter(torch.randn(1))
        self.s.data.fill_(1)
        self.A = torch.nn.Parameter(torch.randn(self.dim,self.B.shape[1]))
        self.centers = torch.nn.Parameter(torch.randn(self.num_cls,self.dim))
        self.batch = 400
        

        self.params = nn.ParameterList()
        for param in self.parameters():
            self.params.append(param)
        print(self.params)


    def load_data(self,x,y):
        num = x.shape[0]
        x = torch.from_numpy(x.astype(np.float32).reshape(num,-1)- self.mu).float()
        #x = x - x.mean(dim=0) 
        #print(x)
        y = torch.from_numpy(y.astype('int')).reshape(num)
        self.num_cls = max(y)+1
        self.B = x.cuda()
        self.labels = y.long().cuda()
        self.x_mean.data = torch.randn(self.B.shape[0],self.dim).cuda()
        self.batch = num 
        self.num_sample = num
        self.batch = num
        #num_y = y.shape
        
    def forward(self):
        idx = random.sample(range(0,self.num_sample),self.batch) 
        r = ((self.A @ self.A.t() - self.E)**2).mean()
        #cl,cmd = self.center_loss(self.labels[idx],self.x_mean[idx])

        #######################################
        # loss=(((self.x_mean[idx] @ self.A - self.B[idx])**2).reshape(600,-1,3).sum(dim=-1)**0.5).mean()
        # KL_loss =0
        ############ vae #########################
        if self.center:
            cl = self.center_loss(self.labels[idx],self.x_mean[idx])
        else:
            cl = 0
        if self.vae:
            x = self.sampling([self.x_mean[idx],self.x_logvar[idx]]).cuda()*self.s
            loss=((x @ self.A - self.B[idx])**2).mean()
            KL_loss =   1 + (self.x_logvar[idx]) - (self.x_mean[idx])**2 - torch.exp(self.x_logvar[idx])
            KL_loss = (-0.5*torch.sum(KL_loss,dim=-1)).mean()
        else:
            loss=((self.s*self.x_mean[idx] @ self.A - self.B[idx])**2).mean()
            KL_loss =0
            


        return loss, r, cl, KL_loss


    def sampling(self,args):
        z_mean, z_log_var = args
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch_size,dim).cuda()
        return z_mean + torch.exp(0.5*z_log_var) * epsilon

    def cal_error(self):
        sum_err=0
        for idx in range(0,self.num_sample):
            loss=(((self.s * self.x_mean[idx] @ self.A  - self.B[idx])**2).reshape(-1,3).sum(dim=-1)**0.5).mean()
            sum_err += loss
        return sum_err/self.num_sample

    def cal_error_test(self):
        pass


    def center_loss(self,y,hidden):
        #batch_size = hidden.size(0)
        expanded_centers = self.centers.index_select(dim=0,index=y)

        #intre_diff = self.centers.expand(self.num_cls,self.num_cls,self.dim)
        #intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()

        
        
        intra_distances = torch.sum((hidden - expanded_centers)**2, dim=1).mean()
        #re = intra_distances/intre_diff
        #print("cl",expanded_centers.norm(dim=1).shape)
        #cmd = intra_distances / (expanded_centers.norm(dim=1).mean())

        return intra_distances

    def get_single_NMCD(self,ix,iy):
        data = torch.zeros(self.num_cls,self.dim).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x_mean[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())
        
        inner_diff = ((self.x_mean[ix]-self.x_mean[iy])**2).mean()
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.dim)
        
        #print(intre_diff)
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
            #print(intre_diff/(self.x.norm(dim=1).mean()**2))
        re = inner_diff/intre_diff
        return re


    def cal_NMCD(self):
        data = torch.zeros(self.num_cls,self.dim).cuda()
        count = torch.zeros(self.num_cls).long().cuda()
        for idx,label in enumerate(self.labels):
            data[label,:] += self.x_mean[idx]
            count[label]+=1
        centers = data / ((count+1e-8).reshape(-1,1).float())
        #print(centers)
        inner_diff = ((self.x_mean[:self.num_sample] - centers.index_select(dim=0, index = self.labels))**2).mean()
        print(inner_diff/(centers**2).mean())
        #print(inner_diff)
        intre_diff = centers.expand(self.num_cls,self.num_cls,self.dim)
        #print((intre_diff- intre_diff.transpose(0,1)).norm(dim=-1).mean())
        intre_diff = ((intre_diff- intre_diff.transpose(0,1))**2).mean()
        
        print(intre_diff/(centers**2).mean())
        re = inner_diff/intre_diff
        return re


    def getmesh(self,idx):
        err = (((self.s * self.x_mean[idx] @ self.A  - self.B[idx])**2).reshape(-1,3).sum(dim=-1)**0.5).mean()
        print(err)
        return self.mu+(self.s * self.x_mean[idx] @ self.A).data.cpu().numpy(), err.data.cpu().numpy()
    def get_data(self):
        return self.x_mean.data.cpu().numpy(),self.labels.cpu().numpy()

    def ground_truth_mesh(self,idx):
        return self.mu+self.B[idx].data.cpu().numpy()

    def discrimination_metric(self):
        X = self.x_mean.data.cpu().numpy()
        labels = self.labels.data.cpu().numpy()
        #print(X.shape,labels.shape)
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
        davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
        return silhouette_score,calinski_harabasz_score,davies_bouldin_score



def write_obj_without_face(filename,vertices):
        outdir = os.path.dirname(filename)
        # if not os.path.exists(outdir):
        #     os.mkdir(outdir)
        num_vertex = vertices.shape[0]
        #num_index = index.shape[0]
        fobj=open(filename,'w+')
        fobj.write("# {} vertices, \n".format(num_vertex))
        #print(vertices[0])
        for i in range(num_vertex):
            fobj.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
        # for i in range(num_index):
        #     fobj.write("f {} {} {}\n".format(index[i][0], index[i][2], index[i][1]))
        fobj.close()



def test(model):
    import pickle
    dataz =np.load('init.npz')
    faces = np.load('faces.npy')
    A = dataz['ceof']
    #x = dataz['x']
    mu = dataz['x_mean']
    gmm = pickle.load(open('./gmm.pkl','rb'))
    #gmm_data = np.load('../propressing/gmm.npy')
    x,y = gmm.sample(600)
    print(x)
    x = torch.Tensor(x).cuda()
    print(x.norm(dim=1))
    print(x.norm(dim=1).mean())
    print(x.norm(dim=1).std())
    v = x[:,:99] @ model.A[:99,:]
    print(v[9,:10])
    x = x.data.cpu().numpy()

    #x = model.x.data.cpu().numpy()
    A = model.A.data.cpu().numpy()
    #draw_curve_ced(model.x.data.cpu().numpy())

    vertices =(v[9].data.cpu().numpy()+mu).reshape(-1,3)
    v2 = (model.B[600].data.cpu().numpy()+mu).reshape(-1,3)

    
    np.savez('f_model.npz',mu=mu,faces = faces, x = x, w_shape = A)

    
    write_obj('3.obj',vertices,faces+1)
    write_obj('2.obj',v2,faces+1)

def test2():
    v,f = readobj2('meanface.obj')
    info = "/home/jdq/projects/data/model/model_info.mat"
    infos = scio.loadmat(info)
    landmarks_index =  infos['keypoints'][0]
    #############################################
    # net = pca_model(dim=199)
    # net.load_state_dict(torch.load('pca_199.pkl'))
    # data = np.load('p199_model.npz')
    # mu = data['mu']
    # shape_ev = data['shape_ev']
    # w_shape = data['w_shape']
    # x = (net.x).data.cpu().numpy()
    ##############################################
    # net = linear_model().cuda()
    # net.load_state_dict(torch.load('test_R1E4C1.pkl'))
    # x = (net.x_mean*net.s).data.cpu().numpy()
    # shape_ev = np.std(x)
    # w_shape = net.A.data.cpu().numpy()
    # mu = net.mu
    ###################################################

    net = model_non_vae().cuda()
    net.load_state_dict(torch.load('/data0/jdq/L2D199C3.pkl'))
    mlc = net.mlc
    #############################################
    # shape_basis = net.A.data.cpu().numpy()
    x = (net.x_mean).data.cpu().numpy()
    shape_ev = np.std(x)
    norm = np.linalg.norm(x,axis=1)
    print('norm:',norm.shape,np.mean(norm))
    mu = net.mu
    # print(x[0])
    # print(x[0]/shape_ev)
    #s

    # # print(np.std(x))
    # #x = np.random.normal(0, 1, size=1000)
    # x = (x-np.mean(x,axis=0))/np.std(x,axis=0)
     
    k2, p = stats.normaltest(x/shape_ev)
    #print(stats.kstest(x, 'norm'))
    
    print(p.mean())
    print(p.shape)
    print(x.shape)
    

    torch.save(mlc.state_dict(), '../propressing/C_nonlinear_mlc.pkl')
    np.savez('../propressing/C_nonlinear_model.npz',mu=mu,shape_ev = shape_ev,faces = f,x=x, landmarks=landmarks_index, norm = np.mean(norm), normal_test = p.mean())


def test3(server):
    mode = 'vae-center'
    use_cuda = True
    Test = False
    test_file_name = 'V5'
    chech_point = '/data0/jdq/'+test_file_name+'.pkl'
    save_name = 'test_'+test_file_name+'.pkl'
    if Test:
        net = linear_model(server=server,dim=199,vae=False)
        if server == 'lab':
            net.load_state_dict(torch.load(chech_point))
        else: 
            net.load_state_dict(torch.load('/data/jdq/vae_00001.pkl'))
        if use_cuda:
            net = net.cuda()
        optimizer = torch.optim.Adam([{'params':net.params[0],'lr':0.2},{'params':net.params[1],'lr':0.0},{'params':net.params[2:]}], lr=0.0, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,min_lr=1e-6)
    else:
        if mode == 'vae':
            print('======VAE======')
            net = linear_model(server=server,dim=199,vae= True)
        elif mode == 'center':
            print('=======CENTER=======')
            net = linear_model(server=server,dim=199,vae= False,center=True)
        else:
            print('=======VAE==CENTER=======')
            net = linear_model(server=server,dim=199,vae= True,center=True)
        if server == 'lab':
            pass
            #net.load_state_dict(torch.load('/data0/jdq/R1E5.pkl'))
        else: 
            net.load_state_dict(torch.load('vae_init.pkl'))
        if use_cuda: 
            net = net.cuda()
        
        optimizer = torch.optim.Adam([{'params':net.params[0:2],'lr':0.2},{'params':net.params[2],'lr':0.2,'weight_decay':1e-10},{'params':net.params[-1],'lr':1},{'params':net.params[3:-1]}], lr=0.02, weight_decay=1e-6)
        #optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-10)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,min_lr=0)

    # net.params[2:].requires_grad = False
    #trainset = FRGC_DataLoader(path = 'frgc2.npz')
    #train_loader = DataLoader(trainset,batch_size=1024, shuffle = True)
    if Test:
        iter_num = 4000
    else:
        iter_num = 40000


    r_weight = 0
    w_cl = 0.000
    w_kl = 0
    for k in range(iter_num):
        #for batch_idx, data ,label in enumerate(train_loader):
            #data = data.cuda()
            #label = label.cuda()
        
        d,r,cl,kl = net()
        if Test:
            loss = d
        elif mode == 'vae-center':
            loss = d + r*r_weight +cl*w_cl + kl*w_kl
        else:
            loss = d + r*r_weight +cl*w_cl #+ kl*0.0001
        #print(net.A)
        # if k<2500:
        #     loss = d*1+r*k/20
        # else:
            #  loss = d*1+r*k/20+kl*0.001
        #print(loss)
        if (k+1) % 100 == 0:
            print(optimizer.param_groups[0]['lr'])
            scheduler.step(loss)
            print('Iter:',k+1)
            print(loss,r,kl,net.s)
            print('NET:',net.cal_error())
            print('CET:',net.cal_NMCD())
            print('Sot:',net.discrimination_metric())
            if optimizer.param_groups[0]['lr']<1e-7:
                torch.save(net.state_dict(), '/data0/jdq/iter_l_d199_R'+str(r_weight)+'_C'+str(w_cl)+'_V'+str(w_kl)+'_final' +'.pkl')
                break
            #scheduler.step()
            #loss = d
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (k+1) % 10000 == 0:
            
            print('Saving....')
            if not Test:
                if server == 'lab':
                    torch.save(net.state_dict(), '/data0/jdq/iter_l_d199_R'+str(r_weight)+'_C'+str(w_cl)+'_V'+str(w_kl)+'_%d' % k +'.pkl')
                else:
                    torch.save(net.state_dict(), '/data/jdq/iter_l_d199_vae_00001_%d' % k +'.pkl')
        
    if Test:
        torch.save(net.state_dict(), save_name)
            

        #if (k+1) % 5000 == 0:
            #print('Saving....')
            #torch.save(net.state_dict(), 'iter_x_%d' % k +'.pkl')

def test_pca():
    use_cuda = True
    net = pca_model(dim=199)
    if use_cuda: 
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=2, weight_decay=5e-12)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,min_lr=1e-6)
    for k in range(5000):
        loss = net()
        if (k+1) % 100 == 0:
            scheduler.step(loss)
            print('Iter:',k+1)
            print(loss)
            print('NET:',net.cal_error())
            print('CET:',net.cal_NMCD())
            print('Sot:',net.discrimination_metric())
            
            #scheduler.step()
            #loss = d
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(net.state_dict(), 'pca_199.pkl')


    


def test_ex():
    if server == 'own':
        data = np.load('test.npz')
    else:
        data = np.load('/data0/jdq/test.npz')
    x_data = data['x']
    labels = data['y']
    # net = pca_model().cuda()
    # net.load_data(x_data,labels)
    # fitting_pca(net)
    
    # exit()
    # net = linear_model().cuda()
    # net.load_state_dict(torch.load('test_R1E4C1.pkl'))
    # net.load_data(x_data,labels)
    # fitting_pca(net)

    net = model_non_vae(center=True).cuda()
    net.load_state_dict(torch.load('L3E8D199C3.pkl'))
    net.load_data(x_data,labels)
    fitting_pca(net)




def fitting_pca(net):
    #optimizer = optimizer = torch.optim.Adam(net.parameters(), lr=0.5, weight_decay=5e-12)
    optimizer = torch.optim.Adam([{'params':net.params[1],'lr':0.5},{'params':net.params[2:],'lr':0.0},{'params':net.params[0],'lr':1}], lr=0.0, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,min_lr=1e-6)
    for k in range(10000):
        #loss,  KL_loss, cl  = net()
        #loss = net()
        loss,_,_ = net()
        # if k<1000:
        #     loss = loss + cl*0.01
        if (k+1) % 100 == 0:
            print(optimizer.param_groups[0]['lr'])
            scheduler.step(loss)
            print('Iter:',k+1)
            print(loss)
            print('NET:',net.cal_error())
            print('CET:',net.cal_NMCD())
            print('Sot:',net.discrimination_metric())
            #scheduler.step()
            #loss = d
            #optimizer.step(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), 'test_L3E8D199C3_bos.pkl')



def pca():
    pass
    

class FRGC_DataLoader(Dataset):
    def __init__(self,path):
        metedata = np.load(path)
        data = metedata['data']
        num_sample = data.shape[0]
        self.data = self.data.reshape(num_sample,-1)
        self.label = metedata['label'].reshape(num_sample,-1)
    def __getitem__(self,index):
        x = self.data[index]
        y = self.label[x]
        return x,y


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Recognizable Basis')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    # parser.add_argument('--verify', '-v', default=False, action='store_true', help='Verify the net')
    # parser.add_argument('--gpu', default="0", help="select the gpu")
    # parser.add_argument('--net', default="sphere", help="type of network")
    # parser.add_argument('--number_of_class','-nc', default=8631,type=int, help="The number of the class")
    # parser.add_argument('--loadfile', '-l' , default="/data3/jdq/fs2_81000.cl", help="model parameter filename")
    # parser.add_argument('--savefile', '-S' , default="../dict.cl", help="model parameter filename")
    # parser.add_argument('--param-fp-train', default='./train.configs/param_aligned.pkl')
    # parser.add_argument('--filelists-train', default='./train.configs/train_aug_120x120.list.train')
    # parser.add_argument('--epoch', '-e' , default=50, help="training epoch")
    # parser.add_argument('--lfw_vallist', '-vl' , default="/data1/jdq/lfw_crop/")
    # parser.add_argument('--lfw_pairlist', '-pl' , default="../lfw_pair.txt")




    # print("*************")
    # args = parser.parse_args()



    # # pass
    # test2()
    
    
    server = 'own'
    mode =  'center'
    # test_ex()  
    # # # test_pca()
    # # # # # test3(server)
    # exit()
    data_root = './data'
    use_cuda = True
    Test = False
    w_th = 199
    w_kl = 0
    w_cl = 1e-3
    test_file_name = 'L3D199C3'
    if server == 'own':
        data_root = './'
    chech_point = data_root+test_file_name+'.pkl'
    save_name = 'test_'+test_file_name+'.pkl'
    if Test:
        net = model_non_vae(vae=False)
        net.load_state_dict(torch.load(chech_point))

        if use_cuda:
            net = net.cuda()
        optimizer = torch.optim.Adam([{'params':net.params[1],'lr':0.02},{'params':net.params[0],'lr':0.0},{'params':net.params[2:]}], lr=0.0, weight_decay=1e-9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,min_lr=1e-6)
    else:
        if mode == 'normal':
            print('======NORMAL======')
            net = model_non_vae()
        elif mode == 'vae':
            print('======VAE======')
            net = model_non_vae(vae= True)
        elif mode == 'center':
            print('=======CENTER=======')
            net = model_non_vae(vae= False, center=True)
            if server == 'own':
                net.load_state_dict(torch.load('L3E8D199.pkl'))
            else: 
                net.load_state_dict(torch.load('vae_init.pkl'))
        else: 
            print('=======VAE+CENTER=======')
            net = model_non_vae(vae= True, center=True)

        # if server == 'lab':
        #     net.load_state_dict(torch.load('/data0/jdq/L2D199C3.pkl'))
        # else: 
        #     net.load_state_dict(torch.load('vae_init.pkl'))
        if use_cuda: 
            net = net.cuda()
        optimizer = torch.optim.Adam([{'params':net.params[0],'lr':1},{'params':net.params[1:3]},{'params':net.params[3:],'weight_decay':1e-8}], lr=0.02,weight_decay=1e-12)
        #optimizer = torch.optim.SGD(net.parameters(), lr= 0.03,weight_decay=5e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=8,min_lr=0)

    # test_pca()
    # exit()
    # test3(server)
    # exit()
    
    
    #net = model_non_vae(width=w_th)
    #net.load_state_dict(torch.load('icl00001.pkl'))
    # if use_cuda:
    #     net = net.cuda()
   

    # test2(net)
    # exit()

    # dataset = data_loader()
    # dataloader = torch.utils.data.DataLoader(dataset,batch_size=200,shaffle=True)
    #optimizer = torch.optim.Adam(net.parameters(), lr= 0.01,weight_decay=5e-8)
    #optimizer = torch.optim.SGD(net.parameters(), lr= 0.03,weight_decay=5e-6)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,min_lr=0)
    #net.params[2:].requires_grad = False
    if Test:
        iter_num = 4000
    else:
        iter_num = 40000
    for k in range(iter_num):
        # for x,y in dataloader:
        
            # print('################## G ##################')
        # print(k)
        #net.params[0:2].requires_grad = False
        d,  KL_loss, cl  = net()
        if Test or mode == 'normal':
            loss = d
        elif mode == 'vae':
            loss = d + KL_loss*w_kl
        elif mode == 'center':
            loss = d  + cl*w_cl
        else:
            loss = d  + cl*w_cl+KL_loss*w_kl

        if (k+1) % 100 == 0:
            print(optimizer.param_groups[0]['lr'])
            scheduler.step(loss)
            print('Iter:',k+1)
            print(d,  KL_loss, cl)
            print('NET:',net.cal_error())
            print('CET:',net.cal_NMCD())
            print('Sot:',net.discrimination_metric())
            if optimizer.param_groups[0]['lr']<1e-7:
                torch.save(net.state_dict(), data_root+'iter_l_d199'+'_w%d' % w_th +'_V%f' % w_kl+'_C%f' % w_cl+'_final' +'.pkl')
                break
        #loss = d + 0.001*cl #+ 0.01*kl_loss
        #loss = re #+ 0.00001*cl 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

        if (k+1) % 10000 == 0:
            #scheduler.step()
            print('Saving....')
            torch.save(net.state_dict(), data_root+'iter_l2_d199'+'_w%d' % w_th+'_V%f' % w_kl +'_C%f' % w_cl+'_%d' % k +'.pkl')

    if Test:
        torch.save(net.state_dict(), save_name)
    #     # r = 1000
    #     # d = 0
        
    #     # net.params[1].requires_grad = False
    #     # net.params[0].requires_grad = True
    #     # while(r>d/2):
    #     #     d,r, re  = net()
    #     #     loss = 2 * d + 0.1*r
    #     #     optimizer.zero_grad()
    #     #     loss.backward()
    #     #     optimizer.step()
        
    #     # r = 0
        # d =1000
        # # net.params[0].requires_grad = False
        # # net.params[1].requires_grad = True
        # while(d>r):
        #     d,r ,re  = net()
        #     loss = 1000000* d + 2*r
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print(loss,d,r,re)
        # print('################# D ###################')

        
