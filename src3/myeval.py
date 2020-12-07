import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import os
import sys
import lfw
import data_loader
from torchvision import datasets, transforms
import sklearn
from PIL import Image
import torch.nn.functional as F
from data_loader import FeatureDataset
import bcolz
from utils import gen_plot 
from  verifacation import evaluate as eval_ver
import config

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
 )


#normalize = transforms.Normalize(
#   mean=[0.5, 0.5, 0.5],
#   std=[0.5, 0.5, 0.5]
#)

transform_eval=transforms.Compose([
    
        transforms.CenterCrop((112,96)),
        transforms.ToTensor(),
        normalize
    ])

transform_eval_f=transforms.Compose([
    
        transforms.RandomHorizontalFlip(p=1),
        transforms.CenterCrop((112,96)),
        transforms.ToTensor(),
        normalize
    ])

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)

def calROC(pred,y):

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def margin_observe(feat,y,num_cls):
    y = y.view(-1, 1)
    batch_size = feat.size()[0]
    y_onehot = torch.Tensor(batch_size, num_cls).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y.data.view(-1, 1), 1)
    y_onehot.byte()
    y_onehot = Variable(y_onehot)
    max_value = torch.max(where(y_onehot, -1, feat),dim=1)[0]
    y_value = feat.gather(1,y.data.view(-1, 1))
    margin=(y_value-max_value).mean()
    return margin

def show_margin_var(result_file):
    margins, iters, accs=load_result(result_file)
    if "coco" in result_file:
        plt.plot(accs, margins,label = 'NSL')
    elif "cos" in result_file:
        plt.plot(accs, margins,label = 'Cosine loss')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Actual Margin')
    plt.xlabel('Accuracy(%)')
    # plt.xlim([0.5, 100])
    
def show_margin_train(result_file):
    # margins=[]
    # accs=[]
    
    margins,iters, accs=load_result(result_file)
    plt.plot(iters,accs,label =os.path.split(result_file)[-1])
    plt.legend(loc = 'lower right')
    # if "coco" in result_file:
    #     plt.plot(accs, margins,label = 'NSL')
    # elif "cos" in result_file:
    #     plt.plot(accs, margins,label = 'Cosine loss(m=0.35)')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')WW
    plt.ylabel('Accuracy(%)')
    plt.xlabel('Iteration times')
    # plt.xlim([0, 30000])
    #plt.ylim([0, 70])

def gen_feature(model,dataloader, to_file = True):
    model.eval()
    feat_dict = {}
    for batch_idx, (imgnames,data,dataf) in enumerate(dataloader):
        data , dataf = data.cuda(), dataf.cuda()
        feats, _ , _ = model(data)
        featsf, _ , _ = model(dataf)
        feats = (feats+featsf)/2
        feats= feats.data.cpu().numpy()
        for idx,imgname in enumerate(imgnames):
            imgpath = imgname
            imgpath = imgpath.replace(".jpg",".bin")
            feat_dict[imgpath] = feats[idx]
            #feat_dict[imgpath] = np.ones(512)
            if to_file:
                feats[idx].tofile(imgpath)
            sys.stdout.write(str(batch_idx)+'/'+str(len(dataloader))+'\r')
    return feat_dict

def mega_feature(model,eval_loader_facescrub,eval_loader_megaface,megaface_out,facescrub_out):
    model.eval()
    print("Generate Feature of facescrub")
    for batch_idx, (imgnames,data,dataf) in enumerate(eval_loader_facescrub):
        data , dataf = data.cuda(), dataf.cuda()
        feats, _ , _ = model(data)
        featsf, _ , _ = model(dataf)
        feats = (feats+featsf)/2
        feats= feats.data.cpu().numpy()
        feats = feats / np.linalg.norm(feats)
        for idx,imgname in enumerate(imgnames):
            imgpath = imgname
            _path = imgpath.split('/')
            a1,  b = _path[-2], _path[-1]
            out_dir = os.path.join(facescrub_out, a1)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, b+"_%s_%dx%d.bin"%('sphereface', 112, 96))
            feats[idx].tofile(out_path)
            sys.stdout.write(str(batch_idx)+'/'+str(len(eval_loader_facescrub))+'\r')
    print("Generate Feature of Megaface:")
    for batch_idx, (imgnames,data,dataf) in enumerate(eval_loader_megaface):
        data , dataf = data.cuda(), dataf.cuda()
        feats, _ , _ = model(data)
        featsf, _ , _ = model(dataf)
        feats = (feats+featsf)/2
        feats= feats.data.cpu().numpy()
        feats = feats / np.linalg.norm(feats)
        for idx,imgname in enumerate(imgnames):
            imgpath = imgname
            _path = imgpath.split('/')
            a1, a2, b = _path[-3], _path[-2], _path[-1]
            out_dir = os.path.join(megaface_out, a1, a2)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, b+"_%s_%dx%d.bin"%('sphereface', 112, 96))
            feats[idx].tofile(out_path)
            sys.stdout.write(str(batch_idx)+'/'+str(len(eval_loader_megaface))+'\r')
        
    


def eval_megaface(model,root_facescrub,root_megaface,facescrub_out,megaface_out,fdict):
    state = torch.load(fdict)
    state_dict = state['dict']
    model.load_state_dict(state_dict)
    print("Start evaling MegaFace...")
    evalset_facescrub = FeatureDataset(root=root_facescrub,filelist="files.txt",transform=transform_eval,target_transform=transform_eval_f)
    eval_loader_facescrub = torch.utils.data.DataLoader(evalset_facescrub, batch_size=64, shuffle=False, num_workers=16)
    evalset_megaface = FeatureDataset(root=root_megaface,filelist="files.txt",transform=transform_eval,target_transform=transform_eval_f)
    eval_loader_megaface= torch.utils.data.DataLoader(evalset_megaface, batch_size=64, shuffle=False, num_workers=16)
    mega_feature(model,eval_loader_facescrub,eval_loader_megaface,megaface_out,facescrub_out)




def eval_cfp_fp(model,root,fdict = ''):
    preds=[]
    #state = torch.load(fdict)
    #state_dict = state['dict']
    #model.load_state_dict(state_dict)
    print("Start evaling CFP-FP...")
    evalset = FeatureDataset(root,filelist="../../file_path_list_cfp_align.txt",transform=transform_eval,target_transform=transform_eval_f)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=32, shuffle=False, num_workers=72)
    labels=np.load('../cfp_fp_list.npy')
    print(labels)
    exit()
    gen_feature(model,eval_loader)
    labels=np.load('../cfp_fp_list.npy')

    for i in range(7000):
        f_feat = np.fromfile(os.path.join(root,'%d.bin' % (2*i)),dtype='float32')
        p_feat = np.fromfile(os.path.join(root,'%d.bin' % (2*i + 1)),dtype='float32')
        f_feat=f_feat/np.linalg.norm(f_feat)
        p_feat=p_feat/np.linalg.norm(p_feat)
        cosdistance=np.dot(f_feat,p_feat)
        preds.append(cosdistance)
    preds=np.array(preds).flatten()
    accuracy = []
    thd = []
    folds = lfw.KFold(n=7000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        #print(idx, (train, test))
        #print(predicts.shape, labels.shape)
        best_thresh = lfw.find_best_threshold(thresholds, preds[train],labels[train])
        accuracy.append(lfw.eval_acc(best_thresh, preds[test],labels[test]))
        thd.append(best_thresh)
    print('CFP-FP ACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    for i in range(fpr.shape[0]):
        if fpr[i] > 1e-3:
            print('CFP-FP@10-3 ACC={:.4f}'.format(tpr[i]))
            break



def eval_cfp(model, root, fdict):
    # # predicts = []
    # labels=[]
    state = torch.load(fdict)
    state_dict = state['dict']
    model.load_state_dict(state_dict)
    print("Start evaling CFP...")
    evalset = FeatureDataset(root="/home/jdq/database/database/cfp_crop",filelist="files.txt",transform=transform_eval,target_transform=transform_eval_f)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=64, shuffle=False, num_workers=72)
    gen_feature(model,eval_loader)
    if not os.path.exists(root):
        raise ValueError("Cannot find the data!")
    def num2path():
        pair_list_f, pair_list_p = {}, {}
        with open(os.path.join(root , 'Protocol/Pair_list_F.txt')) as f:
            lines = f.readlines()
            for pair in lines:
                pair = pair[:-1].split()
                pair_list_f[int(pair[0])] = pair[1].replace(".jpg","_crop.bin").replace("..",root)
        with open(os.path.join(root , 'Protocol/Pair_list_P.txt')) as f:
            lines = f.readlines()
            for pair in lines:
                pair = pair[:-1].split()
                pair_list_p[int(pair[0])] = pair[1].replace(".jpg","_crop.bin").replace("..",root)
        return pair_list_f, pair_list_p

    pair_list_f, pair_list_p = num2path()	

    labels = np.zeros(7000)
    for i in range(7000):
        if (i // 350) % 2 == 1:
            labels[i] = 1
    preds = []
    for i in range(1, 11):
        #print(i)
        diff_path = os.path.join(root , 'Protocol/Split/FP/%02d/diff.txt' % i)
        same_path = os.path.join(root , 'Protocol/Split/FP/%02d/same.txt' % i)
        with open(diff_path) as f:
            lines = f.readlines()
            for pair in lines:
                pair = pair[:-1].split(',')
                pair = (int(pair[0]),int(pair[1]))
                if os.path.exists(pair_list_f[pair[0]]) and os.path.exists(pair_list_p[pair[1]]):
                    f_feat = np.fromfile(pair_list_f[pair[0]],dtype='float32')
                    p_feat = np.fromfile(pair_list_p[pair[1]],dtype='float32')
                    f_feat=f_feat/np.linalg.norm(f_feat)
                    p_feat=p_feat/np.linalg.norm(p_feat)
                    cosdistance=np.dot(f_feat,p_feat)
                    preds.append(cosdistance)
                else:
                    preds.append(-1.)
                    
        with open(same_path) as f:
            lines = f.readlines()
            for pair in lines:
                pair = pair[:-1].split(',')
                pair = (int(pair[0]),int(pair[1]))
                #print(pair_list_f[pair[0]],pair_list_p[pair[1]])
                if os.path.exists(pair_list_f[pair[0]]) and os.path.exists(pair_list_p[pair[1]]):
                    f_feat = np.fromfile(pair_list_f[pair[0]],dtype='float32')
                    p_feat = np.fromfile(pair_list_p[pair[1]],dtype='float32')
                    f_feat=f_feat/np.linalg.norm(f_feat)
                    p_feat=p_feat/np.linalg.norm(p_feat)
                    cosdistance=np.dot(f_feat,p_feat)
                    preds.append(cosdistance)
                else:
                    preds.append(1.)			
    preds=np.array(preds).flatten()
    #print(preds)
    # fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)

    # if draw_roc:
    #     plt.title('Receiver Operating Characteristic CFP-FP')
    #     roc_auc = metrics.auc(fpr, tpr)
    #     plt.plot(np.log10(fpr), tpr, label = loss_type + ' AUC = %0.2f' % roc_auc)
        
    #     plt.legend(loc = 'lower right')
    #     plt.xlim([-2, 0])
    #     plt.ylim([0.7, 1])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate(log10)')
        
    # if cal_far:
    #     for i in range(fpr.shape[0]):
    #         if fpr[i] > 1e-3:
    #             print(tpr[i])
    #             return
    #     exit()


    #print(preds, preds.shape)
    accuracy = []
    thd = []
    folds = lfw.KFold(n=7000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        #print(idx, (train, test))
        #print(predicts.shape, labels.shape)
        best_thresh = lfw.find_best_threshold(thresholds, preds[train],labels[train])
        accuracy.append(lfw.eval_acc(best_thresh, preds[test],labels[test]))
        thd.append(best_thresh)
    print('CFP-FP ACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

def eval_ytf(model, root ,splitfile='../splits_corrected.txt', fdict = None):
    def gen_feature(model,dataloader, to_file = True):
        model.eval()
        feat_dict = {}
        for batch_idx, (imgnames,data,dataf) in enumerate(dataloader):
            data , dataf = data.cuda(), dataf.cuda()
            feats, _ , _ = model(data)
            featsf, _ , _ = model(dataf)
            feats = (feats+featsf)/2
            feats= feats.data.cpu().numpy()
            for idx,imgname in enumerate(imgnames):
                imgpath = imgname
                imgpath = imgpath.replace(".jpg",".bin")
                feat_dict[imgpath] = feats[idx]
                #feat_dict[imgpath] = np.ones(512)
                if to_file:
                    feats[idx].tofile(imgpath)
                sys.stdout.write(str(batch_idx)+'/'+str(len(dataloader))+'\r')
        return feat_dict
    
    predicts = []
    labels=[]
    if fdict is not None:
        state = torch.load(fdict)
        state_dict = state['dict']
        model.load_state_dict(state_dict)
    print("Start evaling YTF...")
    evalset = FeatureDataset(root=root,filelist="files.txt",transform=transform_eval,target_transform=transform_eval_f)
    # # evalset = YTFDataSet(root="/home/jdq/database/database/",pairfile="../splits.txt",transform=transform_eval)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=64, shuffle=False, num_workers=16)
    # print(len(eval_loader))
    feat_dict = gen_feature(model,eval_loader, False)
    if not os.path.exists(root):
        raise ValueError("Cannot find the data!")
    with open(splitfile) as f:
        split_lines = f.readlines()[1:]
    for batch_idx,line in enumerate(split_lines):
        words = line.replace('\n', '').split(',')
        name1 = os.path.join(root,words[2])
        name2 = os.path.join(root,words[3])
        sameflag = int(words[5])

        # Calculate feature one 
        featnames=os.listdir(name1)
        feats1=[]
        #print(featnames)
        for featname in featnames:
            if os.path.splitext(featname)[-1]=='.bin':
                #print(featname)
                featname = os.path.join(name1,featname)
                feat=np.fromfile(featname,dtype='float32')
                #feat=feat/np.linalg.norm(feat)
                feats1.append(feat)
        feats1=np.array(feats1)
        mean_feat1=feats1.mean(axis=0, keepdims=True)
        mean_feat1=sklearn.preprocessing.normalize(mean_feat1).flatten()
        #print(mean_feat1.shape)
        mean_feat1 = mean_feat1/np.linalg.norm(mean_feat1)
        #mean_feat1=feats[0]
        # Calculate feature two 
        featnames=os.listdir(name2)
        #print(featnames)
        feats2=[]
        for featname in featnames:
            if os.path.splitext(featname)[-1]=='.bin':
                featname = os.path.join(name2,featname)
                feat =  feat_dict[featname]
                #feat=np.fromfile(featname,dtype='float32')
                #feat=feat/np.linalg.norm(feat)
                feats2.append(feat)
        feats2=np.array(feats2)
        mean_feat2=feats2.mean(axis=0,keepdims=True)
        mean_feat2=sklearn.preprocessing.normalize(mean_feat2).flatten()
        #print(mean_feat2.shape)
        #mean_feat2 = mean_feat2/np.linalg.norm(mean_feat2)
        #mean_feat2=feats[0]

        # Calculate the cosine distance
        #print((mean_feat1*mean_feat2).sum())
        cosdistance=np.dot(mean_feat1,mean_feat2)/np.linalg.norm(mean_feat1)/np.linalg.norm(mean_feat2)
        predicts.append(cosdistance)
        labels.append(int(sameflag))
        #print(cosdistance,sameflag)
        sys.stdout.write(str(batch_idx)+'/'+str(len(split_lines))+'\r')
       

    predicts=np.array(predicts).flatten()
    #print(predicts.shape)
    labels = np.array(labels).flatten()     

    accuracy = []
    thd = []
    folds = lfw.KFold(n=5000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        #print(predicts.shape, labels.shape)
        best_thresh = lfw.find_best_threshold(thresholds, predicts[train],labels[train])
        accuracy.append(lfw.eval_acc(best_thresh, predicts[test],labels[test]))
        thd.append(best_thresh)
    print('YTFACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

def eval_lfw(model,root,dicts="",train=False):
    evalset = data_loader.LfwDataSet(root=root,pairfile="../pair.txt",transform=transform_eval,target_transform=transform_eval_f)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=16, shuffle=False, num_workers=24)
    if not train:
        if os.path.isdir(dicts): 
            dict_files=os.listdir(dicts)
            for dict_file in dict_files:
                if os.path.splitext(dict_file)[-1]=='.cl':
                    filename=os.path.join(dicts,dict_file)
                    state = torch.load(filename)
                    state_dict = state['dict']
                    model.load_state_dict(state_dict)
                    model.eval()
                    predicts = []
                    labels=[]
                    for batch_idx,(_,_2,data1,data2,data1f, data2f,target) in enumerate(eval_loader):
                        data1, data2, data1f, data2f,target = data1.cuda(),data2.cuda(),data1f.cuda(),data2f.cuda(),target.cuda()
                        ip1, _ , _ = model(data1)
                        ip1f, _ , _ , _ = model(data1f)
                        ip2, _ , _ = model(data2)
                        ip2f, _ , _ = model(data2f)
                        # ip1=F.normlize(ip1)
                        ip1 = (ip1+ip1f)/2
                        ip2 = (ip2+ip2f)/2
                        cosdistance=ip1*ip2     
                        cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
                        label=target.data.cpu().numpy()
                        cosdistance=cosdistance.data.cpu().numpy()
                        predicts.append(cosdistance)
                        labels.append(label)
                        sys.stdout.write(str(batch_idx)+'/'+str(len(eval_loader))+'\r')
                        sys.stdout.flush()
                    predicts=np.array(predicts).reshape(6000,-1)
                    labels=np.array(labels).reshape(6000,-1)

                    accuracy = []
                    thd = []
                    folds = lfw.KFold(n=6000, n_folds=10)
                    thresholds = np.arange(-1.0, 1.0, 0.005)
                    for idx, (train, test) in enumerate(folds):
                        best_thresh = lfw.find_best_threshold(thresholds, predicts[train],labels[train])
                        accuracy.append(lfw.eval_acc(best_thresh, predicts[test],labels[test]))
                        thd.append(best_thresh)
                    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
        else:
            state = torch.load(dicts)
            state_dict = state['dict']
            model.load_state_dict(state_dict)
            model.eval()
            predicts = []
            labels=[]
            for batch_idx,(_,_2,data1,data2,data1f, data2f,target) in enumerate(eval_loader):
                data1, data2, data1f, data2f,target = data1.cuda(),data2.cuda(),data1f.cuda(),data2f.cuda(),target.cuda()
                ip1, _ , _ = model(data1)
                ip1f, _ , _ = model(data1f)
                ip2, _ , _ = model(data2)
                ip2f, _ , _ = model(data2f)
                # ip1=F.normalize(ip1)
                # ip2=F.normalize(ip2)
                # ip1f=F.normalize(ip1f)
                # ip2f=F.normalize(ip2f)
                ip1 = (ip1+ip1f)/2
                ip2 = (ip2+ip2f)/2
                # ip1= torch.cat((ip1,ip1f),1)
                # ip2= torch.cat((ip2,ip2f),1)
                cosdistance=ip1*ip2     
                cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
                label=target.data.cpu().numpy()
                cosdistance=cosdistance.data.cpu().numpy()
                predicts.append(cosdistance)
                labels.append(label)
                sys.stdout.write(str(batch_idx)+'/'+str(len(eval_loader))+'\r')
            predicts=np.array(predicts).reshape(6000,-1)
            labels=np.array(labels).reshape(6000,-1)
            accuracy = []
            thd = []
            folds = lfw.KFold(n=6000, n_folds=10)
            thresholds = np.arange(-1.0, 1.0, 0.005)
            for idx, (train, test) in enumerate(folds):
                best_thresh = lfw.find_best_threshold(thresholds, predicts[train],labels[train])
                accuracy.append(lfw.eval_acc(best_thresh, predicts[test],labels[test]))
                thd.append(best_thresh)
            print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    else:
        model.eval()
        predicts = []
        labels=[]
        for batch_idx,(_,_2,data1,data2,data1f, data2f,target) in enumerate(eval_loader):
            data1, data2, data1f, data2f,target = data1.cuda(),data2.cuda(),data1f.cuda(),data2f.cuda(),target.cuda()
            '''
            ip1,_, _ = model(data1)
            ip1f, _ , _ = model(data1f)
            ip2, _ , _ = model(data2)
            ip2f, _ , _ = model(data2f)
            '''
            _, ip1, _ = model(data1)
            _, ip1f, _ = model(data1f)
            _, ip2, _ = model(data2)
            _, ip2f, _ = model(data2f)
            
            ip1 = (ip1+ip1f)/2
            ip2 = (ip2+ip2f)/2
            # ip1= torch.cat((ip1,ip1f),1)
            # ip2= torch.cat((ip2,ip2f),1)
            cosdistance=ip1*ip2     
            cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
            label=target.data.cpu().numpy()
            cosdistance=cosdistance.data.cpu().numpy()
            predicts.append(cosdistance)
            labels.append(label)
            #sys.stdout.write(str(batch_idx)+'/'+str(len(eval_loader))+'\r')
            #sys.stdout.flush()
        predicts=np.array(predicts).reshape(6000,-1)
        labels=np.array(labels).reshape(6000,-1)
        accuracy = []
        thd = []
        folds = lfw.KFold(n=6000, n_folds=10)
        thresholds = np.arange(-1.0, 1.0, 0.005)
        for idx, (train, test) in enumerate(folds):
            best_thresh = lfw.find_best_threshold(thresholds, predicts[train],labels[train])
            accuracy.append(lfw.eval_acc(best_thresh, predicts[test],labels[test]))
            thd.append(best_thresh)
        #print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy)

        
def eval_agedb():
    read()

def compare_two_video(model,dir1='/home/jdq/database/database/ytf_crop/Uzi_Landau/2',dir2='/home/jdq/database/database/ytf_crop/Aaron_Guiel/5'):
    imgs1 = os.listdir(dir1)
    imgs2 = os.listdir(dir2)
    # m
    features = []
    for imgname in imgs1:
        if 'jpg.' in imgname:
            img=Image.open(imgname)
            data = transform_eval(img)
            data = data.cuda()
            feature, _ , _ = model(data)
            feature = feature.data.cpu.numpy()
            features.append(feature)
    features=np.array(features).mean(axis=0)
    feat1=sklearn.preprocessing.normalize(features).flatten()

    features = []
    for imgname in imgs2:
        if 'jpg.' in imgname:
            img=Image.open(imgname)
            data = transform_eval(img)
            data = data.cuda()
            feature, _ , _ = model(data)
            feature = feature.data.cpu.numpy()
            features.append(feature)
    features=np.array(features).mean(axis=0)
    feat2=sklearn.preprocessing.normalize(features).flatten()

    print(np.dot(feat1,feat2))

    
        

def load_result(filename):
    margins = []
    iters = []
    accs = []
    print(filename)
    with open(filename,'r') as f:

        while True:
            line=f.readline()
            if not line:      
                break
            if "Margin" in line:
                
                
                Margin = float(line.split("Margin=")[1].split(" ")[0])
                iter_times = int(line.split("/")[0].split(' ')[-1])
                #print(iter_times)
                valline=f.readline()
                acc=float(valline.split("LFWACC=")[1].split(" ")[0])
                # acc = float(line.split('Acc: ')[1].split('%')[0])
                
                margins.append(Margin)
                iters.append(iter_times)
                accs.append(acc)
            

        margins = np.array(margins)
        #print(margins)
        iters = np.array(iters)
        accs = np.array(accs)
        #print(a)

    return margins, iters, accs

def proxy(a, b):
    return EvalTool._cal_vertices_error(a,b)

class EvalTool(object):   
    
    bfm=np.load("../propressing/bfma.npz")
    num_index = bfm['index'].shape[0]
    index = bfm['index']

    micc_obj_root = '/data2/jdq/dbs/florence/'
    def __init__(self, batch_size = 20, criterion=None,BFM=None, tb_writer= None, eval_fr_data_root_path = config.eval_fr_data_root_path, transform =None, aflw_data_root_path = config.aflw_data_root_path):
        from data_loader import AFLW2000DataSet
        #from pytorch_3DMM import BFMA_batch
        import BFMG_batch,BFMN_batch
        self.bfm = BFM

        self.tfm = transform
        self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = self._get_fr_val_data(eval_fr_data_root_path)
        self.writer = tb_writer
        self.batch_size = batch_size
        self.criterion = criterion
        if criterion is not None:
            if config.use_face_recognition_constraint or config.use_mix_data:
                self.bfmf = criterion.module.loss_3d_func.bfmf
            else:
                self.bfmf = criterion.module.bfmf

        # if 'nonlinear' in config.mode:
        #     self.bfmf = BFMN_batch.BFMN_batch().to(config.device)
        # else:
        #     self.bfmf = BFMG_batch.BFMG_batch().to(config.device)
        # self.bfmf = torch.nn.DataParallel(self.bfmf,device_ids=config.gpu_ids)
        print("#####################################################")
        evalset_aflw = AFLW2000DataSet(root=aflw_data_root_path,transform=self.tfm)
        self.eval_loader_aflw = torch.utils.data.DataLoader(evalset_aflw, batch_size=batch_size, shuffle=False, num_workers=4)
        #self.landmark_map = torch.Tensor(np.load(config.f_model_landmark_map)).long()

        transform_eval_f=transforms.Compose([
    
            transforms.RandomHorizontalFlip(p=1),
            self.tfm
        ])

        print("#####################################################")
        evalset_ytf = FeatureDataset(root='/data/jdq/eval_dbs/ytf_crop/',filelist="files.txt",transform=self.tfm,target_transform=transform_eval_f,resample=20)
        # evalset = YTFDataSet(root="/home/jdq/database/database/",pairfile="../splits.txt",transform=transform_eval)
        self.eval_loader_ytf = torch.utils.data.DataLoader(evalset_ytf, batch_size=self.batch_size, shuffle=False, num_workers=4)
        # # print(len(eval_loader))s
       
        # self.micc_image_root = "/data/jdq/florence_mtcnn_crop"
        # self.micc_loader = 1
        # self.dict_file = ''
        # self.filelist = os.path.join(self.micc_image_root, 'files.txt')

        # evalset_micc = data_loader.MICCDataSet(root=self.micc_image_root, filelist=self.filelist, transform=self.tfm)
        # self.eval_loader_micc = torch.utils.data.DataLoader(evalset_micc, batch_size=self.batch_size, shuffle=False, num_workers=16)


    @staticmethod
    def read_mesh(file_name):
        with open(file_name, "r") as f:
            lines=f.readlines()
            vertices=[]
            faces=[]
            for line in lines:
                words = line.split(" ")
                if words[0]=="v":
                    ver=np.zeros(5)
                    ver[:]=float(words[1]),float(words[2]),float(words[3]),1,-1
                    vertices.append(ver)
                if words[0]=="f":
                    face=np.zeros(4,dtype=int)
                    face[:]=int(words[1].split("/")[0]),int(words[2].split("/")[0]),int(words[3].split("/")[0]),1
                    faces.append(face)
            vertices=np.array(vertices)
            faces=np.array(faces)
        return vertices,faces
    @staticmethod
    def crop_radius(mesh_vertices, mesh_faces):
  
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
            if (mesh_vertices[face[0:3]-1] == 0).any():
                mesh_faces[idx,3]=0
            else:
                mesh_faces[idx,0:3]=mesh_vertices[mesh_faces[idx,0:3]-1,4]
                target_faces.append(mesh_faces[idx,0:3])
        target_vertices=np.array(target_vertices)
        target_faces=np.array(target_faces)

        return target_vertices, target_faces

    @staticmethod
    def _cal_vertices_error(vertices, label):
        # import icp
        error =0
        num = 0
        
        num_vertex = vertices.shape[0]
        v2, f2 = EvalTool.crop_radius(vertices,EvalTool.index)
        micc_dir = os.path.join(EvalTool.micc_obj_root,str(label))  
        if os.path.exists(micc_dir):
            files=os.listdir(micc_dir)
            for file_n in files:
                file_n=os.path.join(micc_dir,file_n)
                v1, _ = EvalTool.read_mesh(file_n)
                error += EvalTool._icp(v1,v2)
                num+=1
            return error/num
        else:    
            print(micc_dir)
            return 4

                
        
            
    @staticmethod
    def _icp(v1, v2):
        
        
        def best_fit_transform(A, B):
            '''
            Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
            Input:
            A: Nxm numpy array of corresponding points
            B: Nxm numpy array of corresponding points
            Returns:
            T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
            R: mxm rotation matrix
            t: mx1 translation vector
            '''

            assert A.shape == B.shape

            # get number of dimensions
            m = A.shape[1]

            # translate points to their centroids
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - centroid_A
            BB = B - centroid_B

            # rotation matrix
            H = np.dot(AA.T, BB)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # special reflection case
            if np.linalg.det(R) < 0:
                Vt[m-1,:] *= -1
                R = np.dot(Vt.T, U.T)

            # translation
            t = centroid_B.T - np.dot(R,centroid_A.T)

            # homogeneous transformation
            T = np.identity(m+1)
            T[:m, :m] = R
            T[:m, m] = t

            return T, R, t
        
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


            from sklearn.neighbors import NearestNeighbors
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
        
        vertices1=vertices1[:,0:3]+origin2[0:3]-origin1[0:3]
        vertices2=vertices2[:,0:3]
        
        prev_error = 0

        for i in range(100):
            # find the nearest neighbors between the current source and destination points
            distances, indices = nearest_neighbor(vertices2, vertices1)

            # compute the transformation between the current source and nearest destination points
            T,R,t = best_fit_transform(vertices2, vertices1[indices])

            # update the current source
            vertices2 = (np.dot(R, vertices2.T)).T+t

            # check error
            mean_error = np.sqrt(np.mean(distances**2))
            if np.abs(prev_error - mean_error) < 0.00001:
                break
        
            prev_error = mean_error
            #print(mean_error)

        return mean_error

    def get_micc_rmse(self, model, save_tmp = False):

        from pytorch_3DMM import BFMA_batch
        import concurrent.futures
        model.eval()
        total_loss=0
        num = 0
        for batch_idx, (data, target, image_name) in enumerate(self.eval_loader_micc):
            data ,target=  data.cuda(),target.cuda()
            #_ , (_, vertex, _) = model(data)
            num = num + len(data)
            _, pred, pose_expr = model(data)  
            face = BFMA_batch.BFMA_batch(pred[:, :199], pose_expr[:, 7:36], pose_expr[:, 0:7])
            vertex = face.face_vertex_tensor
            vertex = vertex.reshape(len(vertex),-1,3)
            vertex[:, :, 1] = -vertex[:, :, 1]
            vertex[:, :, 2] = -vertex[:, :, 2]

            
            with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
                futures = [executor.submit(proxy, vert, label) for vert, label in zip(vertex.data.cpu().numpy().reshape(len(vertex),-1,3), target.data.cpu().numpy())]
                for future in concurrent.futures.as_completed(futures):
                    total_loss = total_loss + future.result()
        return total_loss / num
    def _get_fr_val_data(self, data_path):
        def _get_val_pair(path, name):
         
            carray = bcolz.carray(rootdir = os.path.join(path,name), mode='r')
            issame = np.load(os.path.join(path,'{}_list.npy').format(name))
        
            return carray, issame

            n = issame.shape[0] * 2
            data = torch.zeros((n, 3, 112, 96)).float()
            for i in range(0, n):
                img = Image.open(os.path.join(path, '{}_rgb/{}.jpg'.format(name,i)))
                
                if self.tfm is not None:
                    img = self.tfm(img)
    
                data[i] = img
            return data, issame


        agedb_30, agedb_30_issame = _get_val_pair(data_path, 'agedb_30')
        cfp_fp, cfp_fp_issame = _get_val_pair(data_path, 'cfp_fp')
        lfw, lfw_issame = _get_val_pair(data_path, 'lfw')
         
        return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame
    
    
    def update_tb(self, model, x_step, eval_ytf = False, emb_idx = 1, mask = 0):
        if self.writer is not None:
            '''
            cfp_acc, cfp_thd, cfp_roc = self.evaluate(model, self.lfw, self.lfw_issame)
            self.writer.add_scalar('eval_cfp-fp_acc', cfp_acc, x_step)
            self.writer.add_scalar('eval_cfp-fp_thd', cfp_thd, x_step)
            self.writer.add_image('eval_cfp_f', cfp_roc, x_step)
            
            '''

            embedding_size = [512, 199]

            if mask & 1 and eval_ytf:
                ytf_acc, ytf_thd, ytf_roc = self._eval_ytf(model, emb_idx = emb_idx)
                self.writer.add_scalar('eval_ytf_acc', ytf_acc, x_step)
                self.writer.add_scalar('eval_ytf_thd', ytf_thd, x_step)
                self.writer.add_image('eval_ytf_roc', ytf_roc, x_step)
                print('1111111111')
            if mask & 2:
                micc_rmse = self.get_micc_rmse(model)
                self.writer.add_scalar('eval_micc_rmse', micc_rmse, x_step)

            if mask & 4:
                aflw_ced, mean_rst = self._evaluate_aflw(model)
                self.writer.add_scalar('eval_fr/aflw2000', mean_rst, x_step)
                self.writer.add_image('eval_aflw2000_CED', aflw_ced, x_step)
            if mask & 8:
                lfw_acc, lfw_thd, lfw_roc = self._evaluate_fr(model, self.lfw, self.lfw_issame, name = 'LFW', embedding_size = embedding_size[emb_idx], emb_idx = emb_idx)
                agedb_acc, agedb_thd, agedb_roc = self._evaluate_fr(model, self.agedb_30, self.agedb_30_issame, name = 'Agedb_30', embedding_size  = embedding_size[emb_idx], emb_idx = emb_idx)
                cfp_acc, cfp_thd, cfp_roc = self._evaluate_fr(model, self.cfp_fp, self.cfp_fp_issame, name = 'CFP-FP',embedding_size  = embedding_size[emb_idx],  emb_idx = emb_idx)
                print('8888888888')
            
                self.writer.add_scalar('eval_fr/lfw_acc', lfw_acc, x_step)
                self.writer.add_scalar('eval_fr/agedb_30_acc', agedb_acc, x_step)
                self.writer.add_scalar('eval_fr/cfp_fp_acc', cfp_acc, x_step)
            
                self.writer.add_scalar('eval_fr/lfw_thd', lfw_thd, x_step)
                self.writer.add_scalar('eval_fr/agedb_30_thd', agedb_thd, x_step)
                self.writer.add_scalar('eval_fr/cfp_fp_thd', cfp_thd, x_step)

                self.writer.add_image('eval_fr/lfw_roc', lfw_roc, x_step)
                self.writer.add_image('eval_fr/cfp_fp_roc', cfp_roc, x_step) 
                self.writer.add_image('eval_fr/agedb_30_roc', agedb_roc, x_step)
            
                self.writer.add_scalar('heihei', 0, 0)
        else:
            print('TensorBoard Writer is not inited!')

    def get_fr_eval_acc(self, model, embedding_size = 99, emb_idx = 1):
        lfw_acc, lfw_thd, lfw_roc = self._evaluate_fr(model, self.lfw, self.lfw_issame, embedding_size = embedding_size, emb_idx = emb_idx)
        print(lfw_acc)
        agedb_acc, agedb_thd, agedb_roc = self._evaluate_fr(model, self.agedb_30, self.agedb_30_issame, embedding_size = embedding_size, emb_idx = emb_idx)
        print(agedb_acc)
        cfp_acc, cfp_thd, cfp_roc = self._evaluate_fr(model, self.cfp_fp, self.cfp_fp_issame, embedding_size = embedding_size, emb_idx = emb_idx)
        print(cfp_acc)
        ytf_acc, ytf_thd, ytf_roc = self._eval_ytf(model, emb_idx = emb_idx)
        print(ytf_acc)
        #return lfw_acc, agedb_acc, cfp_acc

    def _calROC(self, pred, y, save_path = None, name = ''):
        from sklearn import metrics
        import matplotlib.pyplot as plt
        import io
        plt.switch_backend('agg')

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic {}'.format(name))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if save_path is not None:
            plt.savefig(save_path)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)
        plt.close()
        return roc_curve_tensor

    def _KFold(self, n=5000, n_folds=10):
        folds = []
        base = list(range(n))
        for i in range(n_folds):
            test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
            train = list(set(base) - set(test))
            folds.append([train, test])
        return folds
    
    def _evaluate_fr2(self, model, carray, issame, nrof_folds = 5, name = None, tta = False, embedding_size = 199, device = 'cuda', emb_idx = 1):
        
        
        model.eval()
        idx = 0
        embeddings = torch.zeros([len(carray), embedding_size])
        from pytorch_3DMM import BFMA_batch
        for i in range(len(carray)):
            tmp = np.load('/data2/lmd2/3dmm_cnn-master/demoCode/out_agedb/%d.npy'%i)
            
            embeddings[i] =torch.tensor( tmp)# *torch.tensor(BFMA_batch.BFMA_batch.shape_ev[:99]).float()
            #print(embeddings[i])
        
        folds = self._KFold(n=len(embeddings), n_folds=10)
        from sklearn.decomposition import PCA 

        thresholds = np.arange(-1.0, 1.0, 0.005)
        labels=np.array(issame).reshape(len(issame),-1)
        accs = []
        for idx, (train, test) in enumerate(folds):
            
            pca = PCA(n_components=99)
            pca.fit_transform(embeddings[train])
            new_em = pca.transform(embeddings)
            new_em = np.sign(new_em) * np.sqrt(np.abs(new_em))

            ip1 = torch.tensor(new_em[0::2])[:,:embedding_size]
            ip2 = torch.tensor(new_em[1::2])[:,:embedding_size]

   
            cosdistance=ip1*ip2     
            cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
            
            predicts=np.array(cosdistance).reshape(len(issame),-1)
            test = np.array(test)
            train = np.array(train)
            best_thresh = self.find_best_threshold(thresholds, predicts[(train / 2)[0::2]], labels[(train/2)[0::2]])
            tpr, fpr, acc = self._calculate_accuracy(best_thresh, predicts[(test / 2)[0::2]], labels[(test/2)[0::2]])
            print(acc)
            accs.append(acc)
        print(np.mean(np.array(accs)))
        exit()
        

        cosdistance=ip1*ip2     
        cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
        print(cosdistance) 
        predicts=np.array(cosdistance).reshape(len(issame),-1)
        labels=np.array(issame).reshape(len(issame),-1)
        accuracy = []
        thd = []
        folds = self._KFold(n=len(issame), n_folds=10)
        thresholds = np.arange(-1.0, 1.0, 0.005)
       
        
        for idx, (train, test) in enumerate(folds):
            best_thresh = self.find_best_threshold(thresholds, predicts[train],labels[train])
            tpr, fpr, acc = self._calculate_accuracy(best_thresh, predicts[test], labels[test])
            accuracy.append(acc)
            thd.append(best_thresh)
            
        
        roc_curve_tensor = self._calROC(predicts, labels, name = name)

        return np.mean(accuracy), np.mean(thd), roc_curve_tensor


    def _evaluate_fr(self, model, carray, issame, nrof_folds = 5, name = None, tta = False, embedding_size = 99, device = 'cuda', emb_idx = 1):
        
        
        # _, pred, _, pose_expr = model(data)  
            
        # face_vertex_tensor, rotated, scaled = self.bfma(pred[:, :99], pose_expr[:, 7:36], pose_expr[:, 0:7])


        model.eval()
        idx = 0
        #print(embedding_size) 159645
        embeddings = torch.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + self.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + self.batch_size]).float()
                #print(batch.shape)
                pred = model(batch.to(device))[emb_idx]
                #face_vertex_tensor = self.bfm.module.get_shape(pred[:,:embedding_size])
                #face_vertex_tensor, rotated, scaled = self.bfma(pred[:, :99], pose_expr[:, 7:36], pose_expr[:, 0:7])
               
                embeddings[idx:idx + self.batch_size] = pred[:,:embedding_size].cpu()
                #embeddings[idx:idx + self.batch_size] = face_vertex_tensor.cpu()
                    
                idx += self.batch_size


            if idx < len(carray):
                batch = torch.tensor(carray[idx:]).float()  
                pred = model(batch.to(device))[emb_idx]
                # face_vertex_tensor = self.bfm.module.get_shape(pred[:,:embedding_size])
                # embeddings[idx:idx + self.batch_size] = face_vertex_tensor.cpu()
                #face_vertex_tensor, rotated, scaled = self.bfma(pred[:, :99], pose_expr[:, 7:36], pose_expr[:, 0:7])
                          
                embeddings[idx:] = pred[:,:embedding_size].cpu()
        ip1 = embeddings[0::2]
        ip2 = embeddings[1::2]

        cosdistance=ip1*ip2     
        cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
        Eudistence = (ip1-ip2)**2
        Eudistence = Eudistence.sum(dim=1).sqrt()
        max_number = torch.max(Eudistence)
        min_number = torch.min(Eudistence)
        Eudistence = Eudistence/(max_number-min_number)*2-1

        #predicts=np.array(Eudistence).reshape(len(issame),-1)
        predicts=np.array(Eudistence).reshape(len(issame),-1)
        labels=np.array(issame).reshape(len(issame),-1)
        accuracy = []
        thd = []
        folds = self._KFold(n=len(issame), n_folds=10)
        thresholds = np.arange(-1.0, 1.0, 0.005)
       
        
        for idx, (train, test) in enumerate(folds):
            best_thresh = self.find_best_threshold(thresholds, predicts[train],labels[train])
            tpr, fpr, acc = self._calculate_accuracy(best_thresh, predicts[test], labels[test])
            accuracy.append(acc)
            thd.append(best_thresh)
            
        
        roc_curve_tensor = self._calROC(predicts, labels, name = name)

        return np.mean(accuracy), np.mean(thd), roc_curve_tensor

    def _calculate_accuracy(self, threshold, dist, actual_issame):
        #predict_issame = np.less(dist, threshold)
        predict_issame = dist < threshold
        
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
       
        return tpr, fpr, acc
    


    def find_best_threshold(self, thresholds, predicts, labels):
        best_threshold =0
        best_acc = 0
        for threshold in thresholds:
            accuracy = self._calculate_accuracy(threshold, predicts,labels)[2]
            if accuracy >= best_acc:
                best_acc = accuracy
                best_threshold = threshold
        return best_threshold


    def _evaluate_aflw(self, model, save_path = None, device = 'cuda'):
        
        
        
        print("Evaluating aflw2000...")
        total = 0
        count = np.zeros(1000)
        sum_rst = 0
        #bfma = BFMA_batch.BFMA_batch()
        for batch_idx,(data, target) in enumerate(self.eval_loader_aflw):
            data, target =  data.to(device), target.to(device)
            #print(target.shape)
            
            
            zero_shape = torch.zeros(199 * data.shape[0]).reshape(-1, 199).to(device)
            _, pred, _, pose_expr = model(data)  
            
            scaled,scaled_res = self.bfmf(pred, pose_expr[:, 7:36], pose_expr[:, 0:7])
            #print(self.bfmf.module.get_landmark_68(scaled).shape)
            pred_lms = self.bfmf.get_landmark_68(scaled_res).transpose(2,1).reshape(target.shape[0], -1)
            x_max, x_min, y_max, y_min = target[:, ::2].max(1, True)[0], target[:, ::2].min(1, True)[0], target[:, 1::2].max(1, True)[0], target[:, 1::2].min(1, True)[0]
            d_normalize = torch.sqrt((x_max - x_min) * (y_max - y_min))

           # target = target.reshape(target.shape[0],-1,2)[:,self.landmark_map!=-1,:].reshape(target.shape[0], -1)
            target = target.reshape(target.shape[0],-1,2).reshape(target.shape[0], -1)
            pts = (pred_lms - target.float())
            pts = (pts**2).float().reshape(target.shape[0],-1, 2).sum(2, True)[:, :, 0].sqrt().mean(1, True)
            #print(pts)
            rst = pts / d_normalize.float()
            sum_rst = torch.sum(rst)
            total += rst.shape[0]
            for i in range(1000):
                count[i] = count[i] + torch.sum(rst < i * 1.0 / 1000)
            continue
    
        count = count * 1.0 / total
        mean_rst = sum_rst / total
        #import numpy as np
    
        prn = np.load(config.prn_rst)
        _3ddfa = np.load(config.ddfa_rst)
    
        import matplotlib
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        x_range = 1000
        
        x = np.linspace(0, x_range / 1000., x_range)
        y = count * 100
        
        y_prn = prn['arr_0'] * 100
        y_3ddfa = _3ddfa['arr_0'] * 100
        
        plt.figure()
        plt.grid()
        plt.xlim(0,0.1)
        plt.ylim(0,100)
        plt.plot(x,y[:x_range], color='red', label='ours')
        plt.plot(x,y_prn[:x_range], color='green', label='prn')
        plt.plot(x,y_3ddfa[:x_range], color='yellow', label='3ddfa')
        plt.legend(loc= 'lower right' )
        plt.xlabel("NME normalized by bounding box size")
        plt.ylabel("Number of images (%)")
        plt.title("Alignment Accuracy on AFLW2000 Dataset(68 points)")
    

        if save_path is not None:
            plt.savefig(save_path)
        #plt.savefig('imgs/testced.jpg')
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)
        
        plt.close()
        return img_tensor,mean_rst

    def _eval_ytf(self, model, root = '/data/jdq/ytf_crop/' ,splitfile='splits_corrected.txt', fdict = None, emb_idx = 1):
        def gen_feature(model,dataloader, to_file = True):
            model.eval()
            feat_dict = {}
            feat_count_dict = {}
            # f = open(indentity_name)
            # idxs = f.readlines()
            for batch_idx, (imgnames,data,dataf) in enumerate(dataloader):
                data , dataf = data.cuda(), dataf.cuda()
                feats = model(data)[emb_idx]
                featsf= model(dataf)[emb_idx]
                feats = (feats+featsf)/2
                feats= feats.data.cpu().numpy()
                for idx,imgname in enumerate(imgnames):
                    # print(imgname)
                    # exit()
                    imgpath = imgname.split('/')[-3]+'/'+imgname.split('/')[-2]
                    #imgpath = imgname['']
                    #imgpath = imgpath.replace(".jpg",".bin")
                    #imgpath = imgpath.replace("ytf_crop","ytf_crop_feature")
                    if feat_dict.__contains__(imgpath):
                        feat_count_dict[imgpath] += 1
                        feat_dict[imgpath] += feats[idx]
                    else:
                        feat_count_dict[imgpath] = 1 
                        feat_dict[imgpath] = feats[idx]
                    #feat_dict[imgpath] = np.ones(512)
                    
                sys.stdout.write(str(batch_idx)+'/'+str(len(dataloader))+'\r')
            

            for key in feat_dict.keys():
                #print(key)
                #os.mk
                feat_dict[key] = feat_dict[key] / feat_count_dict[key]
                

            if to_file:
                import pickle  
                f = open("ytf_result_n1.pkl","wb")
                pickle.dump(feat_dict,f)
                f.close()

            return feat_dict
        import pickle 
        

        predicts = []
        labels=[]
        if fdict is not None:
            state = torch.load(fdict)
            state_dict = state['dict']
            model.load_state_dict(state_dict)
        print("Start evaling YTF...")
        feat_dict = gen_feature(model, self.eval_loader_ytf, True)
        # with open('ytf_result_n1.pkl', 'rb') as handle:
        #     feat_dict = pickle.load(handle)
        # if not os.path.exists(root):
        #     raise ValueError("Cannot find the data!")
        with open(splitfile) as f:
            split_lines = f.readlines()[1:]

        ip1 = []
        ip2 = []
        for batch_idx,line in enumerate(split_lines):
            words = line.replace('\n', '').replace(' ', '').split(',')
            name1 =words[2]
            name2 =words[3]
            sameflag = int(words[5])
            #print(name1)
    
            # Calculate feature one 
            # featnames=os.listdir(name1)
            # feats1=[]
            #print(featnames)
            # for featname in featnames:
            #     if os.path.splitext(featname)[-1]=='.bin':
            #         #print(featname)
            #         featname = os.path.join(name1,featname)
            #         if feat_dict.has_key(featname):
            #             feat =  feat_dict[featname]
            #             #feat=np.fromfile(featname,dtype='float32')
            #             #feat=feat/np.linalg.norm(feat)
            #             feats1.append(feat)
            #         else:
            #             continue
            # feats1=np.array(feats1)
            # mean_feat1=feats1.mean(axis=0)
            ip1.append(feat_dict[name1])
            #mean_feat1=sklearn.preprocessing.normalize(mean_feat1).flatten()
            #print(mean_feat1.shape)
            #mean_feat1 = mean_feat1/np.linalg.norm(mean_feat1)
            #mean_feat1=feats[0]
            # Calculate feature two 
            # featnames=os.listdir(name2)
            # #print(featnames)
            # feats2=[]
            # for featname in featnames:
            #     if os.path.splitext(featname)[-1]=='.bin':
            #         featname = os.path.join(name2,featname)
            #         if feat_dict.has_key(featname):
            #             feat =  feat_dict[featname]
            #             #feat=np.fromfile(featname,dtype='float32')
            #             #feat=feat/np.linalg.norm(feat)
            #             feats2.append(feat)
            #         else:
            #             continue
            # feats2=np.array(feats2)
            # mean_feat2=feats2.mean(axis=0)
            ip2.append(feat_dict[name2])

            #mean_feat2=sklearn.preprocessing.normalize(mean_feat2).flatten()
            #print(mean_feat2.shape)
            #mean_feat2 = mean_feat2/np.linalg.norm(mean_feat2)
            #mean_feat2=feats[0]
    
            # Calculate the cosine distance
            #print((mean_feat1*mean_feat2).sum())

            # cosdistance=np.dot(mean_feat1,mean_feat2)/np.linalg.norm(mean_feat1)/np.linalg.norm(mean_feat2)
            # predicts.append(Eudistence)
            labels.append(int(sameflag))
            #print(cosdistance,sameflag)
            sys.stdout.write(str(batch_idx)+'/'+str(len(split_lines))+'\r')
           
        ip1 = torch.Tensor(ip1)[:,:199]
        ip2 = torch.Tensor(ip2)[:,:199]
        print(ip1.shape)
        Eudistence = (ip1-ip2)**2
        Eudistence = Eudistence.sum(dim=1).sqrt()
        max_number = torch.max(Eudistence)
        min_number = torch.min(Eudistence)
        Eudistence = Eudistence/(max_number-min_number)*2-1
        cosdistance=ip1*ip2
        cosdistance=cosdistance.sum(dim=1)/(ip1.norm(dim=1).reshape(-1)*ip2.norm(dim=1).reshape(-1)+1e-12)
 
        
        predicts=np.array(Eudistence).flatten()
        #print(predicts.shape)
        labels = np.array(labels).flatten()     
    
        accuracy = []
        thd = []
        folds = self._KFold(n=5000, n_folds=10)
        thresholds = np.arange(-1.0, 1.0, 0.005)
        for idx, (train, test) in enumerate(folds):
            #print(predicts.shape, labels.shape)
            best_thresh = self.find_best_threshold(thresholds, predicts[train],labels[train])  
            tpr, fpr, acc = self._calculate_accuracy(best_thresh, predicts[test], labels[test])
            accuracy.append(acc)
            thd.append(best_thresh)

        roc_curve_tensor = self._calROC(predicts, labels, name = 'ytf')
        return np.mean(accuracy), np.mean(thd), roc_curve_tensor
        print('YTFACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    @staticmethod
    def inference(model, img_path):
        from pytorch_3DMM import BFMA
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

        device = 'cuda' 
        img = tfm(Image.open(img_path)).unsqueeze(0).to(device)
        model.eval() 
        _, shape, pose_expr= model(img)
        shape_para = shape[0, 0:199].unsqueeze(1)
        exp_para = pose_expr[0, 7:36].unsqueeze(1) * 0
        camera_para = pose_expr[0, 0:7]
        face = BFMA.BFMA(shape_para,  exp_para , camera_para)
            
        face.mesh2off("../off/micc%d.off" % int(img_path[-5]), use_camera = False)




def get_fr_val_data_to_npy(data_path):
    def _get_val_pair(path, name):
    
        carray = bcolz.carray(rootdir = os.path.join(path,name), mode='r')
        issame = np.load(os.path.join(path,'{}_list.npy').format(name))
    
        return carray, issame

        n = issame.shape[0] * 2
        data = torch.zeros((n, 3, 112, 96)).float()
        for i in range(0, n):
            img = Image.open(os.path.join(path, '{}_rgb/{}.jpg'.format(name,i)))
            
            # if self.tfm is not None:
            #     img = self.tfm(img)

            data[i] = img
        return data, issame


    agedb_30, agedb_30_issame = _get_val_pair(data_path, 'agedb_30')
    print(agedb_30.shape, agedb_30_issame.shape)
    print(agedb_30, agedb_30_issame)
    cfp_fp, cfp_fp_issame = _get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = _get_val_pair(data_path, 'lfw')
    
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

if __name__ == "__main__":


    # get_fr_val_data_to_npy('/data/jdq/eval_dbs/')
    # exit()

    tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    from tensorboardX import SummaryWriter
    import time

    #writer = SummaryWriter('../tb_log/train_all' + str(int(time.time()))[-5:] + '/')
    writer = None
    print('Initialize EvalTool')

    import BFMP_batch
    bfm = BFMP_batch.BFMG_batch().to(config.device)
    bfm = torch.nn.DataParallel(bfm,device_ids=config.gpu_ids)
    a = EvalTool(transform = tfm, tb_writer = writer, BFM=bfm)
    import net
    print('Loading model')
    model = net.sphere64a(pretrained=False)
    
    #  = '../model/para_jdq/n1.pkl'
    gpu_ids = range(1)
    
    print('Evaluating')
    model.load_state_dict(torch.load(config.evalation_path))
    model = model.cuda()
    model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    #eval_ytf(model = model, root = '/data2/jdq/dbs/ytf_crop/')
    #a.get_micc_rmse(model)
    a.get_fr_eval_acc(model, embedding_size=199, emb_idx = 1)
    
    #a._evaluate_aflw(model, save_path = '../imgs/stage2.jpg')
    #a.update_tb(model,0)
    #writer.close()
    #EvalTool.inference(model, '../imgs/micc1.jpg')
    #EvalTool.inference(model, '../imgs/micc2.jpg')
    #EvalTool.inference(model, '../imgs/micc3.jpg')

