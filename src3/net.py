
import torch
import torch.nn as nn
from collections import OrderedDict
import math
import misc
import torch.nn.functional as F
import numpy as np
import config
import torchvision
EPS = 1e-7

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class TexDecoderFC(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(TexDecoderFC, self).__init__()
        self.fc1 = nn.Linear(in_feat, 1024, bias = False)
        self.prelu1 = nn.PReLU(1024)
        self.fc2 = nn.Linear(1024, 1024, bias = False)

        self.prelu2 = nn.PReLU(1024)

        self.fc3 = nn.Linear(1024, out_feat, bias = False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.prelu1(x)
        x = self.fc2(x)

        x = self.prelu2(x)
        x = self.fc3(x)
        return x

class TexDecoderConv(nn.Module):
    def __init__(self, in_feat):
        super(TexDecoderConv, self).__init__()
        self.fc = nn.Linear(in_feat, 8*8*320)
        network = [
        # nn.Linear(in_feat, 8*8*320),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=320, out_channels=256, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True),
        #nn.GroupNorm(16, 160),
        nn.ReLU(inplace=True),


        torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        nn.GroupNorm(8, 128),
        nn.ReLU(inplace=True),


        torch.nn.ConvTranspose2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True),
        #nn.GroupNorm(12, 192),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        nn.GroupNorm(6, 96),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True),
        #nn.GroupNorm(8, 128),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        nn.GroupNorm(4, 64),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True),
        #nn.GroupNorm(4, 64),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        nn.GroupNorm(4, 64),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)]
        self.network = nn.Sequential(*network)
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 320, 8, 8) 
        x = self.network(x)
        # x = self.fconv51(x)
        # x = self.fconv43(x)
        # x = self.fconv42(x)
        # x = self.fconv41(x)
        # x = self.fconv33(x)
        # x = self.fconv32(x)
        # x = self.fconv31(x)
        # x = self.fconv23(x)
        # x = self.fconv22(x)
        # x = self.fconv21(x)
        # x = self.fconv13(x)
        # x = self.fconv12(x)
        # x_s = self.fconv11_s(x)
        # x = self.fconv11(x)
        
        return x[:,:3], x[:,3:]


# class TexDecoderConv(nn.Module):
#     def __init__(self, in_feat):
#         super(TexDecoderConv, self).__init__()

#         self.fc = nn.Linear(in_feat, 8*8*320)
#         self.fconv52 = torch.nn.ConvTranspose2d(in_channels=320, out_channels=160, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#         self.fconv51 = torch.nn.ConvTranspose2d(in_channels=160, out_channels=256, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)

#         self.fconv43 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#         self.fconv42 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#         self.fconv41 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)

#         self.fconv33 = torch.nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#         self.fconv32 = torch.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#         self.fconv31 = torch.nn.ConvTranspose2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)

#         self.fconv23 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#         self.fconv22 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#         self.fconv21 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)

#         self.fconv13 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
#         self.fconv12 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#         self.fconv11 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#         self.fconv11_s = torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True)
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.size(0), 320, 8, 8) 
#         x = self.fconv52(x)
#         x = self.fconv51(x)
#         x = self.fconv43(x)
#         x = self.fconv42(x)
#         x = self.fconv41(x)
#         x = self.fconv33(x)
#         x = self.fconv32(x)
#         x = self.fconv31(x)
#         x = self.fconv23(x)
#         x = self.fconv22(x)
#         x = self.fconv21(x)
#         x = self.fconv13(x)
#         x = self.fconv12(x)
#         x_s = self.fconv11_s(x)
#         x = self.fconv11(x)
        
#         return x, x_s





class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, planes, stride=1):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(planes, planes)
        m['relu1'] = nn.PReLU(planes)
        m['conv2'] = conv3x3(planes, planes)
        m['relu2'] = nn.PReLU(planes)
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        residual = x
        out = self.group1(x) + residual
        return out

class BasicMLP(nn.Module):
    expansion = 1
    def __init__(self, planes, stride=1):
        super(BasicMLP, self).__init__()
        m = OrderedDict()
        m['fc1'] = nn.Linear(planes, planes)
        m['relu1'] = nn.PReLU(planes)
        m['fc2'] = nn.Linear(planes, planes)
        m['relu2'] = nn.PReLU(planes)
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        residual = x
        out = self.group1(x) + residual
        return out


class ShapeNet(nn.Module):
    def __init__(self,planes,expansion=4):
        super(ShapeNet,self).__init__()
        m = OrderedDict()
        m['fc1'] = nn.Linear(planes, planes*expansion)
        m['relu1'] = nn.PReLU(planes*expansion)
        m['fc2'] = nn.Linear(planes*expansion, planes*expansion)
        m['relu2'] = nn.PReLU(planes*expansion)
        m['fc3'] = nn.Linear(planes*expansion, 53215*3)
        # m['relu2'] = nn.PReLU(planes)
        self.planes = planes
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        out = self.group1(x)
        return out
        

    
class SphereNet(nn.Module):
    def __init__(self, block, layers, stage = 0):
        super(SphereNet, self).__init__()

        # face recognition
        self.dim = 199
        self.layer1 = self._make_layer(block, 3, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.shape_basis = ShapeNet(planes=199)
        self.fc5 = nn.Linear(512*7*6,2*199)

        if config.use_confidence_map:
            self.conf_Net = ConfNet(cin=3, cout=2, nf=64, zdim=128)


        #self.A_BFMS = nn.Linear(99,140970)
        #self.BFME = nn.Linear(29,159645)
        #bfm=np.load("../propressing/bfma.npz")
        # lqy self.shape_ev=torch.Tensor(bfm['shape_ev'].reshape(-1)).to(config.device)
        
        
        # if stage == 2: 
        #     for p in self.parameters():
        #         p.requires_grad=False

        # # face construction
        # self.fc1 = nn.Linear(512,512)
        # self.re=nn.PReLU(512)
        # self.fc2 = nn.Linear(512,512)
        # self.re2=nn.PReLU(512)
        #self.layer5 = self._make_layer2(BasicMLP,1024,1)
        # self.fc3 = nn.Linear(512,199*2)


        # if stage == 3:
        #     for p in self.parameters():
        #         p.requires_grad=False
        #     for p in self.layer1.parameters():
        #         p.requires_grad=True
        #     for p in self.layer2.parameters():
        #         p.requires_grad=True

        # self.layer2_up = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3_up = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4_up = self._make_layer(block, 256, 512, layers[3], stride=2)

        self.layer3_tex = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4_tex = self._make_layer(block, 256, 512, layers[3], stride=2)
       
       
        self.fc5_tex = nn.Linear(512*7*6,512)
        #  pose expression
        self.fc1_up = nn.Linear(512*7*6,512)
        self.re_up=nn.PReLU(512)
        # self.fc2_up = nn.Linear(1024,1024)
        # self.re2_up=nn.ReLU()
        # #self.layer5 = self._make_layer2(BasicMLP,1024,1)
        # self.fc3_up = nn.Linear(1024,36)

        self.fc2_up = nn.Linear(512,63)  # exprssion 29 + camera 7 + illum 27
        # self.scale =nn.Parameter(torch.Tensor([1.0]))
        if config.use_ConvTex: 
            self.texdecoder = TexDecoderConv(in_feat=512)
        else:
            self.texdecoder = TexDecoderFC(in_feat=199,out_feat=53215*3)

        '''
        # test loop back
        
        for p in self.parameters():
            p.requires_grad=False
        
        
        self.fc1_back = nn.Linear(99,1024)
        self.re_back=nn.PReLU(1024)
        self.fc2_back = nn.Linear(1024,1024)
        self.re2_back =nn.PReLU(1024)
        #self.layer5 = self._make_layer2(BasicMLP,1024,1)
        self.fc3_back = nn.Linear(1024,512)
        '''
        

    def _make_layer(self, block, inplanes, planes, blocks, stride=2):

        layers = []
        layers.append(nn.Sequential( 
                conv3x3(inplanes, planes, stride=stride),
                nn.PReLU(planes))
                )

        for _ in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def _make_layer2(self, block,  planes, blocks):

        layers = []
        for _ in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.group1(x)i
        conf, conf_lm = None, None
        if config.use_confidence_map:
            conf,conf_lm = self.conf_Net(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_mid = self.layer3(x)
        x_mid = self.layer4(x_mid)
        x_mid = x_mid.view(x_mid.size(0),-1)
        

        shape = self.fc5(x_mid)

        #print(feat.shape)
        
        # mid = self.re(self.fc1(feat))
        # mid = self.re2(self.fc2(mid))
        # mid = self.fc3(mid)
         
        #abedlo = self.texdecoder(texture)

        x_up = self.layer3_up(x)
        x_up = self.layer4_up(x_up)
        x_up = x_up.view(x_up.size(0),-1)
        
        
        up = self.re_up(self.fc1_up(x_up))
        #up = self.re2_up(self.fc2_up(up))
        #up = self.fc3_up(up)
        
        up = self.fc2_up(up)

        x_tex = self.layer3_tex(x)
        x_tex = self.layer4_tex(x_tex)
        x_tex = x_tex.view(x_tex.size(0),-1)
        texture  = self.fc5_tex(x_tex)
        abedlo = self.texdecoder(texture)
        
        #test loop back
        '''
        feat_back = self.re_back(self.fc1_back(mid))
        feat_back = self.re2_back(self.fc2_back(feat_back))
        feat_back = self.fc3_back(feat_back)
        '''
        return  (conf,conf_lm), shape, abedlo, up  #, feat_back

def sphere64a(pretrained=False, model_root=None, stage = 0):
    model=SphereNet(BasicBlock, [3,8,16,3], stage)
    if pretrained:
        misc.load_state_dict(model, model_root)
    return model


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        #out = x/2 + 0.5
        out = (x - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def forward(self, im1, im2, mask=None, conf_sigma=None):
        #print(im1.shape,im2.shape)
        im = torch.cat([im1[...,:3].permute(0,3,1,2),im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)

class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        self.nf= nf
        ## downsampling
        network_e = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32 112*96 -> 56*48
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16 56*48 -> 28*24
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8 28*24 -> 14*12
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4 14*12 -> 7*6
            nn.LeakyReLU(0.2, inplace=True)]
            
            #nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            #nn.ReLU(inplace=True)
        self.fc_e = nn.Linear(nf*8*7*6,zdim,bias=False)
        self.relu = nn.ReLU()
        self.network_e = nn.Sequential(*network_e)
        self.fc_d = nn.Linear(zdim,nf*8*7*6,bias=False)
        ## upsampling
        network_d = [
            #nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network_d = nn.Sequential(*network_d)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        x = self.network_e(input)
        x = x.view(x.size(0),-1)
        x = self.relu(self.fc_e(x))
        x = self.fc_d(x).view(x.size(0), self.nf*8, 7, 6)
        out = self.network_d(x)
        return self.out_net1(out), self.out_net2(out)