import torch
import torch.nn as nn
from collections import OrderedDict
import math
import misc
import torch.nn.functional as F
import numpy as np
import config


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m["conv1"] = conv3x3(planes, planes)
        m["relu1"] = nn.PReLU(planes)
        m["conv2"] = conv3x3(planes, planes)
        m["relu2"] = nn.PReLU(planes)
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
        m["fc1"] = nn.Linear(planes, planes)
        m["relu1"] = nn.PReLU(planes)
        m["fc2"] = nn.Linear(planes, planes)
        m["relu2"] = nn.PReLU(planes)
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        residual = x
        out = self.group1(x) + residual
        return out


class SphereNet(nn.Module):
    def __init__(self, block, layers, stage=0):
        super(SphereNet, self).__init__()

        # face recognition
        self.layer1 = self._make_layer(block, 3, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.fc5 = nn.Linear(512 * 7 * 6, 512)
        # self.A_BFMS = nn.Linear(99,140970)
        # self.BFME = nn.Linear(29,159645)
        # bfm=np.load("../propressing/bfma.npz")
        # lqy self.shape_ev=torch.Tensor(bfm['shape_ev'].reshape(-1)).to(config.device)

        # if stage == 2:
        #     for p in self.parameters():
        #         p.requires_grad=False

        # face construction
        self.fc1 = nn.Linear(512, 512)
        self.re = nn.PReLU(512)
        self.fc2 = nn.Linear(512, 512)
        self.re2 = nn.PReLU(512)
        # self.layer5 = self._make_layer2(BasicMLP,1024,1)
        self.fc3 = nn.Linear(512, 99)

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
        #  pose expression
        self.fc1_up = nn.Linear(512 * 7 * 6, 512)
        self.re_up = nn.PReLU(512)
        # self.fc2_up = nn.Linear(1024,1024)
        # self.re2_up=nn.ReLU()
        # #self.layer5 = self._make_layer2(BasicMLP,1024,1)
        # self.fc3_up = nn.Linear(1024,36)

        self.fc2_up = nn.Linear(512, 36)
        self.scale = nn.Parameter(torch.Tensor([1.0]))

        """
        # test loop back
        
        for p in self.parameters():
            p.requires_grad=False
        
        
        self.fc1_back = nn.Linear(99,1024)
        self.re_back=nn.PReLU(1024)
        self.fc2_back = nn.Linear(1024,1024)
        self.re2_back =nn.PReLU(1024)
        #self.layer5 = self._make_layer2(BasicMLP,1024,1)
        self.fc3_back = nn.Linear(1024,512)
        """

    def _make_layer(self, block, inplanes, planes, blocks, stride=2):

        layers = []
        layers.append(
            nn.Sequential(conv3x3(inplanes, planes, stride=stride), nn.PReLU(planes))
        )

        for _ in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks):

        layers = []
        for _ in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.group1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_mid = self.layer3(x)
        x_mid = self.layer4(x_mid)
        x_mid = x_mid.view(x_mid.size(0), -1)

        feat = self.fc5(x_mid)
        # print(feat.shape)

        mid = self.re(self.fc1(feat))
        mid = self.re2(self.fc2(mid))
        mid = self.fc3(mid)
        shape = mid * self.scale

        x_up = self.layer3_up(x)
        x_up = self.layer4_up(x_up)
        x_up = x_up.view(x_up.size(0), -1)

        up = self.re_up(self.fc1_up(x_up))
        # up = self.re2_up(self.fc2_up(up))
        # up = self.fc3_up(up)

        up = self.fc2_up(up)

        # test loop back
        """
        feat_back = self.re_back(self.fc1_back(mid))
        feat_back = self.re2_back(self.fc2_back(feat_back))
        feat_back = self.fc3_back(feat_back)
        """
        return feat, shape, mid, up  # , feat_back


def sphere64a(pretrained=False, model_root=None, stage=0):
    model = SphereNet(BasicBlock, [3, 8, 16, 3], stage)
    if pretrained:
        misc.load_state_dict(model, model_root)
    return model
