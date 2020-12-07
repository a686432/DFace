import torch
from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    Dropout2d,
    Dropout,
    AvgPool2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
    Parameter,
)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd.function import Function
import math

#################### implementation of ArcLoss in https://arxiv.org/abs/1801.05599 ################


class Arcface(Module):
    def __init__(self, embedding_size=512, classnum=8631, s=64.0, m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = cos_theta - self.mm  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        label = label.view(-1, 1)  # size=(B,1)
        output = (
            cos_theta * 1.0
        )  # a little bit hacky way to prevent in_place operation on cos_theta
        output[torch.range(0, nB - 1).to(torch.long), label] = cos_theta_m[
            torch.range(0, nB - 1).to(torch.long), label
        ]
        output *= (
            self.s
        )  # scale up in order to make softmax work, first introduced in normface
        return output


################################ implementation of CosLoss #######################################

# CosLinear is a FC Layer: transform 512D feature into 8631D score.
class CosLinear(nn.Module):
    def __init__(self, in_features=512, out_features=8631):
        super(CosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight=nn.Parameter(torch.randn(out_features,in_features))

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input):
        # global Scale
        # xlen = input.pow(2).sum(1).pow(0.5).mean()
        # #wlen = self.weight.pow(2).sum(1).pow(0.5)
        # #xlen = xlen.view(-1,1).mm(wlen.view(1,-1)).clamp(max=72).exp().mean()
        # xlen = float(xlen.data.cpu().numpy())
        # if Scale == 0:
        #     Scale = xlen
        # else:
        #      Scale = xlen
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        # output2 = F.linear(input, F.normalize(self.weight))
        output = cos_theta.clamp(-1, 1)

        return output


def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)


class CosLoss(nn.Module):
    def __init__(self, num_cls=512, s=32, alpha=0.2):
        super(CosLoss, self).__init__()
        self.num_cls = num_cls
        self.alpha = alpha
        self.scale = s
        self.m = 0
        # self.phi=nn.Parameter(torch.Tensor(1))
        # self.phi.data.uniform_(s, -s)

    def forward(self, score, y):

        # xlen = input.pow(2).sum(1).pow(0.5).mean()
        # xlen = float(xlen.data.cpu().numpy())
        # sys.stdout.write('Scale={:.4f} | '.format(xlen))
        # cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        # output = cos_theta.clamp(-1, 1)

        # global Scale
        y = y.view(-1, 1)
        batch_size = score.size()[0]
        # feat = feat + self.phi.expand_as(feat)
        margin_xw_norm = score - self.alpha
        y_onehot = torch.Tensor(batch_size, self.num_cls).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, y.data.view(-1, 1), 1)
        y_onehot.byte()
        y_onehot = Variable(y_onehot)
        """
        margin = myeval.margin_observe(feat,y,self.num_cls)
        l=0.99*self.m+0.01*margin.mean().detach()
        self.m=float(l.data.cpu().numpy())
        #sys.stdout.write('Margin={:.4f} | '.format(self.m))
        """

        value = self.scale * where(y_onehot, margin_xw_norm, score)
        # value = value
        # logpt = F.log_softmax(value)
        y = y.view(-1)
        loss = nn.CrossEntropyLoss(size_average=False, reduce=False)
        output = loss(value, y)

        # output = 32.0/self.scale * output
        # loss = loss.mean()
        return output


################################ implementation of SoftmaxLoss #######################################
class softmaxLoss(nn.Module):
    def __init__(self, num_cls):
        super(softmaxLoss, self).__init__()

    def forward(self, feat, y):

        loss = nn.CrossEntropyLoss()
        output = loss(feat, y)

        return output


class softmaxLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(softmaxLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight=nn.Parameter(torch.randn(out_features,in_features))
        # self.weight = Parameter(torch.Tensor(out_features,in_features))
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input):
        output = F.linear(input, self.weight)

        return output


################################ implementation of AngleLoss #######################################
class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=2, phiflag=False):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        # cos_theta = cos_theta / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            cos_m_theta = (self.m * theta).cos()
            # phi_theta = myphi(theta,self.m)
            # phi_theta = phi_theta.clamp(-1*self.m,1)
            # cos_m_theta = self.mlambda[self.m](cos_theta)
            # theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / math.pi).floor()
            n_one = -1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta, xlen)
        return output  # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0, num_cls=10):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.num_cls = num_cls
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.m = 0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta, xlen = input

        feat = cos_theta / xlen.view(-1, 1)

        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.001 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


################################ implementation of RingLoss #######################################


class RingLoss(nn.Module):
    def __init__(self, type="auto", loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if (
            self.radius.data[0] < 0
        ):  # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == "l1":  # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == "auto":  # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else:  # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


################################ implementation of CenterLoss #######################################


class CenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1):
        super(CenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c

        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))

    def forward(self, y, hidden, ind=None, weights=None):
        batch_size = hidden.size(0)
        expanded_centers = self.centers.index_select(dim=0, index=y)
        # centers = self.centers
        if ind is None:
            c_norm = expanded_centers.norm(dim=1).reshape(batch_size, 1)
            c_norm = c_norm.clamp(min=1)
            f_norm = hidden.norm(dim=1).reshape(batch_size, 1).clamp(min=1)
            intra_distances = ((hidden - expanded_centers) ** 2 / f_norm / c_norm).sum()
            loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
            return loss

        else:
            # c_norm = expanded_centers.norm(dim=1).reshape(batch_size,1)
            # c_norm = c_norm.clamp(min=1)
            # f_norm =  hidden.norm(dim=1).mean().clamp(min=1).detach()
            # c_norm = expanded_centers.norm(dim=1).mean().clamp(min=1).detach()
            intra_distances = torch.sum(
                (hidden - expanded_centers) ** 2, dim=1
            ).reshape(batch_size, 1) * ind.reshape(batch_size, 1)
            loss = (self.lambda_c / 2.0) * intra_distances
            weighted_center = None
            if not weights is None:
                weighted_center = loss * weights.reshape(batch_size, 1).detach()
                # weighted_center = (self.lambda_c / 2.0 ) * torch.sum((hidden.detach() - expanded_centers)**2,dim=1).reshape(batch_size,1)*ind.reshape(batch_size,1)
                # weighted_center = weighted_center * weights.reshape(batch_size,1)
            # return loss, expanded_centers, weighted_center
            # loss = (self.lambda_c / 2.0 ) * intra_distances
        # loss = loss if loss < 1 else loss*0
        # print(loss)
        return loss, expanded_centers, weighted_center


if __name__ == "__main__":
    a = CosLinear()
