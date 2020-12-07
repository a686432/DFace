import torch
import torch.nn as nn
import misc
import numpy as np
import scipy.io as sio


class BFMNet(nn.Module):
    def __init__(self):
        super(BFMNet, self).__init__()
        # self.Z_BFMS = nn.Linear(199,159645)
        self.A_BFMS = nn.Linear(99, 140970)
        # self.BFME = nn.Linear(29,159645)

    def forward(self, s):
        s1 = self.A_BFMS(s)
        # s2=self.Z_BFMS(s)
        # e=self.BFME(e)
        # out = s+e
        return s1


def BFMNeta(model_root):
    model = BFMNet()
    misc.load_state_dict_bfm(model, model_root=model_root)
    return model


def main():
    model = BFMNeta(model_root="../propressing/bfma.npz")
    # data_image_face = sio.loadmat('./mat/image_008_1.mat')
    # Shape_Para=torch.Tensor(data_image_face['Shape_Para'].reshape(-1))
    # Exp_Para=torch.Tensor(data_image_face['Exp_Para'].reshape(-1))
    data = sio.loadmat("/home/diqong/shape.mat")
    Shape_Para = torch.Tensor(data["2"].reshape(-1))
    ve = model(Shape_Para)
    print(ve)

    # vertices=param2vertices(Shape_Para,Exp_Para)
    # vertices2off('2.off', vertices.reshape(-1, 3))
    pass


if __name__ == "__main__":
    main()
