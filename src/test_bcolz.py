import bcolz
import os
import torch
import numpy as np
from PIL import Image

path = "/ssd-data/lmd/eval_dbs"

names = ["agedb_30"]
for name in names:
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode="r")
    print(carray.shape)
    print(carray[-1].transpose(1, 2, 0).shape)
    print((carray[-1].transpose(1, 2, 0))[55:65, 55:65])
    img = Image.fromarray(
        (carray[-1].transpose(1, 2, 0).astype(np.float32) * 255).astype(np.uint8)
    )
    img.save("/data2/lmd_jdq/cfp-fp/%d.jpg" % 0)
    for i in range(1, 20):
        print(np.sum(carray[-i] - carray[-1]))
