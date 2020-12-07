import numpy as np
import scipy.io as sio
import math


def bfm_z(inputfile, outputfile):
    # 1. construct  the mean face and the base from local .mat files
    data_shape = sio.loadmat("../mat/Model_Shape.mat")
    data_expression = sio.loadmat("../mat/Model_Expression.mat")
    data_tri_mouth = sio.loadmat("../mat/Model_tri_mouth.mat")
    # 1.1 load the data from .mat files
    data_anh_model = (sio.loadmat(inputfile, squeeze_me=True, struct_as_record=False))[
        "BFM"
    ]
    # 1.2 load the data from the dicts
    tri = data_shape["tri"]
    keypoints = data_shape["keypoints"]
    tri_mouth = data_tri_mouth["tri_mouth"]
    tri_mouth[[0, 2], :] = tri_mouth[[2, 0], :]
    index = np.append(tri, tri_mouth, axis=1)

    index = index.transpose(1, 0) - 1

    # mu_shape = data_anh_model.shapeMU
    mu_shape = data_shape["mu_shape"]

    # gl coor-sys --> cv coor-sys
    # (x, y, z) --> (x, -y, -z)
    # mu_shape[1::3] = -mu_shape[1::3]
    # mu_shape[2::3] = -mu_shape[2::3]
    print(mu_shape.shape)
    # mu_shape_tensor = torch.from_numpy(mu_shape.astype(np.float32)) # to calculate loss and bp

    # w = data_anh_model.shapePC
    w = data_shape["w"]
    # w[1::3, :] = -w[1::3, :]
    # w[2::3, :] = -w[2::3, :]
    print(w)
    # w_tensor = torch.from_numpy(w.astype(np.float32)) # to calculate loss and bp

    shape_ev = data_anh_model.shapeEV
    # shape_ev_tensor = torch.from_numpy(shape_ev.astype(np.float32))

    # index = data_anh_model.faces -1
    # index[:,[0,1,2]]=index[:,[0,2,1]]
    # inner_landmark = data_anh_model.innerLandmarkIndex.astype('int') - 1
    # outer_landmark = data_anh_model.outerLandmarkIndex.astype('int') - 1

    # w_expression is the exp base
    # w_expression = data_anh_model.expPC
    w_expression = data_expression["w_exp"]
    # w_expression[1::3, :] = -w_expression[1::3, :]
    # w_expression[2::3, :] = -w_expression[2::3, :]
    # w_expression_tensor = torch.from_numpy(w_expression.astype(np.float32))# to calculate loss and bp

    mu_expression = data_anh_model.expMU
    mu_expression = data_expression["mu_exp"]
    # mu_expression[1::3] = -mu_expression[1::3]
    # mu_expression[2::3] = -mu_expression[2::3]
    # mu_expression_tensor =torch.from_numpy(mu_expression.astype(np.float32)) # to calculate loss and bp

    exp_ev = data_anh_model.expEV

    mu = mu_shape + mu_expression
    mu = mu.reshape(-1, 1)
    print(mu)
    # mu_vertex = mu.reshape(-1, 3)
    np.savez(
        outputfile,
        mu_shape=mu_shape,
        mu_expression=mu_expression,
        w_shape=w,
        w_expression=w_expression,
        mu_vertex=mu,
        shape_ev=shape_ev,
        exp_ev=exp_ev,
        index=index,
    )


def bfm_a(inputfile, outputfile):

    data_anh_model = (sio.loadmat(inputfile, squeeze_me=True, struct_as_record=False))[
        "BFM"
    ]
    mu_shape = data_anh_model.shapeMU

    # gl coor-sys --> cv coor-sys
    # (x, y, z) --> (x, -y, -z)
    # mu_shape[1::3] = -mu_shape[1::3]
    # mu_shape[2::3] = -mu_shape[2::3]
    # mu_shape_tensor = torch.from_numpy(mu_shape.astype(np.float32)) # to calculate loss and bp

    w = data_anh_model.shapePC
    # w[1::3, :] = -w[1::3, :]
    # w[2::3, :] = -w[2::3, :]

    shape_ev = data_anh_model.shapeEV

    index = data_anh_model.faces - 1
    index[:, [0, 1, 2]] = index[:, [0, 2, 1]]
    # inner_landmark = data_anh_model.innerLandmarkIndex.astype('int') - 1
    # outer_landmark = data_anh_model.outerLandmarkIndex.astype('int') - 1

    # w_expression is the exp base
    w_expression = data_anh_model.expPC
    # w_expression = data_expression['w_exp']
    # w_expression[1::3, :] = -w_expression[1::3, :]
    # w_expression[2::3, :] = -w_expression[2::3, :]
    # w_expression_tensor = torch.from_numpy(w_expression.astype(np.float32))# to calculate loss and bp

    mu_expression = data_anh_model.expMU
    # mu_expression = data_expression['mu_exp']
    # mu_expression[1::3] = -mu_expression[1::3]
    # mu_expression[2::3] = -mu_expression[2::3]
    # mu_expression_tensor =torch.from_numpy(mu_expression.astype(np.float32)) # to calculate loss and bp

    exp_ev = data_anh_model.expEV

    mu = mu_shape + mu_expression
    mu = mu.reshape(-1, 1)
    # mu_vertex = mu.reshape(-1, 3)
    np.savez(
        outputfile,
        mu_shape=mu_shape,
        mu_expression=mu_expression,
        w_shape=w,
        w_expression=w_expression,
        mu_vertex=mu,
        shape_ev=shape_ev,
        exp_ev=exp_ev,
        index=index,
    )


def param2vertices(Shape_Para, Exp_Para=None):
    """
    Input:
            shape_para: 99 * 1
            exp_rapa: 29 * 1
            camera_para:4 + 2 + 1 R-4, T-2, S-1

    """
    bfm = np.load("bfma.npz")
    mu = bfm["mu_shape"].reshape(-1, 1)
    w_shape = bfm["w_shape"]
    w_expression = bfm["w_expression"]

    a = np.dot(w_shape, Shape_Para)
    # print(a.shape)
    # print(mu.shape)
    print(w_shape)
    # print(Shape_Para.shape)
    face_vertex_tensor = mu + np.dot(
        w_shape, Shape_Para
    )  # +np.dot(w_expression, Exp_Para)
    # print(np.dot(w_shape, Shape_Para))
    # print(np.dot(w_expression, Exp_Para))
    print(face_vertex_tensor)
    # np.dot(w_shape, Shape_Para) + np.dot(w_expression, Exp_Para)

    # self.face_vertex = self.face_vertex_tensor.reshape(-1, 3).data.numpy()
    return face_vertex_tensor


def vertices2off(filename, vertices, use_camera=False):
    """
    save the face mesh to off file in ASCII format

    Input:
            filename: string
    """

    # vertices = self.face_vertex
    # if use_camera:
    # 	vert = self.transform_face().transpose(1, 0)
    # 	if config.use_cuda:
    # 		vert = vert.data.cpu().numpy()
    # 	else:
    # 		vert = vert.data.numpy()
    # 	vertices = vert

    bfm = np.load("../propressing/bfma.npz")
    num_vertex = vertices.shape[0]
    num_index = bfm["index"].shape[0]
    index = bfm["index"]
    # print(bfm['mu_vertex'])
    # print(num_index)
    fobj = open(filename, "w+")
    fobj.write("OFF\n")
    fobj.write(str(num_vertex) + " " + str(num_index) + " 0\n")
    # print(vertices[0])
    for i in range(num_vertex):
        # print(i)
        fobj.write(
            str(vertices[i][0])
            + " "
            + str(vertices[i][1])
            + " "
            + str(vertices[i][2])
            + "\n"
        )
    for i in range(num_index):
        fobj.write(
            str(3)
            + " "
            + str(index[i][0])
            + " "
            + str(index[i][2])
            + " "
            + str(index[i][1])
            + "\n"
        )
    fobj.close()

    # print('=> saved to ' + filename)


def main():
    bfm_a(inputfile="../mat/BaselFaceModel_mod.mat", outputfile="./bfma.npz")
    # bfm=np.load("bfma.npz")
    # vertices=bfm['mu_vertex']
    # vertices2off('1.off', vertices.reshape(-1, 3))
    data = sio.loadmat("/home/diqong/shape.mat")
    Shape_Para = data["2"].reshape(-1, 1)
    vertices = param2vertices(Shape_Para)
    vertices2off("2.off", vertices.reshape(-1, 3))


if __name__ == "__main__":
    main()
