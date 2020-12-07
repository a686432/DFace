import config
import numpy as np
from sklearn.neighbors import NearestNeighbors

# import icp
import misc
import os
from tqdm import tqdm
import data_loader
import sys
from torchvision import datasets, transforms
import net
import torch


class EvalToolBox(object):
    def __init__(self):

        self.micc_image_root = "/home/jdq/projects/FR_AAAI/output3"
        # self.micc_loader = 1
        # self.dict_file = ''
        self.micc_obj_root = "/home/jdq/data/florence/"
        self.filelist = "/home/jdq/projects/FR_AAAI/output3/files_fs.txt"

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform_eval_fs = transform_eval_fs = transforms.Compose(
            [transforms.CenterCrop((112, 96)), transforms.ToTensor(), normalize]
        )
        self.evalset_micc = data_loader.MICCDataSet(
            root=self.micc_image_root,
            filelist=self.filelist,
            transform=self.transform_eval_fs,
        )
        self.eval_loader_micc = torch.utils.data.DataLoader(
            self.evalset_micc, batch_size=1, shuffle=False, num_workers=1
        )
        self.bfm = np.load("../propressing/bfma.npz")
        self.num_index = self.bfm["index"].shape[0]
        self.index = self.bfm["index"]

    def read_mesh(self, file_name):
        with open(file_name, "r") as f:
            lines = f.readlines()
            vertices = []
            faces = []
            for line in lines:
                words = line.split(" ")
                if words[0] == "v":
                    ver = np.zeros(5)
                    ver[:] = float(words[1]), float(words[2]), float(words[3]), 1, -1
                    vertices.append(ver)
                if words[0] == "f":
                    face = np.zeros(4, dtype=int)
                    face[:] = (
                        int(words[1].split("/")[0]),
                        int(words[2].split("/")[0]),
                        int(words[3].split("/")[0]),
                        1,
                    )
                    faces.append(face)
            vertices = np.array(vertices)
            faces = np.array(faces)
        return vertices, faces

    def crop_radius(self, mesh_vertices, mesh_faces):
        def radius(v1, v2):
            return np.sqrt(((v1[0:3] - v2[0:3]) ** 2).sum())

        zmax_index = np.argmax(mesh_vertices[:, 2])
        vertices = np.zeros((mesh_vertices.shape[0], 5))
        vertices[:, 0:3] = mesh_vertices
        vertices[:, 3] = 1
        vertices[:, 4] = -1
        faces = np.zeros((mesh_faces.shape[0], 4), dtype=int)
        faces[:, 0:3] = mesh_faces
        faces[:, 3] = 1

        mesh_vertices = vertices
        mesh_faces = faces

        origin = mesh_vertices[zmax_index]
        target_vertices = []
        target_faces = []
        # face=np.zeros(mesh_vertices.shape())

        for idx, vertex in enumerate(mesh_vertices):
            if radius(vertex, origin) > 95:
                mesh_vertices[idx, 3] = 0
            else:
                target_vertices.append(mesh_vertices[idx, 0:3])

        i = 0
        for idx, vertex in enumerate(mesh_vertices):
            if mesh_vertices[idx, 3] == 1:
                i += 1
                mesh_vertices[idx, 4] = i
        for idx, face in enumerate(mesh_faces):
            if (mesh_vertices[face[0:3] - 1] == 0).any():
                mesh_faces[idx, 3] = 0
            else:
                mesh_faces[idx, 0:3] = mesh_vertices[mesh_faces[idx, 0:3] - 1, 4]
                target_faces.append(mesh_faces[idx, 0:3])
        target_vertices = np.array(target_vertices)
        target_faces = np.array(target_faces)

        return target_vertices, target_faces

    def vertices2obj(self, vertices_group, labels, imname):
        # import icp
        batch_size = vertices_group.shape[0]
        error = 0
        num = 0
        root = "../output"

        for idx, vertices in enumerate(vertices_group):
            imname = os.path.splitext(os.path.split(imname[0])[1])[0]
            num_vertex = vertices.shape[0]
            v2, f2 = self.crop_radius(vertices, self.index)
            micc_dir = os.path.join(self.micc_obj_root, str(labels[idx]))
            if os.path.exists(micc_dir):
                files = os.listdir(micc_dir)
                for file_n in files:
                    file_n = os.path.join(micc_dir, file_n)
                    v1, _ = self.read_mesh(file_n)
                    # v2, _ = icp.read_mesh(filename)
                    err = self._icp(v1, v2)
                    error += err
                    num += 1
                return error / num
            else:
                print(micc_dir)
                return 4

    def get_micc_rmse(self, model, save_tmp=False):

        """
        save the face mesh to off file in ASCII format

        Input:
            filename: string
        """

        print("evaluting...")
        model.eval()
        total_loss = 0
        for batch_idx, (data, target, image_name) in enumerate(self.eval_loader_micc):
            data, target = data.cuda(), target.cuda()
            _, (_, vertex, _) = model(data)
            loss = self.vertices2obj(
                vertex.data.cpu().numpy().reshape(len(vertex), -1, 3),
                target.data.cpu().numpy(),
                image_name,
            )
            total_loss += loss
            sys.stdout.write(
                "%d/%d | Loss: %.5f \r"
                % (
                    batch_idx + 1,
                    len(self.eval_loader_micc),
                    total_loss / (batch_idx + 1),
                )
            )
        sys.stdout.write("\n")
        print("Fanished!")

    def _icp(self, v1, v2):
        def best_fit_transform(A, B):
            """
            Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
            Input:
            A: Nxm numpy array of corresponding points
            B: Nxm numpy array of corresponding points
            Returns:
            T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
            R: mxm rotation matrix
            t: mx1 translation vector
            """

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
                Vt[m - 1, :] *= -1
                R = np.dot(Vt.T, U.T)

            # translation
            t = centroid_B.T - np.dot(R, centroid_A.T)

            # homogeneous transformation
            T = np.identity(m + 1)
            T[:m, :m] = R
            T[:m, m] = t

            return T, R, t

        def nearest_neighbor(src, dst):
            """
            Find the nearest (Euclidean) neighbor in dst for each point in src
            Input:
                src: Nxm array of points
                dst: Nxm array of points
            Output:
                distances: Euclidean distances of the nearest neighbor
                indices: dst indices of the nearest neighbor
            """

            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(dst)
            distances, indices = neigh.kneighbors(src, return_distance=True)
            return distances.ravel(), indices.ravel()

        vertices1 = v1
        vertices2 = v2

        zmax_index1 = np.argmax(vertices1[:, 2])
        origin1 = vertices1[zmax_index1]

        zmax_index2 = np.argmax(vertices2[:, 2])
        origin2 = vertices2[zmax_index2]

        vertices1 = vertices1[:, 0:3] + origin2[0:3] - origin1[0:3]
        vertices2 = vertices2[:, 0:3]

        prev_error = 0

        for i in range(100):
            # find the nearest neighbors between the current source and destination points
            distances, indices = nearest_neighbor(vertices2, vertices1)

            # compute the transformation between the current source and nearest destination points
            T, R, t = best_fit_transform(vertices2, vertices1[indices])

            # update the current source
            vertices2 = (np.dot(R, vertices2.T)).T + t

            # check error
            mean_error = np.sqrt(np.mean(distances ** 2))
            if np.abs(prev_error - mean_error) < 0.00001:
                break

            prev_error = mean_error
            # print(mean_error)

        return mean_error


if __name__ == "__main__":
    print("loading model...")
    dict_file = "/home/jdq/model/dict.cl_382500.cl"
    model = net.sphere64a(pretrained=True, model_root=dict_file)
    model = model.cuda()
    print("Loading...\n")
    state = torch.load(dict_file)
    state_dict = state
    misc.load_state_dict(model, dict_file)
    print("evaluting...")
    e = EvalToolBox()
    e.get_micc_rmse(model)
    # evalMICC()
