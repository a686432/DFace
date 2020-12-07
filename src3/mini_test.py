# from dataloader import readobj2,write_obj
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(80000000, 5000))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh,result_ransac,voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
    return result

def icp_test(src,dst):
    #points, _ = readobj2("meanface.obj")
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(dst)

    #points = readobj("/home/jdq/projects/data/bs000_YR_R90_0.obj")
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(src)

    voxel_size = 4
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    #draw_registration_result(source, target,
    #                            result_ransac.transformation)

    # source_temp = copy.deepcopy(source)
    # source_temp.transform(result_ransac.transformation)
    # voxel_size = 4
    # source_down, source_fpfh = preprocess_point_cloud(source_temp, voxel_size)
    # # target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)

    print("Apply point-to-point ICP")
    # reg_p2p = o3d.registration.registration_icp(
    #     source, target, 10,
    #     estimation_method=o3d.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    print("")
    #draw_registration_result(source, target, reg_p2p.transformation)

    # draw_registration_result(source_temp, target,
    #                             result_ransac.transformation)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac,
                                        voxel_size)

    source = source.transform(result_icp.transformation)
    print(result_icp)
    print(np.asarray(source.points))
    print(np.asarray(result_icp.correspondence_set))
    return np.asarray(source.points),np.asarray(result_icp.correspondence_set)
    


    
    #draw_registration_result(source, target, result_icp.transformation)


def sym_correspondence(src,dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def flip_mesh(vertices,faces):
    write_obj('../tmp_out/meanface_raw.obj',vertices.reshape(-1,3),faces)
    vertices_f = vertices.copy()
    vertices_f[:,0] = -vertices[:,0]
    vertices_f,corres = icp_test(vertices_f.reshape(-1,3),vertices.reshape(-1,3))
    dist,indices = sym_correspondence(vertices_f.reshape(-1,3),vertices.reshape(-1,3))
    np.save('../propressing/sym_flip.npy',corres)
    print(indices,dist.mean())
    write_obj('../tmp_out/meanface_flip_r.obj',vertices_f.reshape(-1,3),faces)
    vertices_f[corres[:,0]] = vertices_f[corres[:,1]]
    write_obj('../tmp_out/meanface_flip.obj',vertices_f.reshape(-1,3),faces)
    

def get_adj_matrix(mesh):
    max_adj = 0
    adj_sum = np.zeros(len(mesh.adjacency_list),dtype='int')
    for i,vertices_adj in enumerate(mesh.adjacency_list):
        #adj_sum.append(len(vertices_adj))
        if len(vertices_adj)>max_adj:
            max_adj = len(vertices_adj)
            
            
    print('max_adj:',max_adj)
    adj_matrix  = np.zeros((len(mesh.adjacency_list),max_adj))
    adj_mask = np.zeros((len(mesh.adjacency_list),max_adj))
    
    for i,vertices_adj in enumerate(mesh.adjacency_list):
        for j in vertices_adj:
            print(adj_sum[i])
            adj_matrix[i,adj_sum[i]] = j
            adj_mask[i,adj_sum[i]] = 1
            adj_sum[i] +=1
    
    return adj_matrix,adj_mask

    





 
def edge_list():
    edge_list = []
    count_boundary_edge = 0
    mesh = o3d.io.read_triangle_mesh('../propressing/meanface.obj')
    mesh = mesh.compute_adjacency_list()
    

            

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    adj_matrix,adj_mask = get_adj_matrix(mesh)
    np.savez('../propressing/shape_smooth.npz',adj_matrix=adj_matrix,adj_mask=adj_mask)


    mesh_face = np.array(mesh.faces)

    # vertex_manifold = mesh.is_vertex_manifold()
    # self_intersecting = mesh.is_self_intersecting()
    # watertight = mesh.is_watertight()
    # orientable = mesh.is_orientable()

    #print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    print(np.asarray(edges))
    edges = np.asarray(edges)
    #edges_face = 
    print(edges.shape)

    for i,vertices_adj in enumerate(mesh.adjacency_list):
        for j in vertices_adj:
            
            b = ([i,j] == edges).all(1).any()
            c = ([j,i] == edges).all(1).any()
            if b or c:
                print([i,j],edges[b])
                count_boundary_edge +=1
                edge_list.append([i,j,1])
            else:
                edge_list.append([i,j,0])


        
                
            
            # for k in edges:
            #     #print(k)
            #     if (np.array([i,j])==k).all() and i<j:
            #         # x,y = np.where(edges == np.array([i,j]))
            #         # print(np.where(edges == np.array([i,j])))
            #         # print(edges[x[0],y[0]])
            #         count_boundary_edge +=1
            #         print([i,j])
            #         break
                    #print(count_boundary_edge)
    edge_numpy = np.array(edge_list)
    np.save('edges.npy',edge_numpy)
    print(count_boundary_edge)
    # print(f"  vertex_manifold:        {vertex_manifold}")
    # print(f"  self_intersecting:      {self_intersecting}")
    # print(f"  watertight:             {watertight}")
    # print(f"  orientable:             {orientable}")
    # print(mesh)
    # print(len(mesh.adjacency_list))
    # hmesh =o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    # print(hmesh.half_edges)
    # print(hmesh)
    # print(len(hmesh.get_boundaries()))


def write_obj_with_texuture():
    #import open3d as o3d
    import scipy.io as sio
    from glob import glob
    import os
    C = sio.loadmat('Model_Tex.mat')
    colors = C['tex'].T; triangles = C['tri'].T-1; uv_coords=C['uv']
    uv_coords[:,1] = 1-uv_coords[:,1]
    uv_h = uv_w = 128
    triangles_uv = uv_coords[triangles].reshape(-1,2)
    print(triangles_uv.shape)

    image_folder='test/'
    outdir = 'result/'
    types = ('*.obj')
    image_path_list= []
    for files in types:
        print(types)
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    print(total_num)
    for i, imgname in enumerate(image_path_list):
    # mesh = o3d.io.read_triangle_mesh('test/1_shape.obj')
    # mesh.texture = o3d.io.read_image('test/1_abedlo.png')
    #mesh_mean = o3d.io.read_triangle_mesh('meanface.obj')
        mesh = o3d.io.read_triangle_mesh(imgname)
        print(imgname)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.texture = o3d.io.read_image(imgname.replace('shape.obj','texture.png'))

        mesh.triangle_uvs = triangles_uv
        #print(np.asarray(mesh.vertex_colors))
        #print(mesh.triangle_uvs)
        o3d.io.write_triangle_mesh(outdir+os.path.basename(imgname),mesh,write_triangle_uvs=True)
    #mesh.uv_texture_map

def test_fft():
    import cv2
    import torch
    img = cv2.imread('uv_texture_map.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)*1./255
    img = torch.Tensor(img)
    a = torch.rfft(img,2,normalized=True,onesided=False)
    print(a.shape)

    a = a.norm(dim=-1).clamp(min=0,max=1)
    print(a.shape)
    print(a)
    #cv2.namedWindow("the window0")
    cv2.imwrite("test.jpg", a.data.numpy()*255)
    # cv2.waitKey()


def bfm():
    bfm=np.load("../propressing/bfmz.npz")
    print(list(bfm.keys()))
    print(bfm['index'].min())
    

    




if __name__ == "__main__":
    #v,f=readobj2('../propressing/meanface.obj')
    #v = v.reshape(-1)
    #flip_mesh(v,f)
    bfm()
    
    
