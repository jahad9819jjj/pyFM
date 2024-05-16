import os
import numpy as np

from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping
from tqdm.auto import tqdm

import polyscope as ps

def plot_mesh(myMesh, cmap=None):
    ps.init()
    ps_mesh = ps.register_surface_mesh("My Mesh", myMesh.vertlist, myMesh.facelist)
    if (cmap is not None):
        if (cmap.shape[1] == 1):
            ps_mesh.add_scalar_quantity("Color Map", cmap)
        elif (cmap.shape[1] == 3):
            ps_mesh.add_color_quantity("Color Map", cmap)
        else:
            raise ValueError("Invalid color map shape")
    ps.show()

def double_plot(myMesh1, myMesh2, cmap1=None, cmap2=None):
    ps.init()
    
    ps_mesh1 = ps.register_surface_mesh("Mesh 1", myMesh1.vertlist, myMesh1.facelist)
    if cmap1 is not None:
        if (cmap1.shape[1] == 1):
            ps_mesh1.add_scalar_quantity("Color Map 1", cmap1)
        elif (cmap1.shape[1] == 3):
            ps_mesh1.add_color_quantity("Color Map 1", cmap1)
        else:
            raise ValueError("Invalid color map shape")
    
    ps_mesh2 = ps.register_surface_mesh("Mesh 2", myMesh2.vertlist, myMesh2.facelist)
    if cmap2 is not None:
        if (cmap2.shape[1] == 1):
            ps_mesh2.add_scalar_quantity("Color Map 2", cmap2)
        elif (cmap2.shape[1] == 3):
            ps_mesh2.add_color_quantity("Color Map 2", cmap2)
        else:
            raise ValueError("Invalid color map shape")
    
    ps.show()

def visu(vertices):
    min_coord = np.min(vertices, axis=0, keepdims=True)
    max_coord = np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap

if __name__ == "__main__":
    # mesh1 = TriMesh('examples/data/cat-00.off', area_normalize=True, center=False)
    # # mesh1 = TriMesh('/Users/jinhirai/Downloads/Dataset/Mug/Mug4_remesh.ply', area_normalize=True, center=False)
    # mesh2 = TriMesh(mesh1.vertlist, mesh1.facelist)
        
    # # Attributes are computed on the fly and cached
    # edges = mesh1.edges
    # area = mesh1.area
    # face_areas = mesh1.face_areas
    # vertex_areas = mesh1.vertex_areas
    # face_normals = mesh1.normals

    # # AREA WEIGHTED VERTEX NORMALS
    # vertex_normals_a = mesh1.vertex_normals

    # # UNIFORM WEIGHTED VERTEX NORMALS
    # mesh1.set_vertex_normal_weighting('uniform')
    # vertex_normals_u = mesh1.vertex_normals

    # dists = mesh1.geod_from(1000, robust=True)
    # S1_geod = mesh1.get_geodesic(verbose=True)
    # mesh1.process(k=10, intrinsic=False, verbose=True)
    # plot_mesh(mesh1, mesh1.eigenvectors[:,2])

    # mesh1 = TriMesh('examples/data/cat-00.off')
    # mesh2 = TriMesh('examples/data/lion-00.off')
    mesh1 = TriMesh('/Users/jinhirai/Downloads/Dataset/Mug/Mug4_remesh.ply')
    mesh2 = TriMesh('/Users/jinhirai/Downloads/Dataset/Mug/Mug12_remesh.ply')
    print(f'Mesh 1 : {mesh1.n_vertices:4d} vertices, {mesh1.n_faces:5d} faces\n'
        f'Mesh 2 : {mesh2.n_vertices:4d} vertices, {mesh2.n_faces:5d} faces')

    # double_plot(mesh1,mesh2)

    process_params = {
        # 'n_ev': (35,35),  # Number of eigenvalues on source and Target
        'n_ev': (10,10),  # Number of eigenvalues on source and Target
        # 'landmarks': np.loadtxt('examples/data/landmarks.txt',dtype=int)[:5],  # loading 5 landmarks
        'subsample_step': 5,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
    }

    model = FunctionalMapping(mesh1,mesh2)
    model.preprocess(**process_params,verbose=True)

    fit_params = {
        'w_descr': 1e0,
        'w_lap': 1e-2,
        'w_dcomm': 1e-1,
        'w_orient': 0
    }



    model.fit(**fit_params, verbose=True)

    p2p_21 = model.get_p2p(n_jobs=1)
    cmap1 = visu(mesh1.vertlist); cmap2 = cmap1[p2p_21]
    double_plot(mesh1,mesh2,cmap1,cmap2)

    model.icp_refine(verbose=True)
    p2p_21_icp = model.get_p2p()
    cmap1 = visu(mesh1.vertlist); cmap2 = cmap1[p2p_21_icp]
    double_plot(mesh1,mesh2,cmap1,cmap2)

    model.change_FM_type('classic') # We refine the first computed map, not the icp-refined one
    model.zoomout_refine(nit=15, step = 1, verbose=True)
    print(model.FM.shape)
    p2p_21_zo = model.get_p2p()
    cmap1 = visu(mesh1.vertlist); cmap2 = cmap1[p2p_21_zo]
    double_plot(mesh1,mesh2,cmap1,cmap2)