import os
import numpy as np

from pyFM.mesh import TriMesh
from tqdm.auto import tqdm

import polyscope as ps

def plot_mesh(myMesh, cmap=None):
    ps.init()
    ps_mesh = ps.register_surface_mesh("My Mesh", myMesh.vertlist, myMesh.facelist)
    if cmap is not None:
        ps_mesh.add_color_quantity("Color Map", cmap)
    ps.show()

def double_plot(myMesh1, myMesh2, cmap1=None, cmap2=None):
    ps.init()
    
    ps_mesh1 = ps.register_surface_mesh("Mesh 1", myMesh1.vertlist, myMesh1.facelist)
    if cmap1 is not None:
        ps_mesh1.add_color_quantity("Color Map 1", cmap1)
    
    ps_mesh2 = ps.register_surface_mesh("Mesh 2", myMesh2.vertlist, myMesh2.facelist)
    if cmap2 is not None:
        ps_mesh2.add_color_quantity("Color Map 2", cmap2)
    
    ps.show()

def visu(vertices):
    min_coord = np.min(vertices, axis=0, keepdims=True)
    max_coord = np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap

if __name__ == "__main__":
    meshlist = [TriMesh(f'examples/data/camel_gallop/camel-gallop-{i:02d}.off', area_normalize=True, center=True).process(k=150, intrinsic=True) for i in tqdm(range(1,11))]
    # double_plot(meshlist[0], meshlist[5])
    
    
    import pyFM.spectral as spectral
    K = 30  # Size of initial functional maps, small value since our initial maps have noise

    # All pointwise maps are located here, with format 'ind2_to_ind1' for the map from mesh ind2 to mesh ind1
    map_files = os.listdir('examples/data/camel_gallop/maps')

    maps_dict = {}

    for map_filename in tqdm(map_files):
        ind2, ind1 = map_filename.split('_to_')
        ind1, ind2 = int(ind1), int(ind2)

        # Indicing starts at 1 in the names, but at 0 on the meshlist
        mesh1, mesh2 = meshlist[ind1-1], meshlist[ind2-1]
        
        # Load the pointwise map
        p2p_21 = np.loadtxt(f'examples/data/camel_gallop/maps/{map_filename}', dtype=int)

        # Convert to functional map
        FM_12 = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=K)
        
        # Populate the dictionary
        maps_dict[(ind1-1, ind2-1)] = FM_12
    from pyFM.FMN import FMN
    # Build the network
    fmn_model = FMN(meshlist, maps_dict.copy())

    # Compute CCLB
    fmn_model.compute_CCLB(m=20)
    
    all_embs_a = []
    all_embs_c = []

    for i in range(fmn_model.n_meshes):
        CSD_a, CSD_c = fmn_model.get_CSD(i)

        all_embs_a.append(CSD_a.flatten())
        all_embs_c.append(CSD_c.flatten())

    all_embs_a = np.array(all_embs_a)
    all_embs_c = np.array(all_embs_c)
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=2)
    emb_red_a = pca_model.fit_transform(all_embs_a)
    emb_red_c = pca_model.fit_transform(all_embs_c)
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].scatter(emb_red_a[:, 0], emb_red_a[:, 1], c=np.arange(len(emb_red_a)))
    axs[0].set_title('Area CSD')

    axs[1].scatter(emb_red_c[:, 0], emb_red_c[:, 1], c=np.arange(len(emb_red_c)))
    axs[1].set_title('Conformal CSD')
    
    fmn_model.zoomout_refine(nit=10, step=5, subsample=3000, isometric=True, weight_type='icsm',
                    M_init=None, cclb_ratio=.9, n_jobs=1, equals_id=False,
                    verbose=True)
    fmn_model.compute_CCLB(m=int(0.9*fmn_model.M))
    
    all_embs_a = []
    all_embs_c = []

    for i in range(fmn_model.n_meshes):
        CSD_a, CSD_c = fmn_model.get_CSD(i)

        all_embs_a.append(CSD_a.flatten())
        all_embs_c.append(CSD_c.flatten())

    all_embs_a = np.array(all_embs_a)
    all_embs_c = np.array(all_embs_c)
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=2)
    emb_red_a = pca_model.fit_transform(all_embs_a)
    emb_red_c = pca_model.fit_transform(all_embs_c)
    
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].scatter(emb_red_a[:, 0], emb_red_a[:, 1], c=np.arange(len(emb_red_a)))
    axs[0].set_title('Area CSD')

    axs[1].scatter(emb_red_c[:, 0], emb_red_c[:, 1], c=np.arange(len(emb_red_c)))
    axs[1].set_title('Conformal CSD')
    
    plt.show()