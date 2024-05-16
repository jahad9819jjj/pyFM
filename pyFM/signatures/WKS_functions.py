import numpy as np


def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    全ての頂点についてWKSを計算する。
    WKS(x, e) = Σ_i exp(-((log(e) - log(λ_i))^2) / (2σ^2)) * φ_i(x)^2 
    λ_i:i番目の固有値, φ_i(x):頂点xにおけるi番目の固有ベクトルの値, e:エネルギー値, σ:標準偏差
    scaled=Trueの場合、各エネルギースケールでのWKSの値を正規化する。
    これにより、異なる時間スケールでのWKSの比較が可能となる。
    異なる時間スケールでWKSの比較をすることで以下の利点を得る。
    1. マルチスケール特徴の抽出:長い時間スケールのHKSでは帯域的な形状の特徴を, 短い時間スケールのHKSでは局所的な形状の詳細を
    2. ロバスト性の向上：単一時間スケールの場合はノイズや形状の微小な変化の影響を受けやすい.
    3. 識別力の高い特徴記述子の獲得
    各頂点における各時間値でのHKSの値を含む, (N, num_T)の配列を返す。
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      :
        (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       :
        (K,) array of the K corresponding eigenvalues
    energy_list :
        (num_E,) values of e to use
    sigma       :
        (float) [positive] standard deviation to use
    scaled      :
        (bool) Whether to scale each energy level

    Returns
    ------------------------
    WKS : np.ndarray
        (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:,None] - np.log(np.abs(evals))[None,:])/(2*sigma**2))  # (num_E,K)

    natural_WKS = np.einsum('tk,nk->nt', coefs, np.square(evects))  # (N,num_E)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None,:] * natural_WKS

    else:
        return natural_WKS


def lm_WKS(evals, evects, landmarks, energy_list, sigma, scaled=False):
    """
    各ランドマークにおける各エネルギー値でのWKSの値を含む, (N, num_T)の配列を返す。
    Returns the Wave Kernel Signature for some landmarks and energy values.


    Parameters
    ------------------------
    evects      :
        (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       :
        (K,) array of the K corresponding eigenvalues
    landmarks   :
        (p,) indices of landmarks to compute
    energy_list :
        (num_E,) values of e to use
    sigma       : int
        standard deviation

    Returns
    ------------------------
    landmarks_WKS : np.ndarray
        (N,num_E*p) array where each column is the WKS for a given e for some landmark
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-2)[0].flatten()
    evals = evals[indices]
    evects = evects[:,indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2*sigma**2))  # (num_E,K)
    weighted_evects = evects[None, landmarks, :] * coefs[:,None,:]  # (num_E,p,K)

    landmarks_WKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)  # (p,num_E,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E,)
        landmarks_WKS = ((1/inv_scaling)[None,:,None] * landmarks_WKS)

    return landmarks_WKS.reshape(-1,evects.shape[0]).T  # (N,p*num_E)


def auto_WKS(evals, evects, num_E, landmarks=None, scaled=True):
    """
    auto_KSを用いて, エネルギー値のリストを生成する。この関数は固有値の最小値と最大値に基づいて対数スケールで等間隔に選択する。

    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       :
        (K,) array of  K eigenvalues
    evects      :
        (N,K) array with K eigenvectors
    landmarks   :
        (p,) If not None, indices of landmarks to compute.
    num_E       :
        (int) number values of e to use
    Returns
    ------------------------
    WKS or lm_WKS : np.ndarray
        (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min,e_max = np.log(abs_ev[1]),np.log(abs_ev[-1])
    sigma = 7*(e_max-e_min)/num_E

    e_min += 2*sigma
    e_max -= 2*sigma

    energy_list = np.linspace(e_min,e_max,num_E)

    if landmarks is None:
        return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)
    else:
        return lm_WKS(abs_ev, evects, landmarks, energy_list, sigma, scaled=scaled)


def mesh_WKS(mesh, num_E, landmarks=None, k=None):
    """
    メッシュ上の各頂点における熱拡散仮定の時間変化を表現（AKAZEっぽい？）
    1. Laplace-Beltrami演算を用いて固有値と固有ベクトルを計算(mesh.eigenvalues, mesh.eigenvectors)
    2. auto_WKS(内部でエネルギー値のリストと対応する標準偏差を自動で計算してWKS, lm_WKSを計算)
    3. 記述子を得る
    Compute the Wave Kernel Signature for a mesh

    Parameters
    ------------------------
    mesh    : TriMesh
        mesh on which to compute the XKS
    num_T   : int
        number of time values to use
    landmarks : np.ndarray, optional
        (p,) indices of landmarks to use
    k       : int, optional
        number of eigenvalues to use

    Returns
    ------------------------
    WKS: np.ndarray
        (N,num_T) array where each line is the HKS for a given t
    """
    assert mesh.eigenvalues is not None, "Eigenvalues should be processed"

    if k is None:
        k = len(mesh.eigenvalues)
    else:
        assert len(mesh.eigenvalues >= k), f"At least ${k}$ eigenvalues should be computed, not {len(mesh.eigenvalues)}"

    return auto_WKS(mesh.eigenvalues[:k], mesh.eigenvectors[:, :k], num_E, landmarks=landmarks, scaled=True)
