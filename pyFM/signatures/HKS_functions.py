import numpy as np


def HKS(evals, evects, time_list,scaled=False):
    """
    全ての頂点についてHKSを計算する。
    $HKS(x, t) = Σ_i exp(-λ_i * t) * φ_i(x)^2$
    λ_i:i番目の固有値, φ_i(x):頂点xにおけるi番目の固有ベクトルの値, t:時間値
    scaled=Trueの場合、各時間値についてHKSの値を正規化する。
    これにより、異なる時間スケールでのHKSの比較が可能となる。
    異なる時間スケールでHKSの比較をすることで以下の利点を得る。
    1. マルチスケール特徴の抽出:長い時間スケールのHKSでは帯域的な形状の特徴を, 短い時間スケールのHKSでは局所的な形状の詳細を
    2. ロバスト性の向上：単一時間スケールの場合はノイズや形状の微小な変化の影響を受けやすい.
    3. 識別力の高い特徴記述子の獲得
    各頂点における各時間値でのHKSの値を含む, (N, num_T)の配列を返す。
    Returns the Heat Kernel Signature for num_T different values.
    The values of the time are interpolated in logscale between the limits
    given in the HKS paper. These limits only depends on the eigenvalues.

    Parameters
    ------------------------
    evals     :
        (K,) array of the K eigenvalues
    evecs     :
        (N,K) array with the K eigenvectors
    time_list :
        (num_T,) Time values to use
    scaled    :
        (bool) whether to scale for each time value

    Returns
    ------------------------
    HKS : np.ndarray
        (N,num_T) array where each line is the HKS for a given t
    """
    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    natural_HKS = np.einsum('tk,nk->nt', coefs, np.square(evects))

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T)
        return (1/inv_scaling)[None,:] * natural_HKS

    else:
        return natural_HKS


def lm_HKS(evals, evects, landmarks, time_list, scaled=False):
    """
    ランドマークにて指定された頂点のみについてHKSを計算する
    各ランドマークにおける各時間値でのHKSの値を含む, (N, num_T)の配列を返す。

    Returns the Heat Kernel Signature for some landmarks and time values.

    Parameters
    ------------------------
    evects      :
        (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       :
        (K,) array of the K corresponding eigenvalues
    landmarks   :
        (p,) indices of landmarks to compute
    time_list   :
        (num_T,) values of t to use

    Returns
    ------------------------
    landmarks_HKS : np.ndarray
        (N,num_E*p) array where each column is the HKS for a given t for some landmark
    """

    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    weighted_evects = evects[None, landmarks, :] * coefs[:,None,:]  # (num_T,p,K)

    landmarks_HKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)  # (p,num_T,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T,)
        landmarks_HKS = (1/inv_scaling)[None,:,None] * landmarks_HKS

    return landmarks_HKS.reshape(-1, evects.shape[0]).T  # (N,p*num_E)


def auto_HKS(evals, evects, num_T, landmarks=None, scaled=True):
    """
    auto_HKSを用いて, 時間値のリストを生成する。この関数は固有値の最小値と最大値に基づいて対数スケールで時間値を選択する。

    Compute HKS with an automatic choice of tile values

    Parameters
    ------------------------
    evals       :
        (K,) array of  K eigenvalues
    evects      :
        (N,K) array with K eigenvectors
    landmarks   :
        (p,) if not None, indices of landmarks to compute.
    num_T       :
        (int) number values of t to use
    Returns
    ------------------------
    HKS or lm_HKS : np.ndarray
        (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    for some landmark
    """

    abs_ev = sorted(np.abs(evals))
    t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], num_T)

    if landmarks is None:
        return HKS(abs_ev, evects, t_list, scaled=scaled)
    else:
        return lm_HKS(abs_ev, evects, landmarks, t_list, scaled=scaled)


def mesh_HKS(mesh, num_T, landmarks=None, k=None):
    """
    メッシュ上の各頂点における熱拡散仮定の時間変化を表現（AKAZEっぽい？）
    1. Laplace-Beltrami演算を用いて固有値と固有ベクトルを計算(mesh.eigenvalues, mesh.eigenvectors)
    2. auto_HKS(内部で時間値を自動で計算してHKS, lm_HKSを計算)
    3. 記述子を得る
    Compute the Heat Kernel Signature for a mesh

    Parameters
    ------------------------
    mesh    : TriMesh
        mesh on which to compute the HKS
    num_T   : int
        number of time values to use
    landmarks : np.ndarray, optional
        (p,) indices of landmarks to use
    k       : int, optional
        number of eigenvalues to use

    Returns
    ------------------------
    HKS: np.ndarray
        (N,num_T) array where each line is the HKS for a given t
    """

    assert mesh.eigenvalues is not None, "Eigenvalues should be processed"

    if k is None:
        k = len(mesh.eigenvalues)
    else:
        assert len(mesh.eigenvalues >= k), f"At least ${k}$ eigenvalues should be computed, not {len(mesh.eigenvalues)}"

    return auto_HKS(mesh.eigenvalues[:k], mesh.eigenvectors[:,:k], num_T, landmarks=landmarks, scaled=True)
