import numpy as np

# MRFの事後分布の解析解を出力
def calc_gaussian_post_prob_analytical(alpha, beta, n_grad, image_noise, N_itr_EM, threshold_EM):
    """
    :param alpha: 潜在変数間の結合度を表す変数
    :param beta: ノイズ精度
    :param n_grad: 階調数
    :param image_noise: ノイズが加わった画像
    :param N_itr_EM: EMアルゴリズム最大繰り返し回数
    :param threshold_EM: EMアルゴリズム閾値
    :return: image_out: 事後分布の解析解を画像化
    """

    # 画像情報取得
    height, width = image_noise.shape
    # edgeの組み合わせを取得
    edge_list = create_edge_list(height, width)
    # MRFの事前分布の精度行列計算
    Cov_inv_prior = calc_mrf_cov_inv_mat(height, width)
    # u, s, vh = np.linalg.svd(Cov_inv_prior)  # 特異値分解(debug用)
    # 画像を行列からベクトルに変換
    image_vec = image2vec(image_noise)

    # ### EM loop start ### #
    for itr_em in range(N_itr_EM):
        # ログ表示 (更新パラメータ)
        print("itr_em:", itr_em + 1,
              ", alpha:", np.round(alpha, decimals=5),
              ", beta", np.round(beta, decimals=5))

        # 事後分布の計算
        Cov_inv_post = alpha * Cov_inv_prior + beta * np.eye(height * width)  # 事後分布の精度行列
        Cov_post = np.linalg.inv(Cov_inv_post)  # 事後分布の共分散行列
        mu_post = beta * Cov_post @ image_vec  # 事後分布の平均
        # u, s, vh = np.linalg.svd(Cov_inv_post)  # 特異値分解(debug用)

        # パラメータ推定
        alpha_new = estimate_alpha_gaussian_analytical(edge_list, Cov_post, mu_post)
        # alpha_new = alpha
        beta_new = estimate_beta_gaussian_analytical(Cov_post, mu_post, image_vec)

        # EMアルゴリズム終了条件
        if np.abs((alpha_new - alpha)/alpha) + np.abs((beta_new - beta)/beta) < threshold_EM:
            break

        # パラメータ更新
        alpha = alpha_new
        beta = beta_new
    # ### EM loop end ### #

    # 事後平均をuint8に変換
    mu_post_uint = np.round(mu_post)
    mu_post_uint[mu_post_uint > (n_grad - 1)] = n_grad - 1
    mu_post_uint[mu_post_uint < 0] = 0
    mu_post_uint = mu_post_uint.astype('uint8')

    # ベクトルから画像に変換
    image_out = vec2image(mu_post_uint, height, width)

    return image_out


# MRFの精度行列の作成
def calc_mrf_cov_inv_mat(height, width):
    """
    MRFの事前分布の精度行列の作成
        p(x_all) = Π_ij {f_ij} = N(x_all|, 0, Cov)
        f_ij = exp[ - alpha/2 * (x_i-x_j)^2] (精度alphaは1とする)
        (ノードの番号は横向きに振っていく)
    :param height: 画像の縦サイズ
    :param width: 画像の横サイズ
    :return: Cov_inv_mat 精度行列 (height*width, height*width)
    """
    # 事前分布の精度行列 (共分散行列の逆行列) の作成

    base_diag = np.zeros((width, width), dtype="float64")  # 精度行列計算のためのベースになる行列を作成
    for k in range(width):
        if k == 0:
            base_diag[k, k] = 3
            base_diag[k, k + 1] = -1

        elif k == width - 1:
            base_diag[k, k] = 3
            base_diag[k, k - 1] = -1

        else:
            base_diag[k, k] = 4
            base_diag[k, k + 1] = -1
            base_diag[k, k - 1] = -1

    # 事前分布の精度行列 (共分散行列の逆行列) の作成
    Cov_inv_mat = np.zeros((height * width, height * width), dtype="float64")
    for y in range(height):
        if y == 0:
            Cov_inv_mat[(y * width):((y + 1) * width), (y * width):((y + 1) * width)] = base_diag[:, :] - np.eye(
                width)
            Cov_inv_mat[(y * width):((y + 1) * width), ((y + 1) * width):((y + 2) * width)] = - np.eye(width)
        elif y == height - 1:
            Cov_inv_mat[(y * width):((y + 1) * width), (y * width):((y + 1) * width)] = base_diag[:, :] - np.eye(
                width)
            Cov_inv_mat[(y * width):((y + 1) * width), ((y - 1) * width):(y * width)] = - np.eye(width)
        else:
            Cov_inv_mat[(y * width):((y + 1) * width), (y * width):((y + 1) * width)] = base_diag[:, :]
            Cov_inv_mat[(y * width):((y + 1) * width), ((y + 1) * width):((y + 2) * width)] = - np.eye(width)
            Cov_inv_mat[(y * width):((y + 1) * width), ((y - 1) * width):(y * width)] = - np.eye(width)

    return Cov_inv_mat


# 潜在変数間の結合を表す係数alphaの計算
def estimate_alpha_gaussian_analytical(edge_list, Cov_post, mu_post):
    """
    潜在変数間の結合を表す係数alphaの計算
        alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
    :return: edge_list: 推定したパラメータ
    :return: Cov_post: 事後分布の平均 (画像サイズ)
    :return: mu_post: 事後分布の共分散 (画像サイズ)
    """
    n_node = Cov_post.shape[0]  # 画像サイズ
    param_tmp = 0
    for edge in edge_list:
        i, j = edge
        param_tmp += Cov_post[i, i] - 2 * Cov_post[i, j] + Cov_post[j, j] + \
                     mu_post[i] ** 2 - 2 * mu_post[i] * mu_post[j] + mu_post[j] ** 2

    alpha_new = n_node / param_tmp

    return alpha_new


def estimate_beta_gaussian_analytical(Cov_post, mu_post, image_vec):
    """
    観測ノイズの精度の推定  p(g_i|f_i) = c * exp[-beta/2 * (g_i - f_i)^2]
    :return: Cov_post: 事後分布の平均 (画像サイズ)
    :return: mu_post: 事後分布の共分散 (画像サイズ)
    :return: image_vec: 観測画像
    """
    param_tmp = 0
    for i in range(image_vec.size):
            obs = image_vec[i]  # 観測画素
            param_tmp += obs ** 2 - 2 * obs * mu_post[i] + Cov_post[i, i] + mu_post[i] ** 2

    beta_new = (image_vec.size) / param_tmp  # 更新パラメータ

    return beta_new


# MRFの全組み合わせのedgeリスト [(1,2), (2,3), ...]を作成
def create_edge_list(height, width):
    """
    MRFの全組み合わせのedgeリスト [(1,2), (2,3), ...]を作成
    :param height: 画像縦サイズ
    :param width: 画像横サイズ
    :return:
    """
    # ### edgeの登録 ### #
    edge_list = []  # 全組み合わせのedgeリスト [(1,2), (2,3), ...]

    # (height-1)*(width-1)の範囲のedgeの登録
    for y in range(height - 1):
        for x in range(width - 1):
            id = y * width + x
            pair_id_1 = np.array([id, id + 1], dtype='int32')  # 右のノードとのedge
            pair_id_2 = np.array([id, id + width], dtype='int32')  # 下のノードとのedge
            edge_list.append(pair_id_1)
            edge_list.append(pair_id_2)

    # 一番下の行の登録
    for x in range(width - 1):
        id = width * (height - 1) + x
        pair_id = np.array([id, id + 1], dtype='int32')  # 右のノードとのedge
        edge_list.append(pair_id)

    # 一番右の列の登録
    for y in range(height - 1):
        id = width * y + width - 1
        pair_id = np.array([id, id + width], dtype='int32')  # 下のノードとのedge
        edge_list.append(pair_id)

    return edge_list


# 画像を行列からベクトルに変換
def image2vec(image):
    """
    画像を行列からベクトルに変換
    :param image: 入力画像
    :return: vec: ベクトル化した画像
    """
    height, width = image.shape
    vec = np.zeros(height * width, dtype="float64")
    for y in range(height):
        vec[(y * width):((y + 1) * width)] = image[y, :]  # 画像の各行を1列に並べる

    return vec


# ベクトルから画像に変換
def vec2image(vec, height, width):
    """
    ベクトルから画像に変換
    :param vec: 画像を1列に並べたベクトル
    :param height: 画像縦サイズ
    :param width: 画像横サイズ
    :return: image: 出力画像 (height, width)
    """
    image = np.zeros((height, width), dtype="uint8")
    for y in range(height):
        image[y, :] = vec[(y * width):((y + 1) * width)]  # 画像の各行を1列に並べる

    return image
