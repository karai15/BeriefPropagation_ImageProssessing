import numpy as np
import sys


class Edge:
    def __init__(self, pair_id, n_grad):
        self.pair_id = pair_id  # edgeが接続するノードのペアのid サイズ2のndarray
        self.n_grad = n_grad  # 階調数
        self.post_joint_prob = np.zeros((n_grad, n_grad), dtype='float64')  # 結合事後分布 p(f_i,f_j|f_all)


# MRF作成 (ノードの集合を保存)
class MRF:
    def __init__(self, n_grad):
        self.nodes = {}  # MRF上のノード (key:id, value:node)
        self.edges = []  # MRF上のエッジの集合
        self.n_grad = n_grad  # 階調数

    """
    #### memo ####
    # MRFにnodeを追加するとき
    mrf.nodes[node.id] = node

    # MRFからidに対応するnodeを返してほしいとき
    node = mrf.nodes[node.id]
    """

    # 確率伝搬BPを実行
    def belief_propagation(self, N_itr_BP, N_itr_EM, alpha, beta, q_error, image_noise, threshold_EM, option_model):
        """
        確率伝搬BPを実行 (動機スケジューリング)
        :param N_itr_BP: BPイタレーション
        :param N_itr_EM: EMイタレーション
        :param alpha: 潜在変数間の結合係数
        :param threshold_EM: EMアルゴリズム終了条件用の閾値 (パラメータの変化率)
        :param option_model: モデル選択オプション "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q), "gaussian":ガウス分布(精度beta)
        :return: メッセージの更新を行い、各ノードインスタンスにメッセージと周辺事後分布
        """

        # 画像情報取得
        height = image_noise.shape[0]
        width = image_noise.shape[1]
        n_edge = width * (height - 1) + (width - 1) * height  # 画像全体のエッジ数

        # 各ノードのメッセージ初期化
        for node in self.nodes.values():
            node.initialize_message()

        # ### EM loop start ### #
        for itr_em in range(N_itr_EM):

            # ログ表示 (更新パラメータ)
            print("itr_em:", itr_em + 1,
                  ", alpha:", np.round(alpha, decimals=3),
                  ", q_error:", np.round(q_error, decimals=3),
                  ", beta", np.round(beta, decimals=3))

            # 尤度の計算
            # 各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
            for y in range(height):
                for x in range(width):
                    node = self.nodes[width * y + x]  # ノードの取り出し
                    node.calc_likelihood(beta, q_error, image_noise[y, x], option_model)

            # BPイタレーション開始
            for itr_bp in range(N_itr_BP):
                # print("EM iteration:", itr_em, " BP iteration:", itr_bp)  # イタレーションの出力

                # 各ノードが近傍ノードに対してメッセージを送信 (同期スケジューリング)
                for node in self.nodes.values():
                    for neighbor_target in node.neighbor_set:
                        # nodeからneighbor_targetへの送信メッセージをneighborの受信メッセージとして保存
                        neighbor_target.receive_message[node] = node.send_message(neighbor_target, alpha)

            # 各ノードの周辺事後分布を計算 p(f_i|g_all)
            for node in self.nodes.values():
                node.calc_post_marginal()

            # ノード間の結合事後分布を計算 p(f_i,f_j|g_all) (edge.post_joint_probに保存)
            self.calc_post_joint(alpha)

            # ### パラメータの更新 ### #
            if option_model == "sym":  # qの推定 (K次元対称通信路)
                q_error_new = self.estimate_q_error(image_noise)  # qの更新
                beta_new = beta  # betaは更新なし
            elif option_model == "gaussian":  # betaの推定 (ガウスノイズ)
                beta_new = self.estimate_beta(image_noise)  # betaの更新
                q_error_new = q_error  # qは更新なし
            # alphaの推定
            alpha_new = self.estimate_alpha(height, width)

            # 更新無し (debug)
            # alpha_new = alpha
            # beta_new = beta
            # q_error_new = q_error

            # EMアルゴリズム終了条件
            if (np.abs(alpha - alpha_new) + np.abs(beta - beta_new) + np.abs(q_error - q_error_new)) < threshold_EM:
                # パラメータ更新
                alpha = alpha_new
                beta = beta_new
                q_error = q_error_new
                break

            # パラメータ更新
            alpha = alpha_new
            beta = beta_new
            q_error = q_error_new

        # ### EM loop end ### #

        return alpha, beta, q_error

    # q_errorの計算
    def estimate_q_error(self, image_noise):
        """
        潜在変数が異なる1つの画素へ誤る確率q_errorを推定
            p(g|f) = q*(1-δ(g-f)) + (1-(n_grad-1)*q)*δ(g-f)  # g:観測(n_grad状態), f:潜在(n_grad状態)
        :param image_noise: ノイズが加わった観測画像
        :return: q_error: 推定パラメータ
        """
        # 画像情報
        height = image_noise.shape[0]
        width = image_noise.shape[1]

        param_tmp = 0  # 更新パラメータq_error計算用の変数
        for y in range(height):
            for x in range(width):
                node = self.nodes[y * width + x]
                observed = image_noise[y, x]
                post_marginal_prob = np.copy(node.post_marginal_prob)  # 周辺事後確率の取得

                # 観測画素observedと異なる潜在変数をとる事後確率の和を計算 (事後分布での誤り率の和)
                post_marginal_prob[observed] = 0
                param_tmp += np.sum(post_marginal_prob)

        q_error = param_tmp / ((self.n_grad - 1) * image_noise.size)  # 平均化してパラメータを更新

        return q_error

    # alphaの推定
    def estimate_alpha(self, height, width):
        """
        潜在変数間の結合を表す係数alphaの計算
            alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
        :param height: 画像縦サイズ
        :param width: 画像横サイズ
        :return: alpha: 推定したパラメータ
        """
        # expの中身の関数形(ガウス型) (f_i - f_j)^2 の計算
        Phy_ij = np.zeros((self.n_grad, self.n_grad), dtype='float64')
        for k in range(self.n_grad):
            for l in range(self.n_grad):
                Phy_ij[k, l] = (k - l) ** 2  # expの中身の関数形(ガウス型) (f_i - f_j)^2

        # パラメータalphaの計算
        param_tmp = 0  # 更新パラメータalpha計算用の変数
        for edge in self.edges:  # すべてのedgeに対して実行
            post_joint_prob = edge.post_joint_prob  # 結合事後確率を取り出す
            param_tmp += np.sum(Phy_ij * post_joint_prob)  # 期待値計算(Phy_ijに結合事後分布で重み付けして足し合わせる)

        # alpha = n_edge * (1 / (2*param_tmp))  # エッジ数で平均化してパラメータを更新
        alpha = height * width * (1 / param_tmp)  # エッジ数で平均化してパラメータを更新

        return alpha

    # betaの推定
    def estimate_beta(self, image_noise):
        """
        観測ノイズの精度の推定  p(g_i|f_i) = c * exp[(g_i - f_i)^2]
        :param beta: ノイズ精度(分散の逆数)
        :param image_noise: ノイズが加わった観測画像
        :return: beta: 推定したパラメータ
        """
        # 画像情報
        height = image_noise.shape[0]
        width = image_noise.shape[1]

        param_tmp = 0  # 更新パラメータalpha計算用の変数
        var_latent = np.linspace(0, self.n_grad - 1, self.n_grad)  # 潜在変数の候補 (0~階調数までの長さのベクトルを用意)
        for y in range(height):
            for x in range(width):
                node = self.nodes[y * width + x]
                observed = image_noise[y, x]  # 観測画素
                var_observed = observed * np.ones(self.n_grad, dtype="float64")  # 観測変数ベクトル(n_grad次元)
                post_marginal_prob = node.post_marginal_prob  # 周辺事後分布 (n_grad次元のベクトル)
                param_tmp += np.dot((var_latent - var_observed) ** 2, post_marginal_prob)  # 期待値計算(事後分布の重み付き平均化)

        beta_new = height * width * (1 / param_tmp)  # 更新パラメータ

        return beta_new

    # 結合事後分布 p(f_i,f_j|f_all) の計算
    def calc_post_joint(self, alpha):

        for edge in self.edges:  # すべてのedgeに対して結合分布を計算
            # edgeに接続されたnodeを取得
            id_1 = edge.pair_id[0]
            id_2 = edge.pair_id[1]
            node_1 = self.nodes[id_1]
            node_2 = self.nodes[id_2]

            # node_1について node_2以外からのメッセージの積を計算
            message_product_1 = np.ones(self.n_grad, dtype='float64')  # node_1の近傍からのメッセージの積(node_2からのメッセージは除く)
            for neigbor_node in node_1.receive_message.keys():
                if neigbor_node.id != id_2:  # 隣接ノードがnode_2ではないとき
                    message_product_1 *= node_1.receive_message[neigbor_node]

            # node_2について node_1以外からのメッセージの積を計算
            message_product_2 = np.ones(self.n_grad, dtype='float64')  # node_2の近傍からのメッセージの積(node_1からのメッセージは除く)
            for neigbor_node in node_2.receive_message.keys():
                if neigbor_node.id != id_1:  # 隣接ノードがnode_1ではないとき
                    message_product_2 *= node_2.receive_message[neigbor_node]

            # 潜在変数間の結合関数を取得
            F_12 = gaussian_inter_latent_function(alpha, self.n_grad)

            # メッセージの積から結合事後分布を計算
            for k in range(self.n_grad):
                F_12[k, :] *= message_product_1  # 行ベクトルにnode_1近傍のメッセージ積を掛ける
                F_12[:, k] *= message_product_2  # 列ベクトルにnode_2近傍のメッセージ積を掛ける

            post_joint_prob = 1 / np.sum(F_12) * F_12  # 正規化して結合事後分布を計算

            # 結合事後分布をegdeに保存
            edge.post_joint_prob = post_joint_prob


# Node作成
class Node:
    def __init__(self, node_id, n_grad):
        self.id = node_id
        self.n_grad = n_grad  # 画素の階調数
        self.neighbor_set = []  # 近傍ノードをもつリスト (観測ノードは含まない)
        self.receive_message = {}  # 近傍ノードから受信したメッセージを保存する辞書 (階調数ベクトルのメッセージが保存される) (観測ノードからの尤度も含む)
        self.post_marginal_prob = np.zeros(n_grad)  # nodeの周辺事後分布

    # 近傍ノードの追加
    def add_neighbor(self, node):
        self.neighbor_set.append(node)

    def calc_likelihood(self, beta, q_error, observed, option_model):
        """
        各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算し, receive_message[self]に登録
        :param beta: beta: ノイズ精度(分散の逆数)
        :param q_error: 潜在画素が異なる1つの画素へ遷移する確率
        :param observed: 観測画素:
        :param option_model: 尤度のモデルを選択
        """

        # 尤度のモデル選択
        if option_model == "sym":
            likelihood = self.sym_likelihood(q_error, observed)
        elif option_model == "gaussian":
            likelihood = self.gaussian_likelihood(beta, observed)

        self.receive_message[self] = likelihood  # 自分自身のノードのメッセージに尤度を登録

    # 各ノードの(ガウス)観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
    def gaussian_likelihood(self, beta, observed):
        """
         # 尤度(ガウスモデル) p(g_i|f_i) = N(g_i|f_i, 1/beta)
        :param beta: ノイズ精度(分散の逆数)
        :param observed: 観測画素
        :return: likelihood: 尤度関数 (n_grad次元ベクトル)
        """
        var_observed = observed * np.ones(self.n_grad)  # 観測画素 (階調数の長さに拡張)
        var_latent = np.linspace(0, self.n_grad - 1, self.n_grad)  # 潜在変数の候補 (0~階調数-1までの長さのベクトルを用意)
        likelihood = np.sqrt(beta) / np.sqrt(2 * np.pi) * np.exp(-beta / 2 * (var_observed - var_latent) ** 2)
        return likelihood

    # 各ノードの(K次元対称通信路)観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
    def sym_likelihood(self, q_error, observed):
        """
        尤度計算(対称通信路をn_grad次元に拡張したノイズモデル)
        # p(g|f) = q*(1-δ(g-f)) + (1-(n_grad-1)*q)*δ(g-f)  # g:観測(n_grad状態), f:潜在(n_grad状態)
        # 観測g_kがノイズにより潜在f_jに遷移する確率は q (状態 k,jによらない)
        # 観測gが潜在fと一致する確率は 1-(n_grad-1)*q
        :param q_error: 潜在画素が異なる1つの画素へ遷移する確率
        :param observed: 観測画素
        :return: likelihood: 尤度関数 (n_grad次元ベクトル)
        """
        likelihood = q_error * np.ones(self.n_grad)  # 潜在変数が観測と異なる場合の確率の計算
        likelihood[observed] = 1 - (self.n_grad - 1) * q_error  # 潜在変数が観測と一致する場合の確率の計算
        return likelihood

    # メッセージの初期化
    def initialize_message(self):
        for neighbor_node in self.neighbor_set:
            self.receive_message[neighbor_node] = np.ones(self.n_grad)  # メッセージを1で初期化

    # node から neigbor_target への送信メッセージを作成
    def send_message(self, neighbor_target, alpha):
        """
        # node から neigbor_target への送信メッセージを作成
        :param neighbor_target: メッセージ送信先のノnodeインスタンス
        :param alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
        :return: message: node から neigbor_target への送信メッセージ (階調数次元のベクトル)
        """
        message_product = np.ones(self.n_grad)  # メッセージの積: target以外からの近傍メッセージの積を計算するためのベクトル
        for neighbor_node in self.receive_message.keys():  # target以外のすべての近傍ノードを走査 (観測ノードもreceive_message[self]として含まれる)
            if neighbor_target != neighbor_node:  # target以外の近傍ノードからnodeに対してのメッセージをすべて掛け合わせる(観測ノードも含む)
                message_product *= self.receive_message[neighbor_node]  # 近傍ノードからの受信メッセージの積を計算

        # 自ノードjと近傍ノードiの結合を表す行列 (階調数, 階調数)
        F_ij = gaussian_inter_latent_function(alpha, self.n_grad)

        # 周辺化して送信メッセージを作成 (node から neigbor_target へのメッセージ)
        message_tmp = F_ij @ message_product
        message = 1 / np.sum(message_tmp) * message_tmp  # メッセージを正規化 [0, 1]

        return message

    def calc_post_marginal(self):
        """
        周辺事後分布 p(f_i|g_i)を計算して self.post_marginal_prob に保存
        """
        prob_tmp = np.ones(self.n_grad)  # 周辺分布の初期値

        # 近傍ノードからのメッセージの積から周辺分布を計算
        for rcv_msg in self.receive_message.values():
            prob_tmp *= rcv_msg  # メッセージの積
            prob = 1 / np.sum(prob_tmp) * prob_tmp  # 規格化

        # 周辺事後分布をnodeのメンバに登録
        self.post_marginal_prob = prob


# MRF作成
def generate_mrf_network(height, width, n_grad):
    """
    # MRF networkを作成 (ノードとエッジを登録)
    :param height: 画像の高さ
    :param width: 画像の幅
    :param n_grad: 画素の階調数
    :return: network: すべてのノードを含んだネットワーク全体のインスタンス
    """
    # MRF network 作成
    network = MRF(n_grad)  # MRF network インスタンス化
    network.n_grad = n_grad  # 階調数

    # ### nodeの登録 ### #
    for y in range(height):
        for x in range(width):
            node_id = width * y + x  # 横方向にidを振る
            node = Node(node_id, n_grad)  # node 作成
            network.nodes[node_id] = node  # MRFにnodeを追加

    # 各ノードの近傍ノードを登録
    # ####隣接ノードの方向##### #
    dy = [-1, 0, 0, 1]  # 基本的に4方向に隣接ノードを持つ
    dx = [0, -1, 1, 0]
    # ####上, 下, 右, 左 #### #

    for y in range(height):
        for x in range(width):
            node = network.nodes[width * y + x]

            # 4方向に隣接ノードがあるかを確認
            for k in range(4):
                if (y + dy[k] >= 0) and (y + dy[k] < height) \
                        and (x + dx[k] >= 0) and (x + dx[k] < width):
                    neighbor = network.nodes[width * (y + dy[k]) + x + dx[k]]  # networkから近傍ノードを取得
                    node.add_neighbor(neighbor)  # 各ノードに近傍ノードを登録

    # ### edgeの登録 ### #
    # (height-1)*(width-1)の範囲のedgeの登録
    for y in range(height - 1):
        for x in range(width - 1):
            id = y * width + x
            pair_id_1 = np.array([id, id + 1], dtype='int32')  # 右のノードとのedge
            pair_id_2 = np.array([id, id + width], dtype='int32')  # 下のノードとのedge
            edge_1 = Edge(pair_id_1, n_grad)  # edgeインスタンスの作成
            edge_2 = Edge(pair_id_2, n_grad)  # edgeインスタンスの作成

            # edgeインスタンスをMRFインスタンスに保存
            network.edges.append(edge_1)
            network.edges.append(edge_2)

    # 一番下の行の登録
    for x in range(width - 1):
        id = width * (height - 1) + x
        pair_id = np.array([id, id + 1], dtype='int32')  # 右のノードとのedge
        edge = Edge(pair_id, n_grad)  # edgeインスタンスの作成
        network.edges.append(edge)  # edgeインスタンスをMRFインスタンスに保存

    # 一番右の列の登録
    for y in range(height - 1):
        id = width * y + width - 1
        pair_id = np.array([id, id + width], dtype='int32')  # 下のノードとのedge
        edge = Edge(pair_id, n_grad)  # edgeインスタンスの作成
        network.edges.append(edge)  # edgeインスタンスをMRFインスタンスに保存

    return network


# 潜在変数間の結合を表す関数を計算
def gaussian_inter_latent_function(alpha, n_grad):
    """
    潜在変数間の結合を表す関数を計算
    f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
    :param alpha: 潜在変数間の結合度合いを示す変数
    :param n_grad: 階調数 (離散変数xの次元)
    :return: F_ij: 潜在変数間の結合関数行列 {F_i,j}_k,l = f_i,j(x_i^(k), x_j^(l))
    """
    x_ij_vec = np.linspace(0, n_grad - 1, n_grad)  # 階調数次元のベクトル
    F_ij = np.zeros((n_grad, n_grad))  # 潜在変数間の結合関数 {F_i,j}_k,l = f_i,j(x_i^(k), x_j^(l))

    for k in range(n_grad):
        for l in range(n_grad):
            F_ij[k, l] = np.exp(-alpha / 2 * (x_ij_vec[k] - x_ij_vec[l]) ** 2)  # 第iノードと第jノードの結合を表す関数

    return F_ij


def noise_add_gaussian(n_grad, beta, image_in):
    """
    ガウスノイズ: 入力画像image_inに対してノイズを加えてimage_outとして出力
        # p(g|f) = c * exp[-beta/2 * (g - f)^2]  # g:観測(K状態), f:潜在(K状態)
    :param n_grad: 状態数 (画素の階調数)
    :param beta: ノイズ精度
    :param iamge_in: 入力画像
    :return:image_out: 出力画像
            noise_mat: ノイズ行列 (画像サイズ)
    """
    # 画像情報
    height = image_in.shape[0]
    width = image_in.shape[1]

    image_out = np.zeros((height, width), dtype="uint8")  # 出力画像
    noise_mat = np.zeros((height, width), dtype="uint8")  # ノイズ単体の画像
    for y in range(height):
        for x in range(width):
            # i.i.d.ガウスノイズの生成
            noise = np.random.normal(
                loc=0,  # 平均
                scale=np.sqrt(1 / beta),  # 標準偏差
                size=1,  # 出力配列のサイズ
            )
            out = image_in[y, x] + noise  # 出力値

            if out > (n_grad - 1):
                out = n_grad - 1
            elif out < 0:
                out = 0

            image_out[y, x] = np.round(out)
            noise_mat[y, x] = np.round(np.abs(noise))  # ノイズの保存(debug用)

    return image_out, noise_mat


# 対称通信路をK次元に拡張したノイズモデル
def noise_add_sym(K, q_error, image_in):
    """
    対称通信路をK次元に拡張したノイズモデル: 入力画像image_inに対してノイズを加えてimage_outとして出力
        # p(g|f) = q*(1-δ(g-f)) + (1-(K-1)*q)*δ(g-f)  # g:観測(K状態), f:潜在(K状態)
        # 観測g_kがノイズにより潜在f_jに遷移する確率はq (k,jによらない)
        # 観測gが潜在fと一致する確率は1-(K-1)*q
    :param K: 状態数 (画素の階調数)
    :param q_error: 潜在画素が異なる1つの画素へ遷移する確率
    :param iamge_in: 入力画像
    :return:image_out: 出力画像
            noise_mat: ノイズ行列 (画像サイズ)
    """
    p_error = (K - 1) * q_error  # 潜在画素が異なる(どこかの)画素へ遷移する確率

    # エラー処理
    if p_error >= 1:
        print("p_error is should be smaller than 1")
        print("## p_error = (K-1) * q_error = ", p_error, " > 1")
        print("## q_error=", q_error, ", K=", K)
        sys.exit()

    # 確率p_errorでflagが1になる画素数サイズの配列を作成
    error_flags = np.random.binomial(n=1, p=p_error, size=image_in.shape)

    # パラメータ取得
    height = image_in.shape[0]
    width = image_in.shape[1]
    image_out = image_in.copy()  # 出力画像
    noise_mat = np.zeros(image_in.shape, dtype='int32')  # ノイズ行列 debug用

    for y in range(height):
        for x in range(width):
            if error_flags[y, x] == 1:  # 誤りの場合
                image_in_value = image_in[y, x]
                noise_arr = np.random.randint(low=1, high=K, size=1, dtype='int32')  # 1以上K未満の整数乱数の配列作成
                noise = noise_arr[0]
                noise_mat[y, x] = noise  # debug用

                if (image_in_value + noise) >= K:  # ノイズを加えると階調数Kを超えてしまう場合
                    image_out[y, x] = image_in_value + noise - K  # K-1から超過した部分を先頭から数えたインデックスに遷移させる
                else:
                    image_out[y, x] = image_in_value + noise

    return image_out, noise_mat
