import numpy as np


# MRF作成 (ノードの集合を保存)
class MRF:
    def __init__(self):
        self.nodes = {}  # MRF上のノード (key:id, value:node)

    """
    #### memo ####
    # MRFにnodeを追加するとき
    mrf.nodes[node.id] = node

    # MRFからidに対応するnodeを返してほしいとき
    node = mrf.nodes[node.id]
    """

    # 確率伝搬BPを実行
    def belief_propagation(self, N_itr, alpha):
        """
        確率伝搬BPを実行 (動機スケジューリング)
        :param N_itr: BPイタレーション
        :param alpha: 潜在変数間の結合係数
        :return: メッセージの更新を行い、各ノードインスタンスにメッセージと周辺事後分布
        """

        # 各ノードのメッセージ初期化
        for node in self.nodes.values():
            node.initialize_message()

        # BPイタレーション開始
        for itr in range(N_itr):
            print("BP iteration: ", itr)  # イタレーションの出力

            # 各ノードが近傍ノードに対してメッセージを送信 (同期スケジューリング)
            for node in self.nodes.values():
                for neighbor_target in node.neighbor_set:
                    # nodeからneighbor_targetへの送信メッセージをneighborの受信メッセージとして保存
                    neighbor_target.receive_message[node] = node.send_message(neighbor_target, alpha)

        # 各ノードの周辺分布を計算
        for node in self.nodes.values():
            node.marginal()


# Node作成
class Node:
    def __init__(self, node_id, n_grad):
        self.id = node_id
        self.n_grad = n_grad  # 画素の階調数
        self.neighbor_set = []  # 近傍ノードをもつリスト (観測ノードは含まない)
        self.receive_message = {}  # 近傍ノードから受信したメッセージを保存する辞書 (階調数ベクトルのメッセージが保存される) (観測ノードからの尤度も含む)
        self.prob_marginal = np.zeros(4)  # nodeの周辺事後分布

    # 近傍ノードの追加
    def add_neighbor(self, node):
        self.neighbor_set.append(node)

    def calc_likelihood(self, beta, q_error, observed, option_likelihood):
        """
        各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算し, receive_message[self]に登録
        :param beta: beta: ノイズ精度(分散の逆数)
        :param q_error: 潜在画素が異なる1つの画素へ遷移する確率
        :param observed: 観測画素:
        :param option_likelihood: 尤度のモデルを選択
        """

        # 尤度のモデル選択
        if option_likelihood == "sym":
            likelihood = self.sym_likelihood(q_error, observed)
        elif option_likelihood == "gaussian":
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
        var_latent = np.linspace(0, self.n_grad - 1, self.n_grad)  # 潜在変数の候補 (0~階調数までの長さのベクトルを用意)
        likelihood = np.exp(-beta / 2 * (var_observed - var_latent) ** 2)
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

    def marginal(self):
        prob_tmp = np.ones(self.n_grad)  # 周辺分布の初期値

        # 近傍ノードからのメッセージの積から周辺分布を計算
        for rcv_msg in self.receive_message.values():
            prob_tmp *= rcv_msg  # メッセージの積
            prob = 1 / np.sum(prob_tmp) * prob_tmp  # 規格化

        # 周辺分布をnodeのメンバに登録
        self.prob_marginal = prob


# MRF作成
def generate_mrf_Network(height, width, n_grad):
    """
    # MRF networkを作成
    :param height: 画像の高さ
    :param width: 画像の幅
    :param n_grad: 画素の階調数
    :return: network: すべてのノードを含んだネットワーク全体のインスタンス
    """
    # MRF network 作成
    network = MRF()  # MRF network インスタンス化
    network.n_grad = n_grad  # 階調数
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
