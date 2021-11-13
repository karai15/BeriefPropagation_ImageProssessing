import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from src.BriefPropagation.func_BriefPropagation import *


# MRF作成 (ノードの集合を保存)
class MRF:
    def __init__(self):
        self.nodes = {}  # MRF上のノード (key:id, value:node)
        self.n_grad = 0  # 階調数

    """
    #### memo ####
    # MRFにnodeを追加するとき
    mrf.nodes[node.id] = node
    
    # MRFからidに対応するnodeを返してほしいとき
    node = mrf.nodes[node.id]
    """

    def belief_propagation(self, N_itr):
        # 各ノードのメッセージ初期化
        for node in self.nodes.values():
            node.initialize_message(self.n_grad)

        # BPイタレーション開始
        for itr in range(N_itr):
            # 各ノードが近傍ノードに対してメッセージを送信 (同期スケジューリング)
            for node in self.nodes.values():
                for neighbor_target in node.neighbor_set:
                    # nodeからneighbor_targetへの送信メッセージをneighborの受信メッセージとして保存
                    neighbor_target.receive_message[node] = node.send_message(neighbor_target, alpha, n_grad)

        # 各ノードの周辺分布を計算
        for node in self.nodes.values():
            node.marginal(self.n_grad)


# Node作成
class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.neighbor_set = []  # 近傍ノードをもつリスト (観測ノードは含まない)
        self.receive_message = {}  # 近傍ノードから受信したメッセージを保存する辞書 (階調数ベクトルのメッセージが保存される) (観測ノードからの尤度も含む)
        self.prob_marginal = np.zeros(4)  # nodeの周辺事後分布

    # 近傍ノードの追加
    def add_neighbor(self, node):
        self.neighbor_set.append(node)

    # 各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
    def calc_likelihood(self, n_grad, beta, observed):
        """
         # 尤度 p(g_i|f_i) = N(g_i|f_i, 1/beta) を receive_message[self]に登録
        :param n_grad: 画素の階調度
        :param beta: ノイズ精度(分散の逆数)
        :param observed: 観測画素
        :return:
        """
        var_observed = observed * np.ones(n_grad)  # 観測画素 (階調数の長さに拡張)
        var_latent = np.linspace(0, n_grad - 1, n_grad)  # 潜在変数の候補 (0~階調数までの長さのベクトルを用意)
        likelihood = np.exp(-beta / 2 * (var_observed - var_latent) ** 2)
        self.receive_message[self] = likelihood  # 自分自身のノードのメッセージに尤度を登録

    # メッセージの初期化
    def initialize_message(self, n_grad):
        for neighbor_node in self.neighbor_set:
            self.receive_message[neighbor_node] = np.ones(n_grad)  # メッセージを1で初期化

    # node から neigbor_target への送信メッセージを作成
    def send_message(self, neighbor_target, alpha, n_grad):
        """
        # node から neigbor_target への送信メッセージを作成
        :param neighbor_target: メッセージ送信先のノnodeインスタンス
        :param alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
        :param n_grad: 画素の階調数
        :return: message: node から neigbor_target への送信メッセージ (階調数次元のベクトル)
        """
        message_product = np.ones(n_grad)  # メッセージの積: target以外からの近傍メッセージの積を計算するためのベクトル
        for neighbor_node in node.receive_message.keys():  # target以外のすべての近傍ノードを走査 (観測ノードもreceive_message[self]として含まれる)
            if neighbor_target != neighbor_node:  # target以外の近傍ノードからnodeに対してのメッセージをすべて掛け合わせる(観測ノードも含む)
                message_product *= node.receive_message[neighbor_node]  # 近傍ノードからの受信メッセージの積を計算

        # 自ノードjと近傍ノードiの結合を表す行列 (階調数, 階調数)
        F_ij = gaussian_inter_latent_function(alpha, n_grad)

        # 周辺化して送信メッセージを作成 (node から neigbor_target へのメッセージ)
        message_tmp = F_ij @ message_product
        message = 1 / np.sum(message_tmp) * message_tmp  # メッセージを正規化 [0, 1]

        return message

    def marginal(self, n_grad):
        prob_tmp = np.ones(n_grad)  # 周辺分布の初期値

        # 近傍ノードからのメッセージの積から周辺分布を計算
        for rcv_msg in node.receive_message.values():
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
            node = Node(node_id)  # node 作成
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


# ######### main ########## #
# 画像の読み込み
image_input = cv2.imread("./image_folder/Lena_8bit_16x16.jpg", 0)
height, width = image_input.shape  # height:y方向, width：x方向

# 量子化のスケールダウン（2bitに変換）（{0,1,2,3}の4階調）
n_bit = 2
n_grad = 2 ** n_bit  # 階調数
image = (image_input * ((2 ** n_bit - 1) / np.max(image_input))).astype('uint8')

# パラメータ
alpha = 1  # 潜在変数間の結合
beta = 1  # 観測ノイズの精度
N_itr = 2  # BPイタレーション回数

##############################
# MRF networkの作成
network = generate_mrf_Network(height, width, n_grad)

# 尤度の計算
# 各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
for y in range(height):
    for x in range(width):
        node_id = width * y + x
        node = network.nodes[node_id]  # ノードの取り出し
        node.calc_likelihood(n_grad, beta, image[y, x])

# BP実行
network.belief_propagation(N_itr)

# 周辺分布から画像化
image_out = np.zeros((width, height), dtype='uint8')  # 出力画像
for y in range(height):
    for x in range(width):
        node = network.nodes[y * width + x]  # nodeインスタンス
        prob = node.prob_marginal  # 周辺事後分布

        # 出力画像の保存
        image_out[y, x] = np.argmax(prob)  # 確率が最大となる状態を出力画像として保存

test = 1

############################################

# plot
# plt.gray()
# plt.imshow(image)
# plt.show()


# ######### main end ########## #


# test
# plt.gray()
# plt.imshow(image)
# plt.show()
