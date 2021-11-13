import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.BriefPropagation.func_BriefPropagation import *

############################
############################


# ######### main ########## #
# 画像の読み込み
image_origin = cv2.imread("./image_folder/Lena_8bit_16x16.jpg", 0)  # 256, 64, 32, 16
height, width = image_origin.shape  # height:y方向, width：x方向

# 量子化のスケールダウン（2bitに変換）（{0,1,2,3}の4階調）
n_bit = 2
n_grad = 2 ** n_bit  # 階調数
image_in = (image_origin * ((2 ** n_bit - 1) / np.max(image_origin))).astype('uint8')

# パラメータ
q_error = 0.01 / (n_grad - 1)  # n_grad次元の対称通信路ノイズモデルでの, 元画素が異なる一つの画素へ遷移する確率 (誤り率は(1-n_grad)*q_error)
alpha = 5.0  # 潜在変数間の結合
beta = 1  # 観測ノイズの精度
N_itr = 3  # BPイタレーション回数
option_likelihood = "sym"  # "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q), "gaussian":ガウス分布(精度beta)

# 画像にノイズを加える
image_noise, noise_mat = noise_add_sym(n_grad, q_error, image_in)

# MRF networkの作成
network = generate_mrf_network(height, width, n_grad)

# # 尤度の計算
# # 各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
# for y in range(height):
#     for x in range(width):
#         node_id = width * y + x
#         node = network.nodes[node_id]  # ノードの取り出し
#         node.calc_likelihood(beta, q_error, image_noise[y, x], option_likelihood)  # K次元対称通信路の尤度計算

###########################
# # debug
# likelihood_mat = np.zeros((n_grad, height * width))
# for node in network.nodes.values():
#     likelihood = node.receive_message[node]
#     likelihood_mat[:, node.id] = likelihood[:]
# test = 1
###########################

# BP実行
# network.belief_propagation(N_itr, alpha)
network.belief_propagation(N_itr, alpha, beta, q_error, image_noise, option_likelihood)


# 周辺分布から画像化
image_out = np.zeros((width, height), dtype='uint8')  # 出力画像

for y in range(height):
    for x in range(width):
        node = network.nodes[y * width + x]  # nodeインスタンス
        post_marginal_prob = node.post_marginal_prob  # 周辺事後分布p(f_i|g_all)

        # 出力画像の保存
        image_out[y, x] = np.argmax(post_marginal_prob)  # 事後確率が最大となる値を出力画像として保存

#################
# plot
plt.figure()
plt.gray()
plt.imshow(image_origin)

plt.figure()
plt.subplot(2, 2, 1)
plt.gray()
plt.imshow(image_in)
plt.title("image_in")

plt.subplot(2, 2, 2)
plt.gray()
plt.imshow(noise_mat)
plt.title("noise_mat")

plt.subplot(2, 2, 3)
plt.gray()
plt.imshow(image_noise)
plt.title("image_noise")

plt.subplot(2, 2, 4)
plt.gray()
plt.imshow(image_out)
plt.title("image_out(BP)")

plt.show()
#################

# ######### main end ########## #
