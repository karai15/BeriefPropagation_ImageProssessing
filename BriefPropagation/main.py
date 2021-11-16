import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.BriefPropagation.func_BriefPropagation import *

############################
# ベイズ推論(BPを利用せずに一発で解析解を求める)

# 画像の読み込み
image = cv2.imread("./image_folder/Lena_8bit_16x16.jpg", 0)  # 256, 64, 32, 16
height, width = image.shape  # height:y方向, width：x方向

# ####### 精度行列計算#
Cov_inv_prior = calc_mrf_cov_inv_mat(height, width)
u, s, vh = np.linalg.svd(Cov_inv_prior)  # 特異値分解(debug用)
###################

# ####### 画像をベクトルに変換
# 画像を行列からベクトルに変換
image_vec = np.zeros(height * width, dtype="float64")
for y in range(height):
    image_vec[(y * width):((y + 1) * width)] = image[y, :]  # 画像の各行を1列に並べる
##########################

# ########### 事後分布の計算
# 引数
alpha = 1
beta = 1
#
Cov_inv_post = alpha * Cov_inv_prior + beta * np.eye(height * width)  # 事後分布の精度行列
Cov_post = np.linalg.inv(Cov_inv_post)  # 事後分布の共分散行列
mu_post = beta * Cov_post @ image_vec  # 事後分布の平均

u, s, vh = np.linalg.svd(Cov_inv_post)  # 特異値分解(debug用)

# 誤差の測定
err_vec = np.abs(image_vec - mu_post)
err_sum = np.sum(err_vec)

##########################

# ベクトルから画像に変換


test = -1

############################


# ######### main ########## #
# 画像の読み込み
image_origin = cv2.imread("./image_folder/Lena_8bit_16x16.jpg", 0)  # 256, 64, 32, 16
height, width = image_origin.shape  # height:y方向, width：x方向

# 量子化のスケールダウン（2bitに変換）（{0,1,2,3}の4階調）
n_bit = 4
n_grad = 2 ** n_bit  # 階調数
image_in = (image_origin * ((2 ** n_bit - 1) / np.max(image_origin))).astype('uint8')

# パラメータ
q_error_true = 0.05 / (n_grad - 1)  # 元画素が異なる一つの画素へ遷移する確率 (真値)
q_error = 0.1  # n_grad次元の対称通信路ノイズモデルでの, 元画素が異なる一つの画素へ遷移する確率 (誤り率は(1-n_grad)*q_error)
beta_true = 0.5  # 観測ノイズの精度(真値)
beta = 0.4  # 観測ノイズの精度(EM初期値)
alpha = 0.4  # 潜在変数間の結合

N_itr_BP = 3  # BPイタレーション回数
N_itr_EM = 50  # EMイタレーション回数
threshold_EM = 1e-3  # EMアルゴリズム終了条件用の閾値 (パラメータの変化率)
option_model = "gaussian"  # "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q), "gaussian":ガウス分布(精度beta)

# 画像にノイズを加える
if option_model == "sym":
    image_noise, noise_mat = noise_add_sym(n_grad, q_error_true, image_in)
elif option_model == "gaussian":
    image_noise, noise_mat = noise_add_gaussian(n_grad, beta_true, image_in)

# image_noise, noise_mat = noise_add_sym(n_grad, q_error_true, image_in)


###################
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.gray()
# plt.imshow(image_in)
# plt.title("image_in")
#
# plt.subplot(1, 3, 2)
# plt.gray()
# plt.imshow(noise_mat)
# plt.title("noise_mat")
#
# plt.subplot(1, 3, 3)
# plt.gray()
# plt.imshow(image_noise)
# plt.title("image_noise")
# plt.show()
###################


# MRF networkの作成
network = generate_mrf_network(height, width, n_grad)

# BP実行
# network.belief_propagation(N_itr_BP, alpha)
alpha_new, beta_new, q_error_new = network.belief_propagation(N_itr_BP, N_itr_EM, alpha, beta, q_error, image_noise,
                                                              threshold_EM, option_model)
print("\n", "alpha", np.round(alpha, decimals=3), ", alpha_new", np.round(alpha_new, decimals=3))
print("beta", np.round(beta, decimals=3), ", beta_new", np.round(beta_new, decimals=3))
print("q_error_true", np.round(q_error_true, decimals=3), ", q_error_new", np.round(q_error_new, decimals=3),
      ', q_error_0', np.round(q_error, decimals=3))

# 周辺事後分布から画像化
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
