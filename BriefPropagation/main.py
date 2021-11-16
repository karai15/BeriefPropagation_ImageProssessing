import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.BriefPropagation.func_BriefPropagation import *


# ######### main ########## #
# 画像の読み込み
image_origin = cv2.imread("./image_folder/Lena_8bit_64x64.jpg", 0)  # 256, 64, 32, 16
height, width = image_origin.shape  # height:y方向, width：x方向

# 量子化のスケールダウン（2bitに変換）（{0,1,2,3}の4階調）
n_bit = 4
n_grad = 2 ** n_bit  # 階調数
image_in = (image_origin * ((2 ** n_bit - 1) / np.max(image_origin))).astype('uint8')

# パラメータ
q_error_true = 0.05 / (n_grad - 1)  # 元画素が異なる一つの画素へ遷移する確率 (真値)
q_error = 0.1  # n_grad次元の対称通信路ノイズモデルでの, 元画素が異なる一つの画素へ遷移する確率 (誤り率は(1-n_grad)*q_error)
beta_true = 5  # 観測ノイズの精度(真値)
beta = 5  # 観測ノイズの精度(EM初期値)
alpha = 1  # 潜在変数間の結合

N_itr_BP = 1  # BPイタレーション回数
N_itr_EM = 1  # EMイタレーション回数
threshold_EM = 1e-3  # EMアルゴリズム終了条件用の閾値 (パラメータの変化率)
option_model = "sym"  # "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q), "gaussian":ガウス分布(精度beta)

# 画像にノイズを加える
# if option_model == "sym":
#     image_noise, noise_mat = noise_add_sym(n_grad, q_error_true, image_in)
# elif option_model == "gaussian":
#     image_noise, noise_mat = noise_add_gaussian(n_grad, beta_true, image_in)

image_noise, noise_mat = noise_add_gaussian(n_grad, beta_true, image_in)


# MRF networkの作成
network = generate_mrf_network(height, width, n_grad)

# Belief Propagation (BP)
alpha_new, beta_new, q_error_new, image_out_bp = \
    network.belief_propagation(N_itr_BP, N_itr_EM, alpha, beta, q_error,
                               image_noise, threshold_EM, option_model)

# ベイズ推定(BPを利用せずに一発で解析解を求める)
print("aaa")
image_out_anl = calc_gaussian_post_prob_analytical(alpha, beta, n_grad, image_noise)
print("bbb")

#################
# ログ出力
print("\n", "alpha", np.round(alpha, decimals=3), ", alpha_new", np.round(alpha_new, decimals=3))
print("beta", np.round(beta, decimals=3), ", beta_new", np.round(beta_new, decimals=3))
print("q_error_true", np.round(q_error_true, decimals=3), ", q_error_new", np.round(q_error_new, decimals=3),
      ', q_error_0', np.round(q_error, decimals=3))

noise_abs = np.sum(np.abs(noise_mat))
bp_error = np.sum(np.abs(image_out_bp - image_in))
anl_error = np.sum(np.abs(image_out_anl - image_in))

print("\n", "noise_abs", noise_abs)
print("bp_error", bp_error)
print("anl_error", anl_error)



# あとでSNR測定
#################

#################
# plot
# 加工前
plt.figure()
plt.gray()
plt.imshow(image_origin)

# ノイズ
plt.figure()
plt.gray()
plt.imshow(noise_mat)
plt.title("noise_mat")

# 量子化後
plt.figure()
plt.subplot(2, 2, 1)
plt.gray()
plt.imshow(image_in)
plt.title("image_in")
# ノイズ入力後
plt.subplot(2, 2, 2)
plt.gray()
plt.imshow(image_noise)
plt.title("image_noise")
# BP
plt.subplot(2, 2, 3)
plt.gray()
plt.imshow(image_out_bp)
plt.title("image_out(BP)")
# 解析解
plt.subplot(2, 2, 4)
plt.gray()
plt.imshow(image_out_anl)
plt.title("image_out(Analytical)")

plt.show()
#################

# ######### main end ########## #
