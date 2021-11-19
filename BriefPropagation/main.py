import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.BriefPropagation.func_BriefPropagation import *
from src.BriefPropagation.func_GaussianBayseInference import *

##############
# id = 30
# height = 16
# width = 16
# ###
#
# _id = id + 1  # 1から数えたindex
# if _id % width == 0:
#     y = np.int(_id / width)
#     x = width
# else:
#     y = np.int(_id / width) + 1
#     x = _id % width
#
# zzz = np.array([y, x], dtype="int32") - 1
# test = 0
#############

# ######### main ########## #
# 画像の読み込み
image_origin = cv2.imread("./image_folder/Mandrill_8bit_16x16.png", 0)  # Lena, Mandrill 256, 64, 32, 16
height, width = image_origin.shape  # height:y方向, width：x方向

# 量子化のスケールダウン（2bitに変換）（2bitの場合 {0,1,2,3}の4階調）
n_bit = 8
n_grad = 2 ** n_bit  # 階調数
image_in = (image_origin * ((2 ** n_bit - 1) / np.max(image_origin))).astype('uint8')

# パラメータ
q_error_true = 0.05 / (n_grad - 1)  # 元画素が異なる一つの画素へ遷移する確率 (真値)
q_error = 0.1  # n_grad次元の対称通信路ノイズモデルでの, 元画素が異なる一つの画素へ遷移する確率 (誤り率は(1-n_grad)*q_error)
sigma_noise = (n_grad - 1) / 12  # ノイズの標準偏差
beta_true = 1 / sigma_noise ** 2  # 観測ノイズの精度(真値)
beta = 1  # 観測ノイズの精度(EM初期値)
alpha = 1  # 潜在変数間の結合

N_itr_BP = 10  # BPイタレーション回数
N_itr_EM = 1  # EMイタレーション回数
threshold_EM = 1e-2  # EMアルゴリズム終了条件用の閾値 (パラメータの変化率)

# option
option_noise_model = "gaussian"
option_model = "gaussian"  # "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q), "gaussian":ガウス分布(精度beta)

# 画像にノイズを加える
if option_noise_model == "sym":
    image_noise, noise_mat = noise_add_sym(n_grad, q_error_true, image_in)
elif option_noise_model == "gaussian":
    image_noise, noise_mat = noise_add_gaussian(n_grad, beta_true, image_in)

# MRF networkの作成
network = generate_mrf_network(height, width, n_grad)

# Belief Propagation (BP)
print("\n##################################################")
print("Start BP")
image_out_bp, image_out_gabp = \
    network.belief_propagation(N_itr_BP, N_itr_EM, alpha, beta, q_error,
                               image_noise, threshold_EM, option_model)
print("##################################################")

# ベイズ推論(BPを利用せずに解析解を求める)
print("\n##################################################")
print("Start Bayseian Inference in Gaussian Model")
image_out_anl = calc_gaussian_post_prob_analytical(alpha, beta, n_grad, image_noise, N_itr_EM, threshold_EM)
print("##################################################")

#################
# ログ出力
# print("\n", "alpha", np.round(alpha, decimals=3), ", alpha_new", np.round(alpha_new, decimals=3))
# print("beta", np.round(beta, decimals=3), ", beta_new", np.round(beta_new, decimals=3))
# print("q_error_true", np.round(q_error_true, decimals=3), ", q_error_new", np.round(q_error_new, decimals=3),
#       ', q_error_0', np.round(q_error, decimals=3))

print("\nbeta_true", np.round(beta_true, decimals=5))

noise_abs = np.sum(np.abs(image_noise.astype('float64') - image_in.astype('float64')))
bp_error = np.sum(np.abs(image_out_bp.astype('float64') - image_in.astype('float64')))
gabp_error = np.sum(np.abs(image_out_gabp.astype('float64') - image_in.astype('float64')))
anl_error = np.sum(np.abs(image_out_anl.astype('float64') - image_in.astype('float64')))

print("\nnoise_abs", noise_abs)
print("bp_error", bp_error)
print("gabp_error", gabp_error)
print("anl_error", anl_error)

# あとでSNR測定
#################

#################
# plot
# ### 元画像
# 加工前
plt.figure()
plt.subplot(1, 3, 1)
plt.gray()
plt.imshow(image_origin)
# ノイズ
plt.subplot(1, 3, 2)
plt.gray()
plt.imshow(image_in)
plt.title("image_in")
# 量子化後
plt.subplot(1, 3, 3)
plt.gray()
plt.imshow(noise_mat)
plt.title("noise_mat")

# ### 手法比較
# ノイズ入力後
plt.figure()
plt.subplot(2, 2, 1)
plt.gray()
plt.imshow(image_noise)
plt.title("image_noise")
# 離散BP
plt.subplot(2, 2, 2)
plt.gray()
plt.imshow(image_out_bp)
plt.title("image_out(BP)")
# ガウスBP
plt.subplot(2, 2, 3)
plt.gray()
plt.imshow(image_out_gabp)
plt.title("image_out(GaBP)")
# 解析解
plt.subplot(2, 2, 4)
plt.gray()
plt.imshow(image_out_anl)
plt.title("image_out(Analytical)")

plt.show()
#################

# ######### main end ########## #
