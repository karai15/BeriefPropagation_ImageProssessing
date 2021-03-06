import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.BriefPropagation.func_BriefPropagation import *
from src.BriefPropagation.func_GaussianBayseInference import *

# ######### main ########## #
# 画像の読み込み
image_origin = cv2.imread("./image_folder/Mandrill_8bit_32x32.png", 0)  # Lena, Mandrill 256, 64, 32, 16
height, width = image_origin.shape  # height:y方向, width：x方向

# 量子化のスケールダウン（2bitに変換）（2bitの場合 {0,1,2,3}の4階調）
n_bit = 4
n_grad = 2 ** n_bit  # 階調数
image_in = (image_origin * ((2 ** n_bit - 1) / np.max(image_origin))).astype('uint8')

# パラメータ
q_error_true = 0.05 / (n_grad - 1)  # 元画素が異なる一つの画素へ遷移する確率 (真値)
q_error = 0.1  # n_grad次元の対称通信路ノイズモデルでの, 元画素が異なる一つの画素へ遷移する確率 (誤り率は(1-n_grad)*q_error)
sigma_noise = (n_grad - 1) / 10  # ノイズの標準偏差
beta_true = 1 / sigma_noise ** 2  # 観測ノイズの精度(真値)
beta = beta_true  # 観測ノイズの精度(EM初期値)
alpha = 0.1  # 潜在変数間の結合

N_itr_BP = 100  # BPイタレーション回数
N_itr_EM = 100  # EMイタレーション回数
threshold_EM = 1e-2  # EMアルゴリズム終了条件用の閾値 (パラメータの変化率)
threshold_BP = 1e-5  # BP終了条件 (メッセージの変化)

# option
# "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q),
# "gaussian":ガウス分布(精度beta)
option_noise_model = "gaussian"  # "sym" or "gaussian"
option_model = "gaussian"  # "sym" or "gaussian" or "sym+gaussian"
option_gauss_analytical = "true"  # "true" or "false" (ガウスの解析解を求めるか)

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
                               image_noise, threshold_EM, threshold_BP, option_model)
print("##################################################")

# ベイズ推論(BPを利用せずに解析解を求める)
print("\n##################################################")
print("Start Bayseian Inference in Gaussian Model")
if option_gauss_analytical == "true":  # ガウスの解析解
    image_out_anl = calc_gaussian_post_prob_analytical(alpha, beta, n_grad, image_noise, N_itr_EM, threshold_EM)
elif option_gauss_analytical == "false":  # ガウスの解析解無し
    image_out_anl = np.zeros(image_noise.shape, 'uint8')
print("##################################################")

#################
# ログ出力
noise_abs = np.sum(np.abs(image_noise.astype('float64') - image_in.astype('float64')))
bp_error = np.sum(np.abs(image_out_bp.astype('float64') - image_in.astype('float64')))
gabp_error = np.sum(np.abs(image_out_gabp.astype('float64') - image_in.astype('float64')))
anl_error = np.sum(np.abs(image_out_anl.astype('float64') - image_in.astype('float64')))

test_mat = image_noise - image_out_gabp
test_max = np.max(test_mat)
test_min = np.min(test_mat)
test_sum = np.sum(test_mat)


print("\nnoise_error", noise_abs)
print("sym_bp_error", bp_error)
print("ga_bp_error", gabp_error)
print("ga_anlytical_error", anl_error)

print("\nbeta_true", np.round(beta_true, decimals=5))
print("q_error_true", np.round(q_error_true, decimals=5))
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
plt.title("BP")
# ガウスBP
plt.subplot(2, 2, 3)
plt.gray()
plt.imshow(image_out_gabp)
plt.title("GaBP")
# 解析解
plt.subplot(2, 2, 4)
plt.gray()
plt.imshow(image_out_anl)
plt.title("Ga_Analytical")

plt.show()
#################

# ######### main end ########## #
