import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from src.BriefPropagation_Qiita_code.func_BpMrf import *

def main():
    # 使用データ
    image = cv2.imread("./image_folder/Lenna.png", 0)
    # binary = image > threshold_otsu(image).astype(np.int)
    binary = image
    noise = addNoise(binary)


    # MRF構築
    network = generateBeliefNetwork(image)

    # 観測値（画素値）から尤度を作成
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            node = network.getNode(image.shape[1] * i + j)
            node.calcLikelihood(noise[i, j])

    # 確率伝播法を行う
    network.beliefPropagation()

    # 周辺分布は[0の確率,1の確率]の順番
    # もし1の確率が大きければoutputの画素値を1に変える
    output = np.zeros(noise.shape)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            node = network.getNode(output.shape[1] * i + j)
            prob = node.prob
            if prob[1] > prob[0]:
                output[i, j] = 1

    # 結果表示
    plt.gray()
    plt.subplot(121)
    plt.imshow(noise)
    plt.subplot(122)
    plt.imshow(output)
    plt.show()

main()

