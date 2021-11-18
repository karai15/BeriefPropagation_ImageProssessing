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
            # メッセージの更新を行い、各ノードインスタンスにメッセージと周辺事後分布を保存
        :param N_itr_BP: BPイタレーション
        :param N_itr_EM: EMイタレーション
        :param alpha: 潜在変数間の結合係数
        :param threshold_EM: EMアルゴリズム終了条件用の閾値 (パラメータの変化率)
        :param option_model: モデル選択オプション "sym":n_grad次元対称通信路(誤り率(n_grad-1)*q), "gaussian":ガウス分布(精度beta)
        :return: image_out_bp: (離散BP) 出力画像
                image_out_gabp: (ガウスBP) 出力画像
        """

        # ガウスBPと離散BPの初期値をそろえる
        alpha_gauss = alpha
        beta_gauss = beta

        # 画像情報取得
        height = image_noise.shape[0]
        width = image_noise.shape[1]

        # 各ノードのメッセージ初期化
        for node in self.nodes.values():
            # if option_model == "":
            node.initialize_message()  # 離散BP
            node.initialize_message_gauss()  # ガウスBP

        # ### EM loop start ### #
        for itr_em in range(N_itr_EM):

            # ログ表示 (更新パラメータ)
            print("itr_em:", itr_em + 1,
                  ", alpha:", np.round(alpha, decimals=5),
                  ", alpha_gauss:", np.round(alpha_gauss, decimals=5),
                  ", q_error:", np.round(q_error, decimals=5),
                  ", beta_gauss", np.round(beta_gauss, decimals=5))

            # 尤度の計算
            # 各ノードの観測確率モデル(尤度 p(g_i|f_i) g_i:観測, f_i:潜在変数)を計算
            for y in range(height):
                for x in range(width):
                    node = self.nodes[width * y + x]  # ノードの取り出し
                    node.calc_likelihood(q_error, image_noise[y, x])  # 離散BP (ガウスBPは必要なし)

            # BPイタレーション開始
            for itr_bp in range(N_itr_BP):
                # print("EM iteration:", itr_em, " BP iteration:", itr_bp)  # イタレーションの出力
                # 各ノードが近傍ノードに対してメッセージを送信 (同期スケジューリング) 周囲4方向のメッセージを更新
                for node in self.nodes.values():
                    y, x = id2xy(height, width, node.id)  # idから(x, y)に変換
                    observed = image_noise[y, x]  # nodeに対応する観測画素
                    for neighbor_target in node.neighbor_set:  # nodeからneighbor_targetへの送信メッセージをneighborの受信メッセージとして保存
                        # 離散BP
                        neighbor_target.receive_message[node] = node.send_message(neighbor_target, alpha)
                        # ガウスBP (ガウス型メッセージのλ(精度)とμ(平均)を送信) targetへのメッセージ [λ(精度), μ(平均)]
                        neighbor_target.receive_message_gauss[node] = node.send_message_gauss(neighbor_target, observed,
                                                                                              alpha_gauss, beta_gauss)

            # 各ノードの周辺事後分布を計算 p(f_i|g_all)
            for node in self.nodes.values():
                y, x = id2xy(height, width, node.id)  # idから(x, y)に変換
                node.calc_post_marginal()  # 離散BP
                node.calc_post_marginal_gauss(alpha_gauss, beta_gauss, image_noise[y, x])  # ガウスBP

            # ノード間の結合事後分布を計算 p(f_i,f_j|g_all) (edge.post_joint_probに保存)
            self.calc_post_joint(alpha)

            # ### パラメータの更新 ### #
            # 離散BP
            q_error_new = self.estimate_q_error(image_noise)  # qの更新
            alpha_new = self.estimate_alpha(height, width)  # alphaの更新
            # beta_new = self.estimate_beta(image_noise)  # betaの更新

            # ガウスBP
            alpha_new_gauss = self.estimate_alpha_gauss(height, width, alpha_gauss, beta_gauss)
            beta_new_gauss = self.estimate_beta_gauss(height, width, image_noise)

            # 更新無し (debug)
            # alpha_new = alpha
            # beta_new = beta
            # q_error_new = q_error

            ##################################
            # EMアルゴリズム終了条件 (変化率で切ったほうがよさそう)
            # if np.abs((alpha_new - alpha) / alpha) + np.abs((beta_new - beta) / beta) < threshold_EM:
            # if (np.abs(alpha - alpha_new) + np.abs(beta - beta_new) + np.abs(q_error - q_error_new)) < threshold_EM:
            #     # パラメータ更新
            #     alpha = alpha_new
            #     beta = beta_new
            #     q_error = q_error_new
            #     break
            ##################################

            # ### パラメータ更新
            # 離散BP
            alpha = alpha_new
            q_error = q_error_new
            # ガウスBP
            alpha_gauss = alpha_new_gauss
            beta_gauss = beta_new_gauss

        # ### EM loop end ### #

        # 周辺事後分布から画像化
        image_out_bp = np.zeros((width, height), dtype='uint8')  # 出力画像 (離散BP)
        image_out_gabp = np.zeros((width, height), dtype='uint8')  # 出力画像 (ガウスBP)
        for y in range(height):
            for x in range(width):
                node = self.nodes[y * width + x]  # nodeインスタンス
                post_marginal_prob = node.post_marginal_prob  # (離散BP) 周辺事後分布p(f_i|g_all)
                post_marginal_prob_gauss = node.post_marginal_prob_gauss  # (ガウスBP) 周辺事後分布の[精度:λ, 平均:μ]
                # 出力画像の保存
                image_out_bp[y, x] = np.argmax(post_marginal_prob)  # (離散BP) 事後確率が最大となる値を出力画像として保存
                image_out_gabp[y, x] = np.round(post_marginal_prob_gauss[1])  # (ガウスBP) 事後分布の平均値(ガウスなので=最大値)

        return image_out_bp, image_out_gabp

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
        (離散BP) 潜在変数間の結合を表す係数alphaの計算
            alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
        :param height: 画像縦サイズ
        :param width: 画像横サイズ
        :return: alpha_new: 推定したパラメータ
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
        alpha_new = height * width * (1 / param_tmp)  # エッジ数で平均化してパラメータを更新

        return alpha_new

    # betaの推定
    def estimate_beta(self, image_noise):
        """
        (離散BP) 観測ノイズの精度betaの推定  p(g_i|f_i) = c * exp[-beta/2(g_i - f_i)^2]
        :param beta: ノイズ精度(分散の逆数)
        :param image_noise: ノイズが加わった観測画像
        :return: beta_new: 推定パラメータ
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

    def estimate_alpha_gauss(self, height, width, alpha, beta):
        """
        (ガウスBP) 潜在変数間の結合を表す係数alpha_newの計算
        alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
        :param height: 画像縦サイズ
        :param width: 画像横サイズ
        :param alpha: 更新前のalpha
        :param beta: 観測ノイズ精度
        :return: alpha_new: 推定パラメータ
        """

        param_tmp = 0
        for edge in self.edges:  # すべてのedgeに対して結合分布を計算
            # edgeに接続されたnodeを取得
            id_1 = edge.pair_id[0]
            id_2 = edge.pair_id[1]
            node_1 = self.nodes[id_1]
            node_2 = self.nodes[id_2]

            # 事後分布の取得
            lambda_1 = node_1.post_marginal_prob_gauss[0]
            lambda_2 = node_2.post_marginal_prob_gauss[0]
            mu_1 = node_1.post_marginal_prob_gauss[1]
            mu_2 = node_2.post_marginal_prob_gauss[1]

            # node_1について node_2以外からのメッセージ(精度:λ)の和を計算
            lambda_sum_1 = 0  # node_1の近傍からのメッセージの和(node_2からのメッセージは除く)
            for neighbor_node in node_1.receive_message_gauss.keys():
                if neighbor_node.id != id_2:  # 隣接ノードがnode_2ではないとき
                    rcv_msg_neighbor = node_1.receive_message_gauss[neighbor_node]
                    lambda_sum_1 += rcv_msg_neighbor[0]

            # node_2について node_1以外からのメッセージの積を計算
            lambda_sum_2 = 0  # node_2の近傍からのメッセージの積(node_1からのメッセージは除く)
            for neighbor_node in node_2.receive_message_gauss.keys():
                if neighbor_node.id != id_1:  # 隣接ノードがnode_1ではないとき
                    rcv_msg_neighbor = node_2.receive_message_gauss[neighbor_node]
                    lambda_sum_2 += rcv_msg_neighbor[0]

            cov_12 = -alpha / (
                        (alpha + beta + lambda_sum_1) * (alpha + beta + lambda_sum_2) - alpha ** 2)  # node1,2の共分散

            param_tmp += 1 / lambda_1 + mu_1 ** 2 - 2 * (cov_12 + mu_1 * mu_2) + 1 / lambda_2 + mu_2 ** 2

        alpha_new = height * width / param_tmp

        return alpha_new

    def estimate_beta_gauss(self, height, width, image_noise):
        """
        (ガウスBP) 観測ノイズの精度betaの推定  p(g_i|f_i) = c * exp[-beta/2(g_i - f_i)^2]
        :param beta: ノイズ精度(分散の逆数)
        :param image_noise: ノイズが加わった観測画像
        :return: beta_new: 推定パラメータ
        """
        param_tmp = 0
        for y in range(height):
            for x in range(width):
                observed = image_noise[y, x]
                node = self.nodes[y * width + x]
                lambda_post = node.post_marginal_prob_gauss[0]  # 周辺事後分布の精度
                mu_post = node.post_marginal_prob_gauss[1]  # 周辺事後分布の平均

                param_tmp += 1 / lambda_post + mu_post ** 2 - 2 * observed * mu_post + observed ** 2
        beta_new = width * height / param_tmp  # 更新パラメータ

        return beta_new

    # (離散BP) 結合事後分布 p(f_i,f_j|f_all) の計算
    def calc_post_joint(self, alpha):
        """
        # (離散BP) 結合事後分布 p(f_i,f_j|f_all) の計算
        :param alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i-x_j)^2]
        """

        for edge in self.edges:  # すべてのedgeに対して結合分布を計算
            # edgeに接続されたnodeを取得
            id_1 = edge.pair_id[0]
            id_2 = edge.pair_id[1]
            node_1 = self.nodes[id_1]
            node_2 = self.nodes[id_2]

            # node_1について node_2以外からのメッセージの積を計算
            message_product_1 = np.ones(self.n_grad, dtype='float64')  # node_1の近傍からのメッセージの積(node_2からのメッセージは除く)
            for neighbor_node in node_1.receive_message.keys():
                if neighbor_node.id != id_2:  # 隣接ノードがnode_2ではないとき
                    message_product_1 *= node_1.receive_message[neighbor_node]

            # node_2について node_1以外からのメッセージの積を計算
            message_product_2 = np.ones(self.n_grad, dtype='float64')  # node_2の近傍からのメッセージの積(node_1からのメッセージは除く)
            for neighbor_node in node_2.receive_message.keys():
                if neighbor_node.id != id_1:  # 隣接ノードがnode_1ではないとき
                    message_product_2 *= node_2.receive_message[neighbor_node]

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
        self.receive_message = {}  # (離散BP)近傍ノードから受信したメッセージを保存する辞書 (階調数ベクトルのメッセージが保存される) (観測ノードからの尤度も含む)
        self.receive_message_gauss = {}  # (ガウスBP)近傍ノードから受信したメッセージを保存する辞書 {[λ:精度, μ:平均], ...} (観測ノードからの尤度は含まない)
        self.post_marginal_prob = np.zeros(n_grad)  # (離散BP) nodeの周辺事後分布
        self.post_marginal_prob_gauss = np.zeros(2)  # (ガウスBP) nodeの周辺事後分布の[精度:λ, 平均:μ]

    # 近傍ノードの追加
    def add_neighbor(self, node):
        self.neighbor_set.append(node)

    # K次元対称通信路ノイズモデルの尤度の計算
    def calc_likelihood(self, q_error, observed):
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
        self.receive_message[self] = likelihood  # 自分自身のノードのメッセージに尤度を登録

    # メッセージの初期化 (離散BP)
    def initialize_message(self):
        for neighbor_node in self.neighbor_set:
            self.receive_message[neighbor_node] = np.ones(self.n_grad)  # (離散)メッセージを1で初期化

    # メッセージの初期化
    def initialize_message_gauss(self):
        for neighbor_node in self.neighbor_set:
            self.receive_message_gauss[neighbor_node] = np.array([0.0, 0.0])  # (ガウス)メッセージを初期化 [λ:精度, μ:平均]

    # node から neigbor_target への送信メッセージを作成
    def send_message(self, neighbor_target, alpha):
        """
        node から neigbor_target への送信メッセージを作成
        :param neighbor_target: メッセージ送信先のnodeインスタンス
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

    # GaBPのメッセージ送信 [λ:精度, μ:平均]
    def send_message_gauss(self, neighbor_target, observed, alpha, beta):
        """
        node から neigbor_target への送信メッセージを作成
        :param neighbor_target: メッセージ送信先のnodeインスタンス
        :param observed: 観測画素
        :param alpha: 近傍のノードとの結合度を表す変数 f_ij = exp[ - alpha/2 * (x_i - x_j)^2]
        :param beta: ノイズ精度 f_i = exp[ - beta/2 * (x_i - g_i)^2] (g_i;観測変数)
        :return: message: node から neigbor_target への送信メッセージ [λ:精度, μ:平均]
        """
        lambda_sum = 0  # 近傍ノードからのλの和
        lambda_mu_sum = 0  # 近傍ノードからのλμの和
        for neighbor_node in self.receive_message_gauss.keys():  # target以外のすべての近傍ノードを走査 (観測ノードreceive_message[self]は含まれる)
            if neighbor_target != neighbor_node:  # target以外の近傍ノードからnodeに対してのメッセージをすべてけ合わせる(観測ノードも含む)
                rcv_msg = self.receive_message[neighbor_node]  # 近傍ノードからの受信メッセージを取得
                lambda_sum += rcv_msg[0]
                lambda_mu_sum += rcv_msg[0] * rcv_msg[1]

        # ターゲットへのメッセージ M(x_tr) = c * exp[-λ_tr/2 * (x_tr - μ_tr)^2]
        lambda_taget = (alpha * (beta + lambda_sum)) / (alpha + beta + lambda_sum)  # λ_trの計算
        mu_target = (beta * observed + lambda_mu_sum) / (beta + lambda_sum)  # μ_trの計算
        message = np.array([lambda_taget, mu_target], dtype="float64")  # [λ_tr, μ_tr]

        return message

    # 離散BPの周辺事後分布の計算
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

    # ガウスBPの周辺事後分布の計算
    def calc_post_marginal_gauss(self, alpha, beta, observed):
        """
        周辺事後分布の精度λと平均μの計算
        :param alpha:
        :param beta:
        :param observed:
        :return:　[lambda_post, mu_post] = [事後精度:λ, 事後平均:μ]
        """
        lambda_sum = 0  # 近傍ノードからの精度λの和
        lambda_mu_sum = 0  # 近傍ノードからのλμの和 (μ:平均)
        for rcv_msg in self.receive_message_gauss.values():
            lambda_neighbor = rcv_msg[0]  # 近接ノードからのメッセージの精度
            mu_neighbor = rcv_msg[0]  # 近接ノードからのメッセージの平均
            lambda_sum += lambda_neighbor
            lambda_mu_sum += lambda_neighbor * mu_neighbor

        # 事後分布の計算
        lambda_post = beta + lambda_sum  # 事後分布の精度
        mu_post = (beta * observed + lambda_mu_sum) / (beta + lambda_sum)  # 事後分布の平均

        # 周辺事後分布をnodeのメンバに登録
        self.post_marginal_prob_gauss = np.array([lambda_post, mu_post], dtype="float64")


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


def id2xy(height, width, id):
    """
    画像のidから(x, y)座標に変換
    :param height: 画像縦サイズ
    :param width: 画像横サイズ
    :param id: 画素のid (横向きに0~(h*w-1)までのindex)
    :return: [x, y]
    """
    _id = id + 1  # 1から数えたindex
    if _id % width == 0:
        y = np.int(_id / width)
        x = width
    else:
        y = np.int(_id / width) + 1
        x = _id % width
    return np.array([y, x], dtype="int32") - 1
