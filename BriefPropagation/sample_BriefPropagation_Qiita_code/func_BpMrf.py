import numpy as np

def addNoise(image):
    output = np.copy(image)
    flags = np.random.binomial(n=1, p=0.05, size=image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if flags[i, j]:
                output[i, j] = not (output[i, j])

    return output


class MRF:
    def __init__(self):
        self.nodes = []  # MRF上のノード
        self.id = {}  # ノードのID

    # MRFにノードを追加する
    def addNode(self, id, node):
        self.nodes.append(node)
        self.id[id] = node

    # IDに応じたノードを返す
    def getNode(self, id):
        return self.id[id]

    # 全部のノードを返す
    def getNodes(self):
        return self.nodes

    # 確率伝播を開始する
    def beliefPropagation(self, iter=20):

        # 各ノードについて隣接ノードからのメッセージを初期化
        for node in self.nodes:
            node.initializeMessage()

        # 一定回数繰り返す
        for t in range(iter):
            print(t)

            # 各ノードについて，そのノードに隣接するノードへメッセージを送信する
            for node in self.nodes:
                for neighbor in node.getNeighbor():
                    neighbor.message[node] = node.sendMessage(neighbor)  # nodeからneighborへの送信メッセージをneighborの受信メッセージとして保存

        # 各ノードについて周辺分布を計算する
        for node in self.nodes:
            node.marginal()


class Node(object):
    def __init__(self, id):
        self.id = id
        self.neighbor = []  # 近傍ノードをもつリスト
        self.message = {}  # 近傍ノードから受信したメッセージを保存する辞書
        self.prob = None

        # エネルギー関数用パラメータ
        self.alpha = 10.0
        self.beta = 5.0

    def addNeighbor(self, node):
        self.neighbor.append(node)

    def getNeighbor(self):
        return self.neighbor

    # 隣接ノードからのメッセージを初期化
    def initializeMessage(self):
        for neighbor in self.neighbor:
            self.message[neighbor] = np.array([1.0, 1.0])

    # 全てのメッセージを統合
    # probは周辺分布
    def marginal(self):
        prob = 1.0

        for message in self.message.values():
            prob *= message

        prob /= np.sum(prob)
        self.prob = prob

    # 隣接ノードの状態を考慮した尤度を計算
    def sendMessage(self, target):
        neighbor_message = 1.0
        for neighbor in self.message.keys():
            if neighbor != target:
                neighbor_message *= self.message[neighbor]

        compatibility_0 = np.array([np.exp(-self.beta * np.abs(0.0 - 0.0)), np.exp(-self.beta * np.abs(0.0 - 1.0))])
        compatibility_1 = np.array([np.exp(-self.beta * np.abs(1.0 - 0.0)), np.exp(-self.beta * np.abs(1.0 - 1.0))])

        message = np.array([np.sum(neighbor_message * compatibility_0), np.sum(neighbor_message * compatibility_1)])
        message /= np.sum(message)

        return message

    # 観測値から計算する尤度
    def calcLikelihood(self, value):
        likelihood = np.array([0.0, 0.0])

        if value == 0:
            likelihood[0] = np.exp(-self.alpha * 0.0)
            likelihood[1] = np.exp(-self.alpha * 1.0)
        else:
            likelihood[0] = np.exp(-self.alpha * 1.0)
            likelihood[1] = np.exp(-self.alpha * 0.0)

        self.message[self] = likelihood


# 各画素ごとにノードを作成し，隣接画素ノードとの接続を作成する
def generateBeliefNetwork(image):
    network = MRF()
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            nodeID = width * i + j
            node = Node(nodeID)
            network.addNode(nodeID, node)

    dy = [-1, 0, 0, 1]
    dx = [0, -1, 1, 0]

    for i in range(height):
        for j in range(width):
            node = network.getNode(width * i + j)

            for k in range(4):
                if i + dy[k] >= 0 and i + dy[k] < height and j + dx[k] >= 0 and j + dx[k] < width:
                    neighbor = network.getNode(width * (i + dy[k]) + j + dx[k])
                    node.addNeighbor(neighbor)

    return network
