import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.adj_matrix = np.zeros((num_nodes, num_nodes))
        for edge in edges:
            self.adj_matrix[edge[0], edge[1]] = 1
            self.adj_matrix[edge[1], edge[0]] = 1

    def visualize(self):
        G = nx.from_numpy_array(self.adj_matrix)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=1, font_size=15)
        plt.title('Graph Visualization')
        plt.show()

class CliffordAlgebra:
    @staticmethod
    def mult(v1, v2):
        return tuple(np.multiply(v1, v2[::-1]) + np.multiply(v1[::-1], v2))

    @staticmethod
    def add(v1, v2):
        return tuple(np.add(v1, v2))

    @staticmethod
    def scalar_mult(scalar, v):
        return tuple(scalar * np.array(v))

class CliffordGNN:
    def __init__(self, num_nodes, num_features, num_layers):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_layers = num_layers
        self.features = np.zeros((num_nodes, num_features))
        self.messages = np.zeros((num_nodes, num_nodes, num_features))
    
    def generate_messages(self):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.messages[i, j] = self.calculate_message(self.features[i], self.features[j])

    def calculate_message(self, v1, v2):
        return CliffordAlgebra.mult(v1, v2)

    def propagate_messages(self, graph):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if graph.adj_matrix[i, j] == 1:
                    self.messages[i, j] = self.calculate_message(self.features[i], self.features[j])

    def aggregate_messages(self, graph):
        for i in range(self.num_nodes):
            aggregated_message = np.zeros(self.num_features)
            for j in range(self.num_nodes):
                if graph.adj_matrix[i, j] == 1:
                    aggregated_message = CliffordAlgebra.add(aggregated_message, self.messages[j, i])
            self.features[i] = CliffordAlgebra.add(self.features[i], aggregated_message)

    def update_features(self):
        self.features = np.array([CliffordAlgebra.scalar_mult(0.5, feature) for feature in self.features])

    def visualize_features(self):
        plt.figure(figsize=(8, 6))
        for i in range(self.num_nodes):
            plt.arrow(0, 0, self.features[i, 0], self.features[i, 1], head_width=0.05, head_length=0.1, fc='blue', ec='blue')
            plt.text(self.features[i, 0], self.features[i, 1], f'Node {i}', fontsize=12, ha='right', va='bottom')
        plt.title('Feature Vector Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()

# グラフデータの作成
num_nodes = 5
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
graph = Graph(num_nodes, edges)

# CliffordGNNのインスタンス化
num_features = 2
num_layers = 3
clifford_gnn = CliffordGNN(num_nodes, num_features, num_layers)

# メッセージの生成
clifford_gnn.generate_messages()

# メッセージの伝播
clifford_gnn.propagate_messages(graph)

# メッセージの受信と集約
clifford_gnn.aggregate_messages(graph)

# 特徴の更新
clifford_gnn.update_features()

# グラフの可視化
graph.visualize()

# 特徴ベクトルの可視化
clifford_gnn.visualize_features()
