import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import powerlaw

def generate_scale_free_network(n, m, alpha=1):
    """
    Barabási-Albert modelを使用してスケールフリーネットワークを生成
    n: 最終的なノード数
    m: 各新ノードからのエッジ数
    alpha: 優先的接続の強さ
    """
    G = nx.barabasi_albert_graph(n, m, alpha)
    return G

def plot_degree_distribution(G, ax=None):
    """
    ネットワークの次数分布をプロット
    """
    degrees = [d for n, d in G.degree()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # powerlaw パッケージを使用してフィッティング
    fit = powerlaw.Fit(degrees, xmin=min(degrees))
    fit.plot_pdf(ax=ax, color='b', label='Empirical')
    fit.power_law.plot_pdf(ax=ax, color='r', linestyle='--', label='Power law fit')

    ax.set_xlabel('Degree (log scale)')
    ax.set_ylabel('PDF (log scale)')
    ax.set_title(f'Degree Distribution (log-log scale)\nPower law exponent: {fit.power_law.alpha:.2f}')
    ax.legend()
    ax.grid(True)
    return fig, ax

def plot_network(G, ax=None):
    """
    ネットワーク構造を可視化
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    degrees = dict(G.degree())
    nx.draw(G, pos, ax=ax, node_size=[v * 5 for v in degrees.values()],
            node_color=list(degrees.values()),
            cmap=plt.cm.viridis,
            with_labels=False,
            edge_color='gray',
            width=0.1,
            alpha=0.7)
    ax.set_title('Scale-Free Network Visualization')
    plt.colorbar(ax.collections[0], label='Node Degree', ax=ax)
    ax.axis('off')
    return fig, ax

def calculate_network_properties(G):
    """
    ネットワークの基本的な特性を計算
    """
    properties = {
        "Number of nodes": G.number_of_nodes(),
        "Number of edges": G.number_of_edges(),
        "Average clustering coefficient": nx.average_clustering(G),
        "Average shortest path length": nx.average_shortest_path_length(G),
        "Assortativity coefficient": nx.degree_assortativity_coefficient(G)
    }
    return properties

def track_network_evolution(n_final, m, alpha=1, steps=10):
    """
    ネットワークの時間発展を追跡
    """
    n_initial = m + 1  # 初期ノード数
    n_step = (n_final - n_initial) // (steps - 1)
    
    evolution_data = []
    for i in range(steps):
        n = n_initial + i * n_step
        G = generate_scale_free_network(n, m, alpha)
        properties = calculate_network_properties(G)
        properties["Nodes"] = n
        evolution_data.append(properties)
    
    return evolution_data

def plot_network_evolution(evolution_data):
    """
    ネットワーク進化データをプロット
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Network Evolution", fontsize=16)
    
    nodes = [data["Nodes"] for data in evolution_data]
    
    # Average clustering coefficient
    axs[0, 0].plot(nodes, [data["Average clustering coefficient"] for data in evolution_data], 'bo-')
    axs[0, 0].set_xlabel("Number of Nodes")
    axs[0, 0].set_ylabel("Average Clustering Coefficient")
    axs[0, 0].set_title("Clustering Coefficient Evolution")
    
    # Average shortest path length
    axs[0, 1].plot(nodes, [data["Average shortest path length"] for data in evolution_data], 'ro-')
    axs[0, 1].set_xlabel("Number of Nodes")
    axs[0, 1].set_ylabel("Average Shortest Path Length")
    axs[0, 1].set_title("Average Shortest Path Length Evolution")
    
    # Assortativity coefficient
    axs[1, 0].plot(nodes, [data["Assortativity coefficient"] for data in evolution_data], 'go-')
    axs[1, 0].set_xlabel("Number of Nodes")
    axs[1, 0].set_ylabel("Assortativity Coefficient")
    axs[1, 0].set_title("Assortativity Coefficient Evolution")
    
    # Number of edges
    axs[1, 1].plot(nodes, [data["Number of edges"] for data in evolution_data], 'mo-')
    axs[1, 1].set_xlabel("Number of Nodes")
    axs[1, 1].set_ylabel("Number of Edges")
    axs[1, 1].set_title("Edge Count Evolution")
    
    plt.tight_layout()
    plt.show()

# メイン実行部分
if __name__ == "__main__":
    n_final = 10000  # 最終ノード数
    m = 2  # 各新ノードからのエッジ数
    alpha = 1  # 優先的接続の強さ
    
    # 最終ネットワークの生成と分析
    G_final = generate_scale_free_network(n_final, m, alpha)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plot_degree_distribution(G_final, ax=ax1)
    plot_network(G_final, ax=ax2)
    plt.tight_layout()
    plt.show()
    
    # ネットワークの特性を表示
    properties = calculate_network_properties(G_final)
    for key, value in properties.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # ネットワークの時間発展を追跡
    evolution_data = track_network_evolution(n_final, m, alpha, steps=20)
    plot_network_evolution(evolution_data)
