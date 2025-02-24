import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDAカーネルのコード
cuda_code = """
// トライ木ノードの構造体
struct TrieNode {
    int children[10];  // 子ノードのインデックス (-1 は存在しない)
    int valid_states[MAX_STATES];  // 有効な状態の配列
    int valid_states_count;  // 有効な状態の数
    bool is_end;
};

// グレイコード生成のデバイス関数
__device__ int gray_code(int n) {
    return n ^ (n >> 1);
}

// 状態遷移を計算するカーネル
__global__ void compute_transitions(
    TrieNode* nodes,
    int* dp_curr,
    int* dp_next,
    int pos,
    int num_states,
    int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;
    
    int state = tid;
    
    // グレイコード順に遷移を計算
    int prev_digit = -1;
    for (int i = 0; i < 10; i++) {
        int digit = gray_code(i);
        
        if (prev_digit != -1) {
            // 差分更新
            int diff_mask = digit ^ prev_digit;
            while (diff_mask) {
                int bit = diff_mask & (-diff_mask);
                // ビットごとの更新処理
                update_state_for_bit(nodes, dp_curr, dp_next, pos, state, digit, bit);
                diff_mask ^= bit;
            }
        } else {
            // 最初の遷移は全更新
            for (int next_state = 0; next_state < num_states; next_state++) {
                if (is_valid_transition(state, digit, next_state)) {
                    atomicAdd(&dp_next[next_state], dp_curr[state]);
                }
            }
        }
        prev_digit = digit;
    }
}

// 有効な状態を計算するカーネル
__global__ void compute_valid_states(
    TrieNode* nodes,
    int node_idx,
    int* states,
    int num_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;
    
    int state = tid;
    if (is_valid_state(nodes, node_idx, state)) {
        int idx = atomicAdd(&nodes[node_idx].valid_states_count, 1);
        nodes[node_idx].valid_states[idx] = state;
    }
}

// DP配列を初期化するカーネル
__global__ void initialize_dp(
    int* dp,
    int length,
    int num_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;
    
    dp[tid] = (tid == 0) ? 1 : 0;  // 初期状態は0のみ
}
"""

class CUDATrieDP:
    def __init__(self, max_states=1024, max_nodes=10000):
        self.max_states = max_states
        self.max_nodes = max_nodes
        
        # CUDAモジュールのコンパイル
        self.module = SourceModule(cuda_code.replace('MAX_STATES', str(max_states)))
        
        # カーネル関数の取得
        self.compute_transitions_kernel = self.module.get_function("compute_transitions")
        self.compute_valid_states_kernel = self.module.get_function("compute_valid_states")
        self.initialize_dp_kernel = self.module.get_function("initialize_dp")
        
        # デバイスメモリの確保
        self.nodes_gpu = cuda.mem_alloc(max_nodes * self._sizeof_trienode())
        self.dp_curr_gpu = cuda.mem_alloc(max_states * 4)
        self.dp_next_gpu = cuda.mem_alloc(max_states * 4)
    
    def _sizeof_trienode(self):
        # TrieNode構造体のサイズを計算
        return 4 * (10 + self.max_states + 1 + 1)  # children + valid_states + count + is_end
    
    def build_trie(self, patterns):
        """トライ木を構築してGPUに転送"""
        nodes = []  # CPU側でトライ木を構築
        
        # ルートノードの作成
        root = {
            'children': [-1] * 10,
            'valid_states': [],
            'valid_states_count': 0,
            'is_end': False
        }
        nodes.append(root)
        
        # パターンの挿入
        for pattern in patterns:
            current = 0  # ルートノードのインデックス
            for digit in pattern:
                d = int(digit)
                if nodes[current]['children'][d] == -1:
                    # 新しいノードを作成
                    new_node = {
                        'children': [-1] * 10,
                        'valid_states': [],
                        'valid_states_count': 0,
                        'is_end': False
                    }
                    nodes[current]['children'][d] = len(nodes)
                    nodes.append(new_node)
                current = nodes[current]['children'][d]
            nodes[current]['is_end'] = True
        
        # ノードデータをGPUに転送
        nodes_data = self._convert_nodes_to_array(nodes)
        cuda.memcpy_htod(self.nodes_gpu, nodes_data)
        
        return len(nodes)
    
    def _convert_nodes_to_array(self, nodes):
        """ノードリストをバイト配列に変換"""
        import struct
        
        fmt = f'<{10}i{self.max_states}iiB'  # リトルエンディアン、整数配列、カウント、ブール値
        data = bytearray()
        
        for node in nodes:
            node_data = struct.pack(
                fmt,
                *node['children'],
                *([0] * self.max_states),  # valid_states (初期化)
                node['valid_states_count'],
                node['is_end']
            )
            data.extend(node_data)
        
        return np.frombuffer(data, dtype=np.uint8)

    def solve(self, length):
        """DPを実行して結果を計算"""
        # スレッドブロックの設定
        block_size = 256
        grid_size = (self.max_states + block_size - 1) // block_size
        
        # DP配列の初期化
        self.initialize_dp_kernel(
            self.dp_curr_gpu,
            np.int32(length),
            np.int32(self.max_states),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # メインのDP計算
        for pos in range(length):
            # 状態遷移の計算
            self.compute_transitions_kernel(
                self.nodes_gpu,
                self.dp_curr_gpu,
                self.dp_next_gpu,
                np.int32(pos),
                np.int32(self.max_states),
                np.int32(self.num_nodes),
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            # dp_curr と dp_next を交換
            self.dp_curr_gpu, self.dp_next_gpu = self.dp_next_gpu, self.dp_curr_gpu
            
            # dp_next をゼロクリア
            cuda.memset_d32(self.dp_next_gpu, 0, self.max_states)
        
        # 結果を取得
        result = np.zeros(self.max_states, dtype=np.int32)
        cuda.memcpy_dtoh(result, self.dp_curr_gpu)
        
        # 受理状態の結果を合計
        return sum(result[s] for s in range(self.max_states) if self.is_accepting_state(s))
    
    def is_accepting_state(self, state):
        """受理状態の判定（オーバーライド用）"""
        return False
    
    def __del__(self):
        """デストラクタ：GPUメモリの解放"""
        self.nodes_gpu.free()
        self.dp_curr_gpu.free()
        self.dp_next_gpu.free()

  # 具体的な問題に対する実装例
class SpecificProblemSolver(CUDATrieDP):
    def is_accepting_state(self, state):
        # 問題固有の受理条件を実装
        return False

# 使用例
solver = SpecificProblemSolver()
patterns = ["123", "456"]
solver.build_trie(patterns)
result = solver.solve(5)
