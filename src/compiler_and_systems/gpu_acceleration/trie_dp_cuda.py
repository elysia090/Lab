import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDAカーネルの定義
cuda_code = """
struct TrieNode {
    int children[10];       // 子ノードのインデックス（-1は未定義）
    int valid_digits[10];   // 有効なdigitのリスト
    int num_valid_digits;   // 有効なdigitの数
};

__device__ int compute_next_state(TrieNode* nodes, int state, int digit) {
    return nodes[state].children[digit];
}

__global__ void compute_transitions_shared(
    TrieNode* nodes,
    int* dp_curr_global,
    int* dp_next_global,
    int pos,
    int num_states,
    int num_nodes
) {
    extern __shared__ int dp_curr_shared[];
    int tid = threadIdx.x;
    int state = blockIdx.x * blockDim.x + tid;

    if (state < num_states) {
        dp_curr_shared[tid] = dp_curr_global[state];
    }
    __syncthreads();

    if (state < num_states) {
        for (int i = 0; i < nodes[state].num_valid_digits; i++) {
            int digit = nodes[state].valid_digits[i];
            int next_state = compute_next_state(nodes, state, digit);
            if (next_state != -1) {
                atomicAdd(&dp_next_global[next_state], dp_curr_shared[tid]);
            }
        }
    }
}
"""

# カーネルのコンパイル
module = SourceModule(cuda_code)
compute_transitions_shared = module.get_function("compute_transitions_shared")

# トライ木とDPを管理するクラス
class CUDATrieDP:
    def __init__(self, max_states=1024, max_nodes=10000):
        self.max_states = max_states
        self.max_nodes = max_nodes
        self.nodes_gpu = cuda.mem_alloc(max_nodes * self._sizeof_trienode())
        self.dp_curr_gpu = cuda.mem_alloc(max_states * 4)  # int32
        self.dp_next_gpu = cuda.mem_alloc(max_states * 4)  # int32

    def _sizeof_trienode(self):
        return 4 * (10 + 10 + 1)  # children[10] + valid_digits[10] + num_valid_digits

    def build_trie(self, patterns):
        """トライ木を構築しGPUに転送"""
        nodes = [{'children': [-1] * 10, 'valid_digits': [], 'num_valid_digits': 0}]
        for pattern in patterns:
            current = 0
            for digit in pattern:
                d = int(digit)
                if nodes[current]['children'][d] == -1:
                    nodes[current]['children'][d] = len(nodes)
                    nodes.append({'children': [-1] * 10, 'valid_digits': [], 'num_valid_digits': 0})
                current = nodes[current]['children'][d]

        for node in nodes:
            node['valid_digits'] = [d for d in range(10) if node['children'][d] != -1]
            node['num_valid_digits'] = len(node['valid_digits'])

        nodes_data = self._convert_nodes_to_array(nodes)
        cuda.memcpy_htod(self.nodes_gpu, nodes_data)
        self.num_nodes = len(nodes)
        return self.num_nodes

    def _convert_nodes_to_array(self, nodes):
        """ノードをバイト配列に変換"""
        import struct
        fmt = '<' + 'i' * (10 + 10 + 1)
        data = bytearray()
        for node in nodes:
            children = node['children']
            valid_digits = node['valid_digits'] + [-1] * (10 - len(node['valid_digits']))
            num_valid_digits = node['num_valid_digits']
            node_data = struct.pack(fmt, *children, *valid_digits, num_valid_digits)
            data.extend(node_data)
        return np.frombuffer(data, dtype=np.uint8)

    def solve(self, length):
        """DPを実行"""
        block_size = 256
        grid_size = (self.max_states + block_size - 1) // block_size

        dp_curr = np.zeros(self.max_states, dtype=np.int32)
        dp_curr[0] = 1  # 初期状態
        cuda.memcpy_htod(self.dp_curr_gpu, dp_curr)
        cuda.memset_d32(self.dp_next_gpu, 0, self.max_states)

        for pos in range(length):
            compute_transitions_shared(
                self.nodes_gpu,
                self.dp_curr_gpu,
                self.dp_next_gpu,
                np.int32(pos),
                np.int32(self.max_states),
                np.int32(self.num_nodes),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
                shared=block_size * 4
            )
            self.dp_curr_gpu, self.dp_next_gpu = self.dp_next_gpu, self.dp_curr_gpu
            cuda.memset_d32(self.dp_next_gpu, 0, self.max_states)

        result = np.zeros(self.max_states, dtype=np.int32)
        cuda.memcpy_dtoh(result, self.dp_curr_gpu)
        return sum(result[s] for s in range(self.max_states) if self.is_accepting_state(s))

    def is_accepting_state(self, state):
        """受理状態の判定（問題に応じて実装）"""
        return False

    def __del__(self):
        """GPUメモリの解放"""
        self.nodes_gpu.free()
        self.dp_curr_gpu.free()
        self.dp_next_gpu.free()

# 使用例
solver = CUDATrieDP()
patterns = ["123", "456"]
solver.build_trie(patterns)
result = solver.solve(5)
print("Result:", result)
