#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored and Extended Hypercube Virtual CPU Simulator with Native Frontend

【命令形式 (16ビット)】
  [4ビット opcode (Gray code) | 3ビット regA | 3ビット regB | 6ビット immediate]

【サポート命令】
  基本命令:
    0: NOP,   1: LOAD,   2: ADD,    3: SUB,
    4: MUL,   5: DIV,    6: MOV,    7: HALT
  通信命令:
    8: SEND,  9: RECV
  分岐命令:
    10: JMP,  11: JZ,   15: JNZ   (JZ: jump if reg==0, JNZ: jump if reg != 0)
  データメモリ操作:
    12: LOADM, 13: STOREM
  比較命令:
    14: CMP   (CMP regA, regB → flag = regA - regB)

この命令セットは、十分なメモリが与えられればチューリング完全な計算モデルとなります。

各ノードはハイパーキューブ構造上に存在し、ノード間の通信は queue.Queue を用いて高速化されています。
また、各ノードのシミュレーションは並列スレッドで実行され、PyQt5 のフロントエンドでリアルタイムに可視化されます.
"""

import sys
import logging
import threading
import time
import queue
from typing import List, Tuple, Callable, Dict, Optional

# --- PyQt5 関連 ---
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx

# ======================================================
# ロギング設定（必要最小限に調整）
# ======================================================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
def set_debug(enable: bool) -> None:
    if enable:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

# ======================================================
# 再帰的 Gray Code 生成＆変換
# ======================================================
def recursive_gray_codes(n: int) -> List[int]:
    if n == 0:
        return [0]
    if n == 1:
        return [0, 1]
    prev = recursive_gray_codes(n - 1)
    mirror = prev[::-1]
    return [code << 1 for code in prev] + [(code << 1) | 1 for code in mirror]

def gray_to_binary(n: int) -> int:
    mask = n
    while mask:
        mask >>= 1
        n ^= mask
    return n

GRAY_TO_BIN: List[int] = [gray_to_binary(i) for i in range(16)]

def encode_instruction(bin_opcode: int, regA: int = 0, regB: int = 0, imm: int = 0) -> int:
    # サイクリック Gray code 化
    gray_opcode = bin_opcode ^ (bin_opcode >> 1)
    return ((gray_opcode & 0xF) << 12) | ((regA & 0x7) << 9) | ((regB & 0x7) << 6) | (imm & 0x3F)

# ======================================================
# 再帰的ハイパーキューブエッジ生成
# ======================================================
def generate_hypercube_edges(dim: int) -> List[Tuple[int, int]]:
    if dim == 0:
        return []
    if dim == 1:
        return [(0, 1)]
    half = 2 ** (dim - 1)
    sub_edges = generate_hypercube_edges(dim - 1)
    edges = []
    for (a, b) in sub_edges:
        edges.append((a, b))
        edges.append((a + half, b + half))
    for i in range(half):
        edges.append((i, i + half))
    return edges

# ======================================================
# 再帰的フラクタル配置によるハイパーキューブレイアウト
# ======================================================
def recursive_hypercube_layout(dim: int) -> Dict[int, Tuple[float, float]]:
    n_nodes = 2 ** dim
    positions = {}
    for i in range(n_nodes):
        bits = format(i, f'0{dim}b')
        x = sum((1 if bit == '1' else -1) for bit in bits[::2])
        y = sum((1 if bit == '1' else -1) for bit in bits[1::2])
        positions[i] = (x, y)
    return positions

# ======================================================
# VirtualCPU クラス（基本および拡張命令対応）
# ======================================================
class VirtualCPU:
    def __init__(self, program: List[int], debug: bool = False) -> None:
        self.registers: List[int] = [0] * 8
        self.memory: List[int] = program[:]  # プログラムメモリ
        self.pc: int = 0
        self.running: bool = True
        self.debug: bool = debug
        # データメモリとフラグ
        self.data_memory: List[int] = [0] * 256
        self.flag: int = 0
        # 固定長 16 要素のディスパッチテーブル
        self.dispatch: List[Callable[[int, int, int], None]] = [self.unknown] * 16
        self.dispatch[0] = self.nop
        self.dispatch[1] = self.load
        self.dispatch[2] = self.add
        self.dispatch[3] = self.sub
        self.dispatch[4] = self.mul
        self.dispatch[5] = self.div
        self.dispatch[6] = self.mov
        self.dispatch[7] = self.halt
        self.dispatch[8] = self.send
        self.dispatch[9] = self.recv
        self.dispatch[10] = self.jmp
        self.dispatch[11] = self.jz
        self.dispatch[12] = self.loadm
        self.dispatch[13] = self.storem
        self.dispatch[14] = self.cmp
        self.dispatch[15] = self.jnz

    def unknown(self, regA: int, regB: int, imm: int) -> None:
        logger.error("Unknown instruction encountered!")
        self.running = False

    def fetch_decode_execute(self) -> None:
        if self.pc >= len(self.memory):
            self.running = False
            return
        instr: int = self.memory[self.pc]
        self.pc += 1
        opcode_gray: int = (instr >> 12) & 0xF
        opcode_bin: int = GRAY_TO_BIN[opcode_gray]
        regA: int = (instr >> 9) & 0x7
        regB: int = (instr >> 6) & 0x7
        imm: int = instr & 0x3F
        if self.debug:
            logger.debug(f"[PC={self.pc-1:04d}] 0x{instr:04X} | Gray:0x{opcode_gray:X} -> Bin:{opcode_bin} | R{regA} R{regB} | imm={imm}")
        if opcode_bin < len(self.dispatch):
            self.dispatch[opcode_bin](regA, regB, imm)
        else:
            self.unknown(regA, regB, imm)
    
    def run(self) -> None:
        while self.running:
            self.fetch_decode_execute()
    
    # 基本命令
    def nop(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug("  NOP")
    
    def load(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  LOAD: R{regA} <- {imm}")
        self.registers[regA] = imm
    
    def add(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  ADD: R{regA} = {self.registers[regA]} + R{regB}({self.registers[regB]})")
        self.registers[regA] = (self.registers[regA] + self.registers[regB]) & 0xFFFFFFFF
    
    def sub(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  SUB: R{regA} = {self.registers[regA]} - R{regB}({self.registers[regB]})")
        self.registers[regA] = (self.registers[regA] - self.registers[regB]) & 0xFFFFFFFF
    
    def mul(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  MUL: R{regA} = {self.registers[regA]} * R{regB}({self.registers[regB]})")
        self.registers[regA] = (self.registers[regA] * self.registers[regB]) & 0xFFFFFFFF
    
    def div(self, regA: int, regB: int, imm: int) -> None:
        if self.registers[regB] == 0:
            if self.debug:
                logger.debug(f"  DIV: Division by zero, R{regA} set to 0")
            self.registers[regA] = 0
        else:
            if self.debug:
                logger.debug(f"  DIV: R{regA} = {self.registers[regA]} // R{regB}({self.registers[regB]})")
            self.registers[regA] = self.registers[regA] // self.registers[regB]
    
    def mov(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  MOV: R{regA} <- R{regB}({self.registers[regB]})")
        self.registers[regA] = self.registers[regB]
    
    def halt(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug("  HALT")
        self.running = False
    
    # 通信命令
    def send(self, regA: int, regB: int, imm: int) -> None:
        if imm >= self.dim:
            if self.debug:
                logger.debug(f"  SEND: Invalid dimension {imm}")
            return
        target: int = self.node_id ^ (1 << imm)  if hasattr(self, "node_id") else 0
        value: int = self.registers[regA]
        if self.debug:
            logger.debug(f"  SEND: Sending R{regA} value {value} to Node {target} (dim {imm})")
        self.network.deliver_message(target, self.node_id, value)
    
    def recv(self, regA: int, regB: int, imm: int) -> None:
        if imm >= self.dim:
            if self.debug:
                logger.debug(f"  RECV: Invalid dimension {imm}")
            self.registers[regA] = 0
            return
        source: int = self.node_id ^ (1 << imm)
        try:
            # キューから先頭のメッセージを取得
            while True:
                src, val = self.mailbox.get_nowait()
                if src == source:
                    self.registers[regA] = val
                    if self.debug:
                        logger.debug(f"  RECV: Received {val} from Node {source} into R{regA}")
                    return
                else:
                    # 該当しない場合は再度キューへ
                    self.mailbox.put((src, val))
                    break
        except queue.Empty:
            pass
        if self.debug:
            logger.debug(f"  RECV: No message from Node {source}; R{regA} set to 0")
        self.registers[regA] = 0

    # 分岐命令
    def jmp(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  JMP: Jumping to address {imm}")
        self.pc = imm
    
    def jz(self, regA: int, regB: int, imm: int) -> None:
        if self.registers[regA] == 0:
            if self.debug:
                logger.debug(f"  JZ: R{regA} is zero; jumping to {imm}")
            self.pc = imm
        else:
            if self.debug:
                logger.debug(f"  JZ: R{regA} is nonzero; no jump")
    
    def jnz(self, regA: int, regB: int, imm: int) -> None:
        if self.registers[regA] != 0:
            if self.debug:
                logger.debug(f"  JNZ: R{regA} is nonzero; jumping to {imm}")
            self.pc = imm
        else:
            if self.debug:
                logger.debug(f"  JNZ: R{regA} is zero; no jump")
    
    # データメモリ操作命令
    def loadm(self, regA: int, regB: int, imm: int) -> None:
        if imm < 0 or imm >= len(self.data_memory):
            logger.error("LOADM: Memory address out of bounds")
            self.running = False
            return
        if self.debug:
            logger.debug(f"  LOADM: R{regA} <- MEM[{imm}]")
        self.registers[regA] = self.data_memory[imm]
    
    def storem(self, regA: int, regB: int, imm: int) -> None:
        if imm < 0 or imm >= len(self.data_memory):
            logger.error("STOREM: Memory address out of bounds")
            self.running = False
            return
        if self.debug:
            logger.debug(f"  STOREM: MEM[{imm}] <- R{regA} ({self.registers[regA]})")
        self.data_memory[imm] = self.registers[regA]
    
    # 比較命令
    def cmp(self, regA: int, regB: int, imm: int) -> None:
        if self.debug:
            logger.debug(f"  CMP: Comparing R{regA} ({self.registers[regA]}) and R{regB} ({self.registers[regB]})")
        self.flag = self.registers[regA] - self.registers[regB]

# ======================================================
# HypercubeNode クラス（VirtualCPU を継承、並列実行対応）
# ======================================================
class HypercubeNode(VirtualCPU):
    def __init__(self, node_id: int, program: List[int], dim: int, debug: bool = False) -> None:
        super().__init__(program, debug)
        self.node_id: int = node_id
        self.dim: int = dim
        # mailbox を queue.Queue によるスレッドセーフな実装に変更
        self.mailbox: "queue.Queue[Tuple[int,int]]" = queue.Queue()
        self.dispatch[8] = self.send
        self.dispatch[9] = self.recv
        if self.debug:
            logger.debug(f"Node {self.node_id} initialized in {self.dim}D hypercube.")

# ======================================================
# HypercubeCPU クラス（複数ノードの並列シミュレーション）
# ======================================================
class HypercubeCPU:
    def __init__(self, programs: List[List[int]], dim: int, debug: bool = False, visualize: bool = False) -> None:
        self.dim: int = dim
        self.n_nodes: int = 2 ** dim
        self.debug: bool = debug
        self.visualize: bool = visualize
        if len(programs) == 1:
            programs = programs * self.n_nodes
        elif len(programs) != self.n_nodes:
            raise ValueError(f"Invalid number of programs: expected {self.n_nodes}, got {len(programs)}")
        self.nodes: List[HypercubeNode] = [
            HypercubeNode(node_id=i, program=programs[i], dim=dim, debug=debug)
            for i in range(self.n_nodes)
        ]
        for node in self.nodes:
            node.network = self
        self.node_threads: List[threading.Thread] = []
    
    def deliver_message(self, target_id: int, src_id: int, value: int) -> None:
        self.nodes[target_id].mailbox.put((src_id, value))
        if self.debug:
            logger.debug(f"[Network] Node {src_id} -> Node {target_id}: {value}")
    
    def run(self) -> None:
        for node in self.nodes:
            t = threading.Thread(target=node.run, name=f"NodeThread-{node.node_id}")
            t.start()
            self.node_threads.append(t)
        while any(t.is_alive() for t in self.node_threads):
            time.sleep(0.1)
        for t in self.node_threads:
            t.join()
    
    def dump_registers(self) -> None:
        output = "\n--- Final Registers ---\n"
        for node in self.nodes:
            output += f"Node {node.node_id}: {node.registers}\n"
        output += "\n--- Data Memory (first 16 addresses) ---\n" + str(self.nodes[0].data_memory[:16]) + "\n"
        output += "\n--- Mailbox Contents ---\n"
        for node in self.nodes:
            msgs = list(node.mailbox.queue)
            output += f"Node {node.node_id}: {msgs}\n"
        print(output)

# ======================================================
# Qt 用 HypercubeVisualizer（matplotlib 埋め込み）
# ======================================================
class QtHypercubeVisualizer:
    def __init__(self, cpu: HypercubeCPU, figure: Figure) -> None:
        self.cpu = cpu
        self.figure = figure
        self.ax = self.figure.add_subplot(111)
        self.graph = nx.Graph()
        self.pos = {}
    def init_plot(self) -> None:
        for i in range(self.cpu.n_nodes):
            self.graph.add_node(i)
        edges = generate_hypercube_edges(self.cpu.dim)
        for (i, j) in edges:
            self.graph.add_edge(i, j)
        self.pos = recursive_hypercube_layout(self.cpu.dim)
    def update(self) -> None:
        self.ax.clear()
        colors = []
        labels = {}
        for node in self.cpu.nodes:
            colors.append("green" if node.running else "red")
            labels[node.node_id] = f"ID:{node.node_id}\nR0:{node.registers[0]}\nPC:{node.pc}"
        nx.draw(self.graph, pos=self.pos, ax=self.ax, with_labels=False, node_color=colors, node_size=800)
        for node, (x, y) in self.pos.items():
            self.ax.text(x, y, labels[node],
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=10, fontweight='bold', color="white")
        self.ax.set_title("Advanced Parallel Hypercube CPU Simulation")
        self.figure.canvas.draw()

# ======================================================
# SimulationThread（QThread によるシミュレーション実行）
# ======================================================
class SimulationThread(QtCore.QThread):
    simulation_finished = QtCore.pyqtSignal()
    def __init__(self, cpu: HypercubeCPU) -> None:
        super().__init__()
        self.cpu = cpu
    def run(self) -> None:
        self.cpu.run()
        self.simulation_finished.emit()

# ======================================================
# コンパイラ（アセンブラ）部
# ======================================================
MNEMONICS: Dict[str, int] = {
    "NOP": 0,
    "LOAD": 1,
    "ADD": 2,
    "SUB": 3,
    "MUL": 4,
    "DIV": 5,
    "MOV": 6,
    "HALT": 7,
    "SEND": 8,
    "RECV": 9,
    "JMP": 10,
    "JZ": 11,
    "LOADM": 12,
    "STOREM": 13,
    "CMP": 14,
    "JNZ": 15,
}

def parse_register(token: str) -> int:
    token = token.strip().upper()
    if token.startswith("R"):
        return int(token[1:])
    else:
        raise ValueError(f"Invalid register token: {token}")

def parse_immediate(token: str) -> int:
    token = token.strip()
    if token.startswith("0x") or token.startswith("0X"):
        return int(token, 16)
    else:
        return int(token)

def assemble_line(line: str) -> Optional[int]:
    line = line.split(";", 1)[0].strip()
    if not line:
        return None
    if ":" in line:
        parts = line.split(":", 1)
        line = parts[1].strip()
        if not line:
            return None
    tokens = line.split()
    mnemonic = tokens[0].upper()
    if mnemonic not in MNEMONICS:
        raise ValueError(f"Unknown mnemonic: {mnemonic}")
    opcode = MNEMONICS[mnemonic]
    # 命令ごとにオペランドの数を判定
    if mnemonic in ["NOP", "HALT"]:
        return encode_instruction(opcode)
    elif mnemonic == "LOAD":
        if len(tokens) < 3:
            raise ValueError("LOAD requires 2 operands")
        reg = parse_register(tokens[1].replace(",", ""))
        imm = parse_immediate(tokens[2].replace(",", ""))
        return encode_instruction(opcode, regA=reg, imm=imm)
    elif mnemonic in ["ADD", "SUB", "MUL", "DIV", "MOV"]:
        if len(tokens) < 3:
            raise ValueError(f"{mnemonic} requires 2 operands")
        regA = parse_register(tokens[1].replace(",", ""))
        regB = parse_register(tokens[2].replace(",", ""))
        return encode_instruction(opcode, regA=regA, regB=regB)
    elif mnemonic in ["SEND", "RECV"]:
        if len(tokens) < 3:
            raise ValueError(f"{mnemonic} requires 2 operands")
        reg = parse_register(tokens[1].replace(",", ""))
        imm = parse_immediate(tokens[2].replace(",", ""))
        return encode_instruction(opcode, regA=reg, imm=imm)
    elif mnemonic in ["JMP", "JZ", "JNZ"]:
        if len(tokens) < 3 and mnemonic in ["JZ", "JNZ"]:
            raise ValueError(f"{mnemonic} requires 2 operands")
        if mnemonic == "JMP":
            # JMP: 1 operand (immediate)
            imm = parse_immediate(tokens[1].replace(",", ""))
            return encode_instruction(opcode, imm=imm)
        else:
            # JZ, JNZ: 2 operands (register, immediate)
            reg = parse_register(tokens[1].replace(",", ""))
            imm = parse_immediate(tokens[2].replace(",", ""))
            return encode_instruction(opcode, regA=reg, imm=imm)
    elif mnemonic in ["LOADM", "STOREM"]:
        if len(tokens) < 3:
            raise ValueError(f"{mnemonic} requires 2 operands")
        reg = parse_register(tokens[1].replace(",", ""))
        imm = parse_immediate(tokens[2].replace(",", ""))
        return encode_instruction(opcode, regA=reg, imm=imm)
    elif mnemonic == "CMP":
        if len(tokens) < 3:
            raise ValueError("CMP requires 2 operands")
        regA = parse_register(tokens[1].replace(",", ""))
        regB = parse_register(tokens[2].replace(",", ""))
        return encode_instruction(opcode, regA=regA, regB=regB)
    else:
        raise ValueError(f"Unhandled mnemonic: {mnemonic}")

def compile_source(source: str) -> List[int]:
    machine_code: List[int] = []
    for line in source.splitlines():
        try:
            inst = assemble_line(line)
            if inst is not None:
                machine_code.append(inst)
        except Exception as e:
            logger.error(f"Error assembling line '{line}': {e}")
            raise
    return machine_code

# ======================================================
# PyQt5 フロントエンド
# ======================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Advanced Hypercube CPU Simulator")
        self.resize(1100, 750)
        self._setup_ui()
        self.simulation_thread: Optional[SimulationThread] = None
        self.cpu: Optional[HypercubeCPU] = None
        self.visualizer: Optional[QtHypercubeVisualizer] = None

    def _setup_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # エディタ：アセンブリソース編集用
        self.editor = QtWidgets.QPlainTextEdit()
        sample_source = (
            "; Sample: Increment R0 from 0 to 10 and then HALT\n"
            "LOAD R0, 0      ; R0 <- 0\n"
            "LOAD R1, 1      ; R1 <- 1 (constant)\n"
            "LOAD R2, 10     ; R2 <- 10 (end value)\n"
            "LOOP:\n"
            "ADD R0, R1      ; R0 = R0 + 1\n"
            "MOV R3, R0      ; R3 <- R0\n"
            "SUB R3, R2      ; R3 = R3 - R2\n"
            "JZ R3, 8        ; If R3==0 then jump to address 8 (END)\n"
            "JMP 3           ; Otherwise jump back to address 3 (ADD instruction)\n"
            "END:\n"
            "HALT\n"
        )
        self.editor.setPlainText(sample_source)
        
        # ボタン
        self.compile_run_button = QtWidgets.QPushButton("Compile && Run")
        self.compile_run_button.clicked.connect(self.on_compile_and_run)
        
        # 可視化キャンバス（matplotlib 埋め込み）
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        
        # 結果表示エリア
        self.result_display = QtWidgets.QPlainTextEdit()
        self.result_display.setReadOnly(True)
        
        # レイアウト配置
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(QtWidgets.QLabel("Assembly Source"))
        left_layout.addWidget(self.editor)
        left_layout.addWidget(self.compile_run_button)
        left_layout.addWidget(QtWidgets.QLabel("Simulation Result"))
        left_layout.addWidget(self.result_display)
        
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(self.canvas, 4)
        
        central_widget.setLayout(main_layout)
        
        # タイマー：可視化更新（250 msec）
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(250)

    def on_compile_and_run(self) -> None:
        source = self.editor.toPlainText()
        try:
            machine_code = compile_source(source)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Compile Error", str(e))
            return
        # 機械語表示
        code_text = "\n".join(f"{i:04d}: 0x{code:04X}" for i, code in enumerate(machine_code))
        self.result_display.setPlainText("Compiled Machine Code:\n" + code_text)
        # 例：3次元ハイパーキューブ（8ノード）で全ノード共通プログラムを実行
        self.cpu = HypercubeCPU(programs=[machine_code], dim=3, debug=False, visualize=False)
        self.visualizer = QtHypercubeVisualizer(self.cpu, self.figure)
        self.visualizer.init_plot()
        self.simulation_thread = SimulationThread(self.cpu)
        self.simulation_thread.simulation_finished.connect(self.on_simulation_finished)
        self.simulation_thread.start()

    def update_visualization(self) -> None:
        if self.visualizer is not None and self.cpu is not None:
            self.visualizer.update()

    def on_simulation_finished(self) -> None:
        if self.cpu is not None:
            output = "\n--- Final Registers ---\n"
            for node in self.cpu.nodes:
                output += f"Node {node.node_id}: {node.registers}\n"
            output += "\n--- Data Memory (first 16 addresses) ---\n" + str(self.cpu.nodes[0].data_memory[:16]) + "\n"
            output += "\n--- Mailbox Contents ---\n"
            for node in self.cpu.nodes:
                msgs = list(node.mailbox.queue)
                output += f"Node {node.node_id}: {msgs}\n"
            self.result_display.appendPlainText(output)

# ======================================================
# メイン処理
# ======================================================
def main() -> None:
    set_debug(False)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
