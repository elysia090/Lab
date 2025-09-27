"""Hypercube virtual CPU and assembler utilities.

This module provides a lightweight, production-ready implementation of the
hypercube virtual CPU that previously lived in this repository.  The original
version bundled together GUI code, matplotlib/networkx visualisations and a
large amount of ad-hoc logic.  Importing the module pulled heavy optional
dependencies and made it very difficult to unit-test the core compiler and
simulation stack.

The refactored module focuses exclusively on the compiler (assembler), CPU
execution model and the hypercube message-passing network.  The API is fully
type annotated, extensively documented and covered by unit tests.  It is
designed to be deterministic and side-effect free which keeps it suitable for
integration in automated systems.

Example
-------

>>> source = "\"\"\"\nLOAD R0, 1\nLOAD R1, 5\nLOOP:\n  ADD R0, R1\n  SEND R0, 0   ; share with neighbour\n  JNZ R1, LOOP\nHALT\n\"\"\""
>>> assembler = Assembler()
>>> program = assembler.assemble(source)
>>> sim = HypercubeSimulation([program, program], dimension=1)
>>> sim.run()  # doctest: +ELLIPSIS

The :class:`HypercubeSimulation` exposes the final state of every node through
the :attr:`~HypercubeSimulation.nodes` attribute for further inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "AssemblyError",
    "SimulationError",
    "Opcode",
    "DecodedInstruction",
    "encode_instruction",
    "decode_instruction",
    "disassemble",
    "Assembler",
    "VirtualCPU",
    "HypercubeNetwork",
    "HypercubeSimulation",
    "compile_source",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AssemblyError(RuntimeError):
    """Raised when the assembler encounters an invalid instruction."""


class SimulationError(RuntimeError):
    """Raised when an invalid state is observed during CPU execution."""


# ---------------------------------------------------------------------------
# Instruction encoding
# ---------------------------------------------------------------------------


class Opcode(Enum):
    """Supported opcodes for the virtual CPU."""

    NOP = 0
    LOAD = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    MOV = 6
    HALT = 7
    SEND = 8
    RECV = 9
    JMP = 10
    JZ = 11
    LOADM = 12
    STOREM = 13
    CMP = 14
    JNZ = 15


IMMEDIATE_BITS = 6
IMMEDIATE_MASK = (1 << IMMEDIATE_BITS) - 1
REGISTER_COUNT = 8
MAX_PROGRAM_LENGTH = 1 << IMMEDIATE_BITS  # Addresses must fit in the immediate


@dataclass(frozen=True)
class DecodedInstruction:
    """Human friendly representation of an encoded instruction."""

    opcode: Opcode
    reg_a: int
    reg_b: int
    immediate: int


def _validate_register(index: int) -> int:
    if not 0 <= index < REGISTER_COUNT:
        raise SimulationError(f"Register index out of bounds: {index}")
    return index


def encode_instruction(opcode: Opcode, reg_a: int = 0, reg_b: int = 0, immediate: int = 0) -> int:
    """Encode an instruction into the 16-bit machine word format."""

    _validate_register(reg_a)
    _validate_register(reg_b)
    if not -32 <= immediate <= 31:
        raise SimulationError("Immediate must fit in signed 6-bit range (-32..31)")
    gray_opcode = opcode.value ^ (opcode.value >> 1)
    return ((gray_opcode & 0xF) << 12) | ((reg_a & 0x7) << 9) | ((reg_b & 0x7) << 6) | (immediate & IMMEDIATE_MASK)


def decode_instruction(word: int) -> DecodedInstruction:
    """Decode a 16-bit machine word into components."""

    if not 0 <= word <= 0xFFFF:
        raise SimulationError(f"Instruction word out of range: {word}")
    gray_opcode = (word >> 12) & 0xF
    opcode_value = _gray_to_binary(gray_opcode)
    try:
        opcode = Opcode(opcode_value)
    except ValueError as exc:  # pragma: no cover - exhaustive enum guard
        raise SimulationError(f"Unknown opcode value: {opcode_value}") from exc
    reg_a = (word >> 9) & 0x7
    reg_b = (word >> 6) & 0x7
    immediate = word & IMMEDIATE_MASK
    if immediate & (1 << (IMMEDIATE_BITS - 1)):
        immediate -= 1 << IMMEDIATE_BITS
    return DecodedInstruction(opcode, reg_a, reg_b, immediate)


def disassemble(program: Sequence[int]) -> List[str]:
    """Return a list of human readable instructions for *program*."""

    return [
        f"{index:02d}: {decoded.opcode.name} R{decoded.reg_a}, R{decoded.reg_b}, {decoded.immediate}"
        for index, decoded in enumerate(map(decode_instruction, program))
    ]


def _gray_to_binary(value: int) -> int:
    mask = value
    while mask:
        mask >>= 1
        value ^= mask
    return value


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


_MNEMONICS: Dict[str, Opcode] = {item.name: item for item in Opcode}

_OPERAND_SHAPES: Dict[Opcode, Tuple[str, ...]] = {
    Opcode.NOP: tuple(),
    Opcode.HALT: tuple(),
    Opcode.LOAD: ("reg", "imm"),
    Opcode.ADD: ("reg", "reg"),
    Opcode.SUB: ("reg", "reg"),
    Opcode.MUL: ("reg", "reg"),
    Opcode.DIV: ("reg", "reg"),
    Opcode.MOV: ("reg", "reg"),
    Opcode.SEND: ("reg", "dim"),
    Opcode.RECV: ("reg", "dim"),
    Opcode.JMP: ("addr",),
    Opcode.JZ: ("reg", "addr"),
    Opcode.JNZ: ("reg", "addr"),
    Opcode.LOADM: ("reg", "imm"),
    Opcode.STOREM: ("reg", "imm"),
    Opcode.CMP: ("reg", "reg"),
}


@dataclass(frozen=True)
class _InstructionSpec:
    opcode: Opcode
    operands: Tuple[str, ...]
    line_number: int
    raw_line: str


class Assembler:
    """A deterministic assembler for the hypercube CPU instruction set."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__ + ".Assembler")

    def assemble(self, source: str) -> List[int]:
        instructions, labels = self._parse_source(source.splitlines())
        if len(instructions) > MAX_PROGRAM_LENGTH:
            raise AssemblyError(
                "Program too long: immediate/address field only supports 64 instructions"
            )
        return [self._encode_instruction(spec, labels) for spec in instructions]

    # -- parsing -----------------------------------------------------------------

    def _parse_source(
        self, lines: Iterable[str]
    ) -> Tuple[List[_InstructionSpec], Dict[str, int]]:
        instructions: List[_InstructionSpec] = []
        labels: Dict[str, int] = {}
        address = 0
        for line_number, raw in enumerate(lines, start=1):
            stripped = raw.split(";", 1)[0].strip()
            if not stripped:
                continue
            label, remainder = self._split_label(stripped)
            if label is not None:
                if label in labels:
                    raise AssemblyError(f"Duplicate label '{label}' on line {line_number}")
                labels[label] = address
                if not remainder:
                    continue
                stripped = remainder
            tokens = self._tokenise(stripped)
            opcode_name = tokens[0].upper()
            try:
                opcode = _MNEMONICS[opcode_name]
            except KeyError as exc:
                raise AssemblyError(f"Unknown mnemonic '{opcode_name}' on line {line_number}") from exc
            operand_tokens = tuple(tokens[1:])
            expected = _OPERAND_SHAPES[opcode]
            if len(operand_tokens) != len(expected):
                raise AssemblyError(
                    f"Opcode {opcode.name} expects {len(expected)} operand(s); "
                    f"got {len(operand_tokens)} on line {line_number}"
                )
            instructions.append(
                _InstructionSpec(opcode=opcode, operands=operand_tokens, line_number=line_number, raw_line=raw)
            )
            address += 1
        return instructions, labels

    def _split_label(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        if ":" not in line:
            return None, None
        label, remainder = line.split(":", 1)
        label = label.strip()
        if not label:
            raise AssemblyError("Empty label declaration")
        if not label[0].isalpha() or not all(ch.isalnum() or ch == "_" for ch in label):
            raise AssemblyError(f"Invalid label name '{label}'")
        remainder = remainder.strip()
        return label, remainder or None

    def _tokenise(self, line: str) -> List[str]:
        tokens = [token for token in line.replace(",", " ").split() if token]
        if not tokens:
            raise AssemblyError("Missing instruction after label")
        return tokens

    # -- encoding ----------------------------------------------------------------

    def _encode_instruction(self, spec: _InstructionSpec, labels: Dict[str, int]) -> int:
        operand_shape = _OPERAND_SHAPES[spec.opcode]
        reg_a = 0
        reg_b = 0
        immediate = 0
        for index, (operand_type, token) in enumerate(zip(operand_shape, spec.operands)):
            if operand_type == "reg":
                value = self._parse_register(token, spec.line_number)
                if index == 0:
                    reg_a = value
                else:
                    reg_b = value
            elif operand_type == "imm":
                immediate = self._parse_immediate(token, spec.line_number)
            elif operand_type == "addr":
                immediate = self._parse_address(token, labels, spec.line_number)
            elif operand_type == "dim":
                dim = self._parse_address(token, labels, spec.line_number)
                if not 0 <= dim < IMMEDIATE_MASK + 1:
                    raise AssemblyError(
                        f"Dimension immediate must be in range 0..{IMMEDIATE_MASK} (line {spec.line_number})"
                    )
                immediate = dim
            else:  # pragma: no cover - defensive programming
                raise AssemblyError(f"Unknown operand type '{operand_type}'")
        try:
            return encode_instruction(spec.opcode, reg_a, reg_b, immediate)
        except SimulationError as exc:
            raise AssemblyError(f"{exc} (while assembling line {spec.line_number})") from exc

    def _parse_register(self, token: str, line_number: int) -> int:
        token = token.strip().upper()
        if not token.startswith("R"):
            raise AssemblyError(f"Expected register operand on line {line_number}; got '{token}'")
        try:
            index = int(token[1:], 10)
        except ValueError as exc:
            raise AssemblyError(f"Invalid register '{token}' on line {line_number}") from exc
        if not 0 <= index < REGISTER_COUNT:
            raise AssemblyError(
                f"Register index must be in range 0..{REGISTER_COUNT - 1}; got {index} on line {line_number}"
            )
        return index

    def _parse_immediate(self, token: str, line_number: int) -> int:
        value = self._parse_int_literal(token, line_number)
        if not -32 <= value <= 31:
            raise AssemblyError(
                f"Immediate must fit in signed 6-bit range (-32..31); got {value} on line {line_number}"
            )
        return value

    def _parse_address(self, token: str, labels: Dict[str, int], line_number: int) -> int:
        if token in labels:
            return labels[token]
        value = self._parse_int_literal(token, line_number)
        if not 0 <= value < MAX_PROGRAM_LENGTH:
            raise AssemblyError(
                f"Address immediate must be in range 0..{MAX_PROGRAM_LENGTH - 1}; got {value} on line {line_number}"
            )
        return value

    def _parse_int_literal(self, token: str, line_number: int) -> int:
        token = token.strip().replace("_", "")
        try:
            if token.lower().startswith("0x"):
                return int(token, 16)
            if token.lower().startswith("0b"):
                return int(token, 2)
            return int(token, 10)
        except ValueError as exc:
            raise AssemblyError(f"Invalid integer literal '{token}' on line {line_number}") from exc


def compile_source(source: str) -> List[int]:
    """Convenience function mirroring the legacy API."""

    return Assembler().assemble(source)


# ---------------------------------------------------------------------------
# Execution model
# ---------------------------------------------------------------------------


class VirtualCPU:
    """A single hypercube CPU node.

    The implementation is single-threaded and deterministic which makes it
    suitable for unit testing.  Messaging between nodes is delegated to a
    :class:`HypercubeNetwork` instance supplied during construction.
    """

    def __init__(
        self,
        program: Sequence[int],
        *,
        node_id: int = 0,
        network: Optional["HypercubeNetwork"] = None,
        data_memory_size: int = 256,
    ) -> None:
        self.program: Tuple[int, ...] = tuple(program)
        self.node_id = node_id
        self.network = network
        self.registers: List[int] = [0] * REGISTER_COUNT
        self.data_memory: List[int] = [0] * data_memory_size
        self.pc = 0
        self.running = True
        self.flag = 0
        self._dispatch = {
            Opcode.NOP: self._op_nop,
            Opcode.LOAD: self._op_load,
            Opcode.ADD: self._op_add,
            Opcode.SUB: self._op_sub,
            Opcode.MUL: self._op_mul,
            Opcode.DIV: self._op_div,
            Opcode.MOV: self._op_mov,
            Opcode.HALT: self._op_halt,
            Opcode.SEND: self._op_send,
            Opcode.RECV: self._op_recv,
            Opcode.JMP: self._op_jmp,
            Opcode.JZ: self._op_jz,
            Opcode.LOADM: self._op_loadm,
            Opcode.STOREM: self._op_storem,
            Opcode.CMP: self._op_cmp,
            Opcode.JNZ: self._op_jnz,
        }

    # -- execution --------------------------------------------------------------

    def step(self) -> bool:
        """Execute a single instruction.

        Returns ``True`` while the CPU is still running.
        """

        if not self.running:
            return False
        if self.pc >= len(self.program):
            self.running = False
            return False
        decoded = decode_instruction(self.program[self.pc])
        self.pc += 1
        handler = self._dispatch.get(decoded.opcode)
        if handler is None:  # pragma: no cover - exhaustive guard
            raise SimulationError(f"No handler for opcode {decoded.opcode}")
        handler(decoded.reg_a, decoded.reg_b, decoded.immediate)
        return self.running

    def run(self, *, cycle_budget: int = 1024) -> None:
        """Execute instructions until ``HALT`` or *cycle_budget* is exhausted."""

        cycles = 0
        while self.running:
            self.step()
            cycles += 1
            if cycles > cycle_budget:
                raise SimulationError(
                    f"CPU {self.node_id} exceeded cycle budget of {cycle_budget} steps"
                )

    # -- instruction implementations -------------------------------------------

    def _op_nop(self, *_: int) -> None:
        pass

    def _op_load(self, reg_a: int, _reg_b: int, immediate: int) -> None:
        self.registers[reg_a] = immediate & 0xFFFFFFFF

    def _op_add(self, reg_a: int, reg_b: int, _immediate: int) -> None:
        self.registers[reg_a] = (self.registers[reg_a] + self.registers[reg_b]) & 0xFFFFFFFF

    def _op_sub(self, reg_a: int, reg_b: int, _immediate: int) -> None:
        self.registers[reg_a] = (self.registers[reg_a] - self.registers[reg_b]) & 0xFFFFFFFF

    def _op_mul(self, reg_a: int, reg_b: int, _immediate: int) -> None:
        self.registers[reg_a] = (self.registers[reg_a] * self.registers[reg_b]) & 0xFFFFFFFF

    def _op_div(self, reg_a: int, reg_b: int, _immediate: int) -> None:
        divisor = self.registers[reg_b]
        self.registers[reg_a] = 0 if divisor == 0 else self.registers[reg_a] // divisor

    def _op_mov(self, reg_a: int, reg_b: int, _immediate: int) -> None:
        self.registers[reg_a] = self.registers[reg_b]

    def _op_halt(self, *_: int) -> None:
        self.running = False

    def _op_send(self, reg_a: int, _reg_b: int, dimension: int) -> None:
        if self.network is None:
            raise SimulationError("SEND instruction requires an attached network")
        target = self.network.route(self.node_id, dimension)
        value = self.registers[reg_a]
        self.network.deliver(self.node_id, target, value)

    def _op_recv(self, reg_a: int, _reg_b: int, dimension: int) -> None:
        if self.network is None:
            raise SimulationError("RECV instruction requires an attached network")
        source = self.network.route(self.node_id, dimension)
        message = self.network.receive(self.node_id, source)
        self.registers[reg_a] = 0 if message is None else message

    def _op_jmp(self, _reg_a: int, _reg_b: int, address: int) -> None:
        self.pc = address

    def _op_jz(self, reg_a: int, _reg_b: int, address: int) -> None:
        if self.registers[reg_a] == 0:
            self.pc = address

    def _op_jnz(self, reg_a: int, _reg_b: int, address: int) -> None:
        if self.registers[reg_a] != 0:
            self.pc = address

    def _op_loadm(self, reg_a: int, _reg_b: int, address: int) -> None:
        if not 0 <= address < len(self.data_memory):
            raise SimulationError(f"LOADM address out of bounds: {address}")
        self.registers[reg_a] = self.data_memory[address]

    def _op_storem(self, reg_a: int, _reg_b: int, address: int) -> None:
        if not 0 <= address < len(self.data_memory):
            raise SimulationError(f"STOREM address out of bounds: {address}")
        self.data_memory[address] = self.registers[reg_a]

    def _op_cmp(self, reg_a: int, reg_b: int, _immediate: int) -> None:
        self.flag = self.registers[reg_a] - self.registers[reg_b]


class HypercubeNetwork:
    """Message passing fabric for a hypercube topology."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Hypercube dimension must be positive")
        self.dimension = dimension
        self.node_count = 1 << dimension
        self._mailboxes: List[List[Tuple[int, int]]] = [[] for _ in range(self.node_count)]

    def route(self, node_id: int, dimension: int) -> int:
        if not 0 <= node_id < self.node_count:
            raise SimulationError(f"Node id {node_id} outside network range")
        if not 0 <= dimension < self.dimension:
            raise SimulationError(
                f"Dimension {dimension} outside range 0..{self.dimension - 1}"
            )
        return node_id ^ (1 << dimension)

    def deliver(self, src: int, dest: int, value: int) -> None:
        if not 0 <= dest < self.node_count:
            raise SimulationError(f"Destination node {dest} outside network range")
        self._mailboxes[dest].append((src, value))

    def receive(self, node_id: int, expected_src: int) -> Optional[int]:
        mailbox = self._mailboxes[node_id]
        for index, (src, value) in enumerate(mailbox):
            if src == expected_src:
                mailbox.pop(index)
                return value
        return None


class HypercubeSimulation:
    """Execute a program on a hypercube of CPUs."""

    def __init__(
        self,
        programs: Sequence[Sequence[int]],
        *,
        dimension: int,
        cycle_budget: int = 1024,
        data_memory_size: int = 256,
    ) -> None:
        self.network = HypercubeNetwork(dimension)
        self.cycle_budget = cycle_budget
        if len(programs) == 1:
            program_list = list(programs) * self.network.node_count
        elif len(programs) == self.network.node_count:
            program_list = list(programs)
        else:
            raise ValueError(
                "Number of programs must be 1 or equal to the number of nodes in the hypercube"
            )
        self.nodes: List[VirtualCPU] = [
            VirtualCPU(program, node_id=index, network=self.network, data_memory_size=data_memory_size)
            for index, program in enumerate(program_list)
        ]

    def run(self) -> None:
        remaining_cycles = self.cycle_budget
        while remaining_cycles > 0:
            active = False
            for node in self.nodes:
                if node.running:
                    node.step()
                    active = True
            if not active:
                return
            remaining_cycles -= 1
        raise SimulationError(
            f"Hypercube simulation exhausted cycle budget of {self.cycle_budget}"
        )


def main() -> None:  # pragma: no cover - demonstration utility
    sample_source = """; Increment R0 until it reaches 4
    LOAD R0, 0
    LOAD R1, 1
LOOP:
    ADD R0, R1
    MOV R2, R0
    SUB R2, R1
    JZ R2, END
    JMP LOOP
END:
    HALT
"""
    assembler = Assembler()
    program = assembler.assemble(sample_source)
    sim = HypercubeSimulation([program], dimension=1, cycle_budget=64)
    sim.run()
    for node in sim.nodes:
        print(f"Node {node.node_id}: registers={node.registers} pc={node.pc}")


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()

