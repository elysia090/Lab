"""Tests for the hypercube virtual CPU and assembler."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compiler_and_systems.simulation import (
    Assembler,
    HypercubeSimulation,
    Opcode,
    VirtualCPU,
    compile_source,
    decode_instruction,
    encode_instruction,
)


def test_encode_decode_roundtrip() -> None:
    word = encode_instruction(Opcode.ADD, reg_a=2, reg_b=3, immediate=-5)
    decoded = decode_instruction(word)
    assert decoded.opcode is Opcode.ADD
    assert decoded.reg_a == 2
    assert decoded.reg_b == 3
    assert decoded.immediate == -5


def test_assembler_handles_labels_and_literals() -> None:
    source = """
    START:
        LOAD R0, 1
        LOAD R1, 0b10
        ADD R0, R1
        JZ R1, END
        SUB R1, R0
        JMP START
    END:
        HALT
    """

    assembler = Assembler()
    program = assembler.assemble(source)

    # Ensure label was resolved and the program length matches expectations.
    assert len(program) == 7
    decoded = [decode_instruction(word).opcode for word in program]
    assert decoded[0] is Opcode.LOAD
    assert decoded[-1] is Opcode.HALT


def test_virtual_cpu_executes_program() -> None:
    source = """
        LOAD R0, 0
        LOAD R1, 1
        LOAD R2, 4
    LOOP:
        ADD R0, R1
        MOV R3, R0
        SUB R3, R2
        JZ R3, END
        JMP LOOP
    END:
        HALT
    """

    program = compile_source(source)
    cpu = VirtualCPU(program)
    cpu.run(cycle_budget=64)
    assert cpu.registers[0] == 4
    assert not cpu.running


def test_hypercube_message_passing() -> None:
    sender = compile_source(
        """
        LOAD R0, 21
        SEND R0, 0
        HALT
        """
    )
    receiver = compile_source(
        """
    WAIT:
        RECV R1, 0
        JZ R1, WAIT
        HALT
        """
    )

    sim = HypercubeSimulation([sender, receiver], dimension=1, cycle_budget=16)
    sim.run()

    assert sim.nodes[0].registers[0] == 21
    assert sim.nodes[1].registers[1] == 21

