"""Simulation utilities for the compiler and systems package."""

from .hypercube_virtual_cpu import (
    Assembler,
    AssemblyError,
    DecodedInstruction,
    HypercubeNetwork,
    HypercubeSimulation,
    Opcode,
    SimulationError,
    VirtualCPU,
    compile_source,
    decode_instruction,
    disassemble,
    encode_instruction,
)

__all__ = [
    "Assembler",
    "AssemblyError",
    "DecodedInstruction",
    "HypercubeNetwork",
    "HypercubeSimulation",
    "Opcode",
    "SimulationError",
    "VirtualCPU",
    "compile_source",
    "decode_instruction",
    "disassemble",
    "encode_instruction",
]

