"""
NAND → Assembly Compiler
=========================

Compiles NAND circuits to various assembly architectures:
- x86-64
- ARM
- RISC-V
- MIPS
- WebAssembly (WASM)

Each NAND gate maps to: result = NOT(A AND B)
In assembly: AND a, b → NOT result
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from eml_pipeline.nand.nand_core import NANDCircuit, NANDGate


@dataclass
class AssemblyOutput:
    """Result of compiling a NAND circuit to assembly."""
    code: str
    arch: str
    gate_count: int
    instruction_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class NANDToAssembly:
    """Base class for NAND → Assembly compilation."""
    
    arch_name: str = "generic"
    
    def __init__(self):
        self.reg_counter = 0
        self.instructions: List[str] = []
        self.var_map: Dict[int, str] = {}  # wire_id → register/label
    
    def _fresh_reg(self) -> str:
        self.reg_counter += 1
        return f"r{self.reg_counter}"
    
    def _get_wire_reg(self, wire_id: int, circuit: NANDCircuit) -> str:
        if wire_id in self.var_map:
            return self.var_map[wire_id]
        if wire_id == -1:
            return self._const_zero()
        if wire_id == -2:
            return self._const_one()
        if wire_id < circuit.num_inputs:
            reg = f"in{wire_id}"
            self.var_map[wire_id] = reg
            return reg
        reg = self._fresh_reg()
        self.var_map[wire_id] = reg
        return reg
    
    def _const_zero(self) -> str:
        return "0"
    
    def _const_one(self) -> str:
        return "1"
    
    def _emit_nand(self, out_reg: str, a_reg: str, b_reg: str):
        """Emit: out = NOT(a AND b)"""
        # Subclass overrides this
        self.instructions.append(f"# NAND({a_reg}, {b_reg}) -> {out_reg}")
        self.instructions.append(f"and tmp, {a_reg}, {b_reg}")
        self.instructions.append(f"not {out_reg}, tmp")
    
    def compile(self, circuit: NANDCircuit) -> AssemblyOutput:
        """Compile a NAND circuit to assembly."""
        self.instructions = []
        self.var_map = {}
        self.reg_counter = 0
        
        # Header
        self._emit_header(circuit)
        
        # Input setup
        self._emit_inputs(circuit)
        
        # Gate compilation
        for gate in circuit.gates:
            a_reg = self._get_wire_reg(gate.input_a, circuit)
            b_reg = self._get_wire_reg(gate.input_b, circuit)
            out_reg = self._fresh_reg()
            self.var_map[gate.output] = out_reg
            self._emit_nand(out_reg, a_reg, b_reg)
        
        # Output
        self._emit_outputs(circuit)
        
        # Footer
        self._emit_footer(circuit)
        
        code = "\n".join(self.instructions)
        return AssemblyOutput(
            code=code,
            arch=self.arch_name,
            gate_count=circuit.gate_count(),
            instruction_count=len([i for i in self.instructions if not i.startswith("#")]),
            metadata={
                "circuit_gates": circuit.gate_count(),
                "circuit_depth": circuit.depth(),
                "num_inputs": circuit.num_inputs,
            }
        )
    
    def _emit_header(self, circuit: NANDCircuit):
        self.instructions.append(f"# EML-NAND Pipeline — {self.arch_name}")
        self.instructions.append(f"# Gates: {circuit.gate_count()}, Inputs: {circuit.num_inputs}")
        self.instructions.append("")
    
    def _emit_inputs(self, circuit: NANDCircuit):
        for i in range(circuit.num_inputs):
            self.instructions.append(f"# Input wire {i}")
    
    def _emit_outputs(self, circuit: NANDCircuit):
        self.instructions.append("")
        self.instructions.append("# Outputs")
        for i, w in enumerate(circuit.output_wires):
            reg = self.var_map.get(w, f"out_{i}")
            self.instructions.append(f"# Output {i}: wire {w} -> {reg}")
    
    def _emit_footer(self, circuit: NANDCircuit):
        self.instructions.append("")
        self.instructions.append("# End of NAND circuit")


class NANDToX86(NANDToAssembly):
    """Compile NAND circuit to x86-64 assembly."""
    arch_name = "x86_64"
    
    def __init__(self):
        super().__init__()
        self.reg_names = ["rax", "rcx", "rdx", "rbx", "rsi", "rdi",
                          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
        self.reg_idx = 0
    
    def _fresh_reg(self) -> str:
        if self.reg_idx < len(self.reg_names):
            reg = self.reg_names[self.reg_idx]
            self.reg_idx += 1
            return f"%{reg}"
        self.reg_counter += 1
        return f"%stack_{self.reg_counter}"
    
    def _const_zero(self) -> str:
        return "$0"
    
    def _const_one(self) -> str:
        return "$1"
    
    def _emit_nand(self, out_reg: str, a_reg: str, b_reg: str):
        self.instructions.append(f"    # NAND({a_reg}, {b_reg}) -> {out_reg}")
        self.instructions.append(f"    movq {a_reg}, %rax")
        self.instructions.append(f"    andq {b_reg}, %rax")
        self.instructions.append(f"    notq %rax")
        self.instructions.append(f"    movq %rax, {out_reg}")
    
    def _emit_header(self, circuit: NANDCircuit):
        self.instructions.append(".text")
        self.instructions.append(".globl nand_circuit")
        self.instructions.append("nand_circuit:")
        self.instructions.append(f"    # EML-NAND Pipeline — x86_64")
        self.instructions.append(f"    # Gates: {circuit.gate_count()}")
    
    def _emit_footer(self, circuit: NANDCircuit):
        self.instructions.append("    ret")


class NANDToARM(NANDToAssembly):
    """Compile NAND circuit to ARM64 assembly."""
    arch_name = "arm64"
    
    def __init__(self):
        super().__init__()
        self.reg_idx = 0
    
    def _fresh_reg(self) -> str:
        self.reg_idx += 1
        if self.reg_idx <= 28:
            return f"x{self.reg_idx}"
        return f"[sp, #{(self.reg_idx - 28) * 8}]"
    
    def _emit_nand(self, out_reg: str, a_reg: str, b_reg: str):
        self.instructions.append(f"    // NAND({a_reg}, {b_reg}) -> {out_reg}")
        self.instructions.append(f"    and x0, {a_reg}, {b_reg}")
        self.instructions.append(f"    mvn {out_reg}, x0")
    
    def _emit_header(self, circuit: NANDCircuit):
        self.instructions.append(".text")
        self.instructions.append(".globl nand_circuit")
        self.instructions.append("nand_circuit:")
    
    def _emit_footer(self, circuit: NANDCircuit):
        self.instructions.append("    ret")


class NANDToRISCV(NANDToAssembly):
    """Compile NAND circuit to RISC-V assembly."""
    arch_name = "riscv64"
    
    def __init__(self):
        super().__init__()
        self.reg_idx = 0
    
    def _fresh_reg(self) -> str:
        self.reg_idx += 1
        if self.reg_idx <= 31:
            return f"x{self.reg_idx}"
        return f"s{(self.reg_idx - 31) * 8}(sp)"
    
    def _emit_nand(self, out_reg: str, a_reg: str, b_reg: str):
        self.instructions.append(f"    # NAND({a_reg}, {b_reg}) -> {out_reg}")
        self.instructions.append(f"    and x0, {a_reg}, {b_reg}")
        self.instructions.append(f"    not {out_reg}, x0")
    
    def _emit_header(self, circuit: NANDCircuit):
        self.instructions.append(".text")
        self.instructions.append(".globl nand_circuit")
        self.instructions.append("nand_circuit:")
    
    def _emit_footer(self, circuit: NANDCircuit):
        self.instructions.append("    ret")


class NANDToMIPS(NANDToAssembly):
    """Compile NAND circuit to MIPS assembly."""
    arch_name = "mips"
    
    def __init__(self):
        super().__init__()
        self.reg_idx = 0
    
    def _fresh_reg(self) -> str:
        self.reg_idx += 1
        if self.reg_idx <= 25:
            return f"${self.reg_idx}"
        return f"{(self.reg_idx - 25) * 4}($sp)"
    
    def _emit_nand(self, out_reg: str, a_reg: str, b_reg: str):
        self.instructions.append(f"    # NAND({a_reg}, {b_reg}) -> {out_reg}")
        self.instructions.append(f"    and $at, {a_reg}, {b_reg}")
        self.instructions.append(f"    nor {out_reg}, $at, $zero")
    
    def _emit_header(self, circuit: NANDCircuit):
        self.instructions.append(".text")
        self.instructions.append(".globl nand_circuit")
        self.instructions.append("nand_circuit:")
    
    def _emit_footer(self, circuit: NANDCircuit):
        self.instructions.append("    jr $ra")


class NANDToWASM(NANDToAssembly):
    """Compile NAND circuit to WebAssembly (WAT format)."""
    arch_name = "wasm"
    
    def __init__(self):
        super().__init__()
        self.local_counter = 0
    
    def _fresh_reg(self) -> str:
        self.local_counter += 1
        return f"$l{self.local_counter}"
    
    def _emit_nand(self, out_reg: str, a_reg: str, b_reg: str):
        self.instructions.append(f"    ;; NAND({a_reg}, {b_reg}) -> {out_reg}")
        self.instructions.append(f"    local.get {a_reg}")
        self.instructions.append(f"    local.get {b_reg}")
        self.instructions.append(f"    i32.and")
        self.instructions.append(f"    i32.const -1")
        self.instructions.append(f"    i32.xor")  # NOT = XOR with -1
        self.instructions.append(f"    local.set {out_reg}")
    
    def _emit_header(self, circuit: NANDCircuit):
        self.instructions.append("(module")
        self.instructions.append("  (func $nand_circuit (export \"nand_circuit\")")
        for i in range(min(circuit.num_inputs, 16)):
            self.instructions.append(f"    (param $in{i} i32)")
        self.instructions.append(f"    (result i32)")
    
    def _emit_footer(self, circuit: NANDCircuit):
        if circuit.output_wires:
            last_wire = circuit.output_wires[-1]
            reg = self.var_map.get(last_wire, "$l0")
            self.instructions.append(f"    local.get {reg}")
        self.instructions.append("  )")
        self.instructions.append(")")


# ─── Compiler Registry ────────────────────────────────────────────────────────

COMPILERS = {
    "x86": NANDToX86,
    "x86_64": NANDToX86,
    "arm": NANDToARM,
    "arm64": NANDToARM,
    "riscv": NANDToRISCV,
    "riscv64": NANDToRISCV,
    "mips": NANDToMIPS,
    "wasm": NANDToWASM,
    "webassembly": NANDToWASM,
}

def compile_nand_to_asm(circuit: NANDCircuit, arch: str = "x86") -> AssemblyOutput:
    """Compile a NAND circuit to the specified assembly architecture."""
    compiler_cls = COMPILERS.get(arch.lower())
    if compiler_cls is None:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(COMPILERS.keys())}")
    compiler = compiler_cls()
    return compiler.compile(circuit)
