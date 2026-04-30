"""
Assembly Decompiler — Assembly → NAND Circuit Reconstruction
=============================================================

Reconstructs NAND circuits from compiled assembly code for all
supported architectures (x86-64, ARM64, RISC-V, MIPS, WASM).

The decompilation process:
1. Parse assembly to identify NAND gate patterns (AND + NOT sequences)
2. Reconstruct the NAND circuit DAG from the parsed patterns
3. Map registers/locals back to circuit wire IDs
4. Attach metadata for reverse pipeline (T3+T4 transitions)

This enables the full reverse path:
  Assembly → NAND → ε-NAND → ApproxEML → EML → LaTeX

Based on: "The EML–NAND Duality" by Daniel Derycke (2026)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

from eml_pipeline.nand.nand_core import (
    NANDCircuit, NANDGate, nand_bool, soft_nand
)


# ─── Parsed Gate Representation ──────────────────────────────────────────────

@dataclass
class ParsedNANDOp:
    """A single NAND-like operation extracted from assembly."""
    output_reg: str
    input_a_reg: str
    input_b_reg: str
    source_line: str
    line_number: int
    arch: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompileResult:
    """Result of decompiling assembly to NAND circuit."""
    circuit: NANDCircuit
    arch: str
    gates_parsed: int
    gates_reconstructed: int
    register_map: Dict[str, int]  # register/label → wire_id
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Architecture-Specific Parsers ────────────────────────────────────────────

class ASMParserBase:
    """Base class for architecture-specific assembly parsers."""

    arch_name: str = "generic"

    def __init__(self):
        self.ops: List[ParsedNANDOp] = []
        self._reg_counter = 0

    def parse(self, asm_code: str) -> List[ParsedNANDOp]:
        """Parse assembly code and extract NAND operations."""
        self.ops = []
        lines = asm_code.strip().split('\n')

        for line_num, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            if stripped.startswith('.') or stripped.startswith('module'):
                continue  # Skip directives
            self._parse_line(stripped, line_num)

        return self.ops

    def _parse_line(self, line: str, line_num: int) -> None:
        """Override in subclass for architecture-specific parsing."""
        pass

    def _add_op(self, out_reg: str, in_a: str, in_b: str,
                line: str, line_num: int, **meta) -> None:
        self.ops.append(ParsedNANDOp(
            output_reg=out_reg,
            input_a_reg=in_a,
            input_b_reg=in_b,
            source_line=line,
            line_number=line_num,
            arch=self.arch_name,
            metadata=meta
        ))


class X86ASMParser(ASMParserBase):
    """
    Parse x86-64 assembly for NAND patterns.

    The NAND compiler emits sequences like:
        movq src_a, %rax
        andq src_b, %rax
        notq %rax
        movq %rax, dest

    We detect AND + NOT sequences to reconstruct NAND gates.
    """

    arch_name = "x86_64"

    def __init__(self):
        super().__init__()
        self._pending_and: Optional[Dict] = None

    def _parse_line(self, line: str, line_num: int) -> None:
        # Match: andq %src, %rax  or  andq $imm, %reg
        and_match = re.match(r'\s*and[qd]?\s+(.+?),\s*(.+)', line, re.IGNORECASE)
        not_match = re.match(r'\s*not[qd]?\s+(.+)', line, re.IGNORECASE)
        nand_match = re.match(r'\s*nand[qd]?\s+(.+?),\s*(.+)', line, re.IGNORECASE)

        if nand_match:
            # Direct NAND instruction (if available)
            src_a = nand_match.group(1).strip().lstrip('$%')
            src_b = nand_match.group(2).strip().lstrip('$%')
            out_reg = src_b  # Result stored in second operand
            self._add_op(out_reg, src_a, src_b, line, line_num)

        elif and_match:
            src_a = and_match.group(1).strip().lstrip('$%')
            dest = and_match.group(2).strip().lstrip('$%')
            self._pending_and = {
                'src_a': src_a, 'dest': dest, 'line': line, 'line_num': line_num
            }

        elif not_match and self._pending_and:
            dest = not_match.group(1).strip().lstrip('$%')
            if dest == self._pending_and['dest']:
                # AND + NOT = NAND
                self._add_op(
                    out_reg=dest,
                    in_a=self._pending_and['src_a'],
                    in_b=dest,  # AND result was in-place
                    line=f"{self._pending_and['line']} ; {line}",
                    line_num=self._pending_and['line_num'],
                    is_and_not=True
                )
                self._pending_and = None
            else:
                self._pending_and = None

        else:
            self._pending_and = None


class ARMASMParser(ASMParserBase):
    """
    Parse ARM64 assembly for NAND patterns.

    Pattern: and x0, a, b → mvn out, x0
    Or: bic + mvn for NAND with complement.
    """

    arch_name = "arm64"

    def __init__(self):
        super().__init__()
        self._pending_and: Optional[Dict] = None

    def _parse_line(self, line: str, line_num: int) -> None:
        and_match = re.match(r'\s*and\s+(\w+),\s*(\w+),\s*(\w+)', line, re.IGNORECASE)
        mvn_match = re.match(r'\s*mvn\s+(\w+),\s*(\w+)', line, re.IGNORECASE)
        nand_match = re.match(r'\s*bic\s+(\w+),\s*(\w+),\s*(\w+)', line, re.IGNORECASE)

        if and_match:
            dest = and_match.group(1).strip()
            src_a = and_match.group(2).strip()
            src_b = and_match.group(3).strip()
            self._pending_and = {
                'dest': dest, 'src_a': src_a, 'src_b': src_b,
                'line': line, 'line_num': line_num
            }

        elif mvn_match and self._pending_and:
            out_reg = mvn_match.group(1).strip()
            src = mvn_match.group(2).strip()
            if src == self._pending_and['dest']:
                self._add_op(
                    out_reg=out_reg,
                    in_a=self._pending_and['src_a'],
                    in_b=self._pending_and['src_b'],
                    line=f"{self._pending_and['line']} ; {line}",
                    line_num=self._pending_and['line_num'],
                    is_and_mvn=True
                )
            self._pending_and = None

        elif nand_match:
            dest = nand_match.group(1).strip()
            src_a = nand_match.group(2).strip()
            src_b = nand_match.group(3).strip()
            self._add_op(dest, src_a, src_b, line, line_num, is_bic=True)

        else:
            if not and_match:
                self._pending_and = None


class RISCVASMParser(ASMParserBase):
    """
    Parse RISC-V assembly for NAND patterns.

    Pattern: and x0, a, b → not out, x0
    """

    arch_name = "riscv64"

    def __init__(self):
        super().__init__()
        self._pending_and: Optional[Dict] = None

    def _parse_line(self, line: str, line_num: int) -> None:
        and_match = re.match(r'\s*and\s+(\w+),\s*(\w+),\s*(\w+)', line, re.IGNORECASE)
        not_match = re.match(r'\s*not\s+(\w+),\s*(\w+)', line, re.IGNORECASE)

        if and_match:
            dest = and_match.group(1).strip()
            src_a = and_match.group(2).strip()
            src_b = and_match.group(3).strip()
            self._pending_and = {
                'dest': dest, 'src_a': src_a, 'src_b': src_b,
                'line': line, 'line_num': line_num
            }

        elif not_match and self._pending_and:
            out_reg = not_match.group(1).strip()
            src = not_match.group(2).strip()
            if src == self._pending_and['dest']:
                self._add_op(
                    out_reg=out_reg,
                    in_a=self._pending_and['src_a'],
                    in_b=self._pending_and['src_b'],
                    line=f"{self._pending_and['line']} ; {line}",
                    line_num=self._pending_and['line_num'],
                    is_and_not=True
                )
            self._pending_and = None
        else:
            if not and_match:
                self._pending_and = None


class MIPSASMParser(ASMParserBase):
    """
    Parse MIPS assembly for NAND patterns.

    Pattern: and $at, a, b → nor out, $at, $zero
    """

    arch_name = "mips"

    def __init__(self):
        super().__init__()
        self._pending_and: Optional[Dict] = None

    def _parse_line(self, line: str, line_num: int) -> None:
        and_match = re.match(r'\s*and\s+(\$\w+),\s*(\$\w+),\s*(\$\w+)', line, re.IGNORECASE)
        nor_match = re.match(r'\s*nor\s+(\$\w+),\s*(\$\w+),\s*(\$\w+)', line, re.IGNORECASE)

        if and_match:
            dest = and_match.group(1).strip()
            src_a = and_match.group(2).strip()
            src_b = and_match.group(3).strip()
            self._pending_and = {
                'dest': dest, 'src_a': src_a, 'src_b': src_b,
                'line': line, 'line_num': line_num
            }

        elif nor_match and self._pending_and:
            out_reg = nor_match.group(1).strip()
            src = nor_match.group(2).strip()
            zero_reg = nor_match.group(3).strip()
            if src == self._pending_and['dest'] and '$zero' in zero_reg:
                self._add_op(
                    out_reg=out_reg,
                    in_a=self._pending_and['src_a'],
                    in_b=self._pending_and['src_b'],
                    line=f"{self._pending_and['line']} ; {line}",
                    line_num=self._pending_and['line_num'],
                    is_and_nor=True
                )
            self._pending_and = None
        else:
            if not and_match:
                self._pending_and = None


class WASMParser(ASMParserBase):
    """
    Parse WebAssembly (WAT format) for NAND patterns.

    Pattern:
        local.get a
        local.get b
        i32.and
        i32.const -1
        i32.xor       ;; NOT = XOR with -1
        local.set out
    """

    arch_name = "wasm"

    def __init__(self):
        super().__init__()
        self._stack: List[str] = []
        self._in_nand = False
        self._nand_inputs: List[str] = []
        self._nand_out: Optional[str] = None

    def _parse_line(self, line: str, line_num: int) -> None:
        stripped = line.strip().rstrip(')')

        get_match = re.match(r'\s*local\.get\s+(\$\w+)', stripped)
        and_match = re.match(r'\s*i32\.and', stripped)
        const_match = re.match(r'\s*i32\.const\s+(-?\d+)', stripped)
        xor_match = re.match(r'\s*i32\.xor', stripped)
        set_match = re.match(r'\s*local\.set\s+(\$\w+)', stripped)

        if get_match:
            var = get_match.group(1)
            self._stack.append(var)
            if not self._in_nand:
                self._nand_inputs.append(var)

        elif and_match and len(self._stack) >= 2:
            self._in_nand = True
            # Pop two, push AND result
            b = self._stack.pop()
            a = self._stack.pop()
            self._stack.append(f"_and_{a}_{b}")
            self._nand_inputs = [a, b]

        elif const_match:
            val = const_match.group(1)
            self._stack.append(f"_const_{val}")

        elif xor_match and len(self._stack) >= 2:
            b = self._stack.pop()
            a = self._stack.pop()
            # XOR with -1 is NOT
            if '_const_-1' in (a, b):
                and_result = b if '_const_-1' in a else a
                if self._in_nand and len(self._nand_inputs) == 2:
                    self._stack.append(f"_nand_{self._nand_inputs[0]}_{self._nand_inputs[1]}")
                    self._in_nand = False
                else:
                    self._stack.append(f"_not_{and_result}")

        elif set_match and self._stack:
            out_var = set_match.group(1)
            result = self._stack.pop()
            if result.startswith("_nand_"):
                parts = result[6:].split("_")
                if len(parts) >= 2:
                    # Remove _and_ or _const_ prefixes
                    in_a = parts[0]
                    in_b = parts[1]
                    self._add_op(out_var, in_a, in_b, line, line_num, is_wasm=True)


# ─── Parser Registry ─────────────────────────────────────────────────────────

PARSERS = {
    "x86": X86ASMParser,
    "x86_64": X86ASMParser,
    "arm": ARMASMParser,
    "arm64": ARMASMParser,
    "riscv": RISCVASMParser,
    "riscv64": RISCVASMParser,
    "mips": MIPSASMParser,
    "wasm": WASMParser,
    "webassembly": WASMParser,
}


# ─── NAND Circuit Reconstruction ──────────────────────────────────────────────

def _is_constant_reg(reg_name: str) -> Optional[int]:
    """Check if a register/label represents a known constant. Returns wire ID or None."""
    # Constants from NAND circuit conventions
    name = reg_name.lower().lstrip('$%')
    if name in ('0', 'zero', '$zero', '$0', '1\'b0'):
        return -1  # Constant 0
    if name in ('1', 'one', '$1', '1\'b1'):
        return -2  # Constant 1
    if name.startswith('_const_'):
        val_str = name[7:]
        try:
            val = int(val_str)
            if val == 0:
                return -1
            if val == 1 or val == -1:
                return -2  # -1 in XOR context = all 1s
        except ValueError:
            pass
    return None


class NANDCircuitReconstructor:
    """
    Reconstruct a NANDCircuit from a list of parsed NAND operations.

    Maps register/variable names to wire IDs and builds the circuit DAG.
    """

    def __init__(self, num_inputs: int = 2):
        self.num_inputs = num_inputs
        self._reg_to_wire: Dict[str, int] = {}
        self._wire_to_reg: Dict[int, str] = {}
        self._next_wire_id = num_inputs

        # Pre-map input registers
        for i in range(num_inputs):
            reg_name = f"in{i}"
            self._reg_to_wire[reg_name] = i
            self._wire_to_reg[i] = reg_name

        # Map constant registers
        self._reg_to_wire["0"] = -1
        self._reg_to_wire["1"] = -2
        self._reg_to_wire["$0"] = -1
        self._reg_to_wire["$zero"] = -1

    def _get_wire_id(self, reg_name: str) -> int:
        """Get or create a wire ID for a register/variable."""
        if reg_name in self._reg_to_wire:
            return self._reg_to_wire[reg_name]

        # Check for constant
        const_wire = _is_constant_reg(reg_name)
        if const_wire is not None:
            self._reg_to_wire[reg_name] = const_wire
            return const_wire

        # Allocate new wire
        wire_id = self._next_wire_id
        self._next_wire_id += 1
        self._reg_to_wire[reg_name] = wire_id
        self._wire_to_reg[wire_id] = reg_name
        return wire_id

    def reconstruct(self, ops: List[ParsedNANDOp],
                    detect_inputs: bool = True) -> DecompileResult:
        """
        Build a NANDCircuit from parsed NAND operations.

        Args:
            ops: List of parsed NAND operations from assembly
            detect_inputs: If True, automatically detect input wires

        Returns:
            DecompileResult with the reconstructed circuit and metadata
        """
        if detect_inputs:
            self._auto_detect_inputs(ops)

        circuit = NANDCircuit(num_inputs=self.num_inputs)

        # Process each NAND operation
        for op in ops:
            wire_a = self._get_wire_id(op.input_a_reg)
            wire_b = self._get_wire_id(op.input_b_reg)
            output_wire = circuit.add_gate(wire_a, wire_b)
            self._reg_to_wire[op.output_reg] = output_wire
            self._wire_to_reg[output_wire] = op.output_reg

        # Determine output wires
        # The last operation's output is typically the circuit output
        if ops:
            last_out_reg = ops[-1].output_reg
            if last_out_reg in self._reg_to_wire:
                circuit.output_wires = [self._reg_to_wire[last_out_reg]]
            elif circuit.gates:
                circuit.output_wires = [circuit.gates[-1].output]
        else:
            circuit.output_wires = []

        # Store metadata for reverse pipeline
        circuit.metadata = {
            "decompiled_from": "assembly",
            "arch": ops[0].arch if ops else "unknown",
            "register_map": dict(self._reg_to_wire),
            "wire_to_reg": dict(self._wire_to_reg),
            "num_ops_parsed": len(ops),
        }

        return DecompileResult(
            circuit=circuit,
            arch=ops[0].arch if ops else "unknown",
            gates_parsed=len(ops),
            gates_reconstructed=circuit.gate_count(),
            register_map=dict(self._reg_to_wire),
            metadata={
                "decompile_success": True,
                "auto_detected_inputs": detect_inputs,
                "num_inputs": self.num_inputs,
            }
        )

    def _auto_detect_inputs(self, ops: List[ParsedNANDOp]) -> None:
        """Automatically detect which registers are circuit inputs."""
        defined_regs: Set[str] = set()
        used_regs: Set[str] = set()

        for op in ops:
            defined_regs.add(op.output_reg)
            used_regs.add(op.input_a_reg)
            used_regs.add(op.input_b_reg)

        # Input registers are those used but never defined
        input_regs = used_regs - defined_regs

        # Filter out constants
        input_regs = {
            r for r in input_regs
            if _is_constant_reg(r) is None
        }

        # Assign wire IDs for detected inputs
        sorted_inputs = sorted(input_regs)
        self.num_inputs = max(self.num_inputs, len(sorted_inputs))

        # Reset wire mapping for detected inputs
        self._reg_to_wire = {}
        self._wire_to_reg = {}
        self._next_wire_id = len(sorted_inputs)

        for i, reg in enumerate(sorted_inputs):
            self._reg_to_wire[reg] = i
            self._wire_to_reg[i] = reg

        # Map constants
        self._reg_to_wire["0"] = -1
        self._reg_to_wire["1"] = -2

        self.num_inputs = len(sorted_inputs)


# ─── Main Decompilation API ──────────────────────────────────────────────────

def decompile_asm(asm_code: str, arch: str = "x86",
                  num_inputs: int = 2) -> DecompileResult:
    """
    Decompile assembly code to a NAND circuit.

    This is the reverse of compile_nand_to_asm. Given assembly code
    generated by the EML pipeline, reconstruct the NAND circuit.

    Args:
        asm_code: Assembly source code
        arch: Target architecture name
        num_inputs: Expected number of input wires (0 = auto-detect)

    Returns:
        DecompileResult with the reconstructed NAND circuit

    Example:
        >>> from eml_pipeline.assembly.asm_decompiler import decompile_asm
        >>> result = decompile_asm(asm_code, "x86")
        >>> circuit = result.circuit
        >>> circuit.gate_count()
    """
    parser_cls = PARSERS.get(arch.lower())
    if parser_cls is None:
        raise ValueError(
            f"Unsupported architecture: {arch}. "
            f"Available: {list(PARSERS.keys())}"
        )

    parser = parser_cls()
    ops = parser.parse(asm_code)

    reconstructor = NANDCircuitReconstructor(
        num_inputs=num_inputs if num_inputs > 0 else 2
    )
    return reconstructor.reconstruct(ops, detect_inputs=True)


def decompile_with_metadata(asm_code: str, arch: str = "x86",
                            forward_metadata: Dict = None) -> DecompileResult:
    """
    Decompile assembly with forward pipeline metadata for enhanced reconstruction.

    When forward pipeline metadata is available (from the original forward pass),
    it can be used to:
    - Map register names to original wire IDs
    - Determine the correct number of inputs
    - Restore variable names for EML reconstruction

    Args:
        asm_code: Assembly source code
        arch: Target architecture name
        forward_metadata: Metadata from the forward compilation pass

    Returns:
        DecompileResult with enhanced metadata
    """
    meta = forward_metadata or {}

    # Extract information from forward metadata
    num_inputs = meta.get("circuit_gates", meta.get("num_inputs", 2))
    if isinstance(num_inputs, dict):
        num_inputs = 2

    result = decompile_asm(asm_code, arch, num_inputs=0)

    # Enhance with forward metadata
    result.metadata["forward_metadata_available"] = bool(meta)
    result.metadata["forward_arch"] = meta.get("arch", arch)

    # If we have the original circuit metadata, verify gate count
    if meta and "circuit_gates" in meta:
        expected_gates = meta["circuit_gates"]
        actual_gates = result.circuit.gate_count()
        result.metadata["gate_count_match"] = (expected_gates == actual_gates)
        result.metadata["gate_count_expected"] = expected_gates
        result.metadata["gate_count_actual"] = actual_gates

    return result
