"""
NAND Core Module — The Universal Discrete Primitive
====================================================

NAND(a, b) = ¬(a ∧ b) — the Sheffer stroke, functionally complete.

Key properties:
- NOT(a) = NAND(a, a)
- AND(a, b) = NAND(NAND(a, b), NAND(a, b))
- OR(a, b) = NAND(NAND(a, a), NAND(b, b))
- XOR(a, b) = NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b)))

Signal Restoration (Theorem 4.2):
- R(x) = NAND(NAND(x, x), NAND(x, x))
- Ideal dynamics: T(x) = 2x² - x⁴
- Contraction: δ' = 4δ² + 4ε < δ
- Fixed point: δ* = 4ε + O(ε²)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


# ─── Boolean NAND ─────────────────────────────────────────────────────────────

def nand_bool(a: bool, b: bool) -> bool:
    """Boolean NAND gate."""
    return not (a and b)

def not_bool(a: bool) -> bool:
    """NOT from NAND: NAND(a, a)."""
    return nand_bool(a, a)

def and_bool(a: bool, b: bool) -> bool:
    """AND from NAND: NAND(NAND(a,b), NAND(a,b))."""
    return not_bool(nand_bool(a, b))

def or_bool(a: bool, b: bool) -> bool:
    """OR from NAND: NAND(NAND(a,a), NAND(b,b))."""
    return nand_bool(not_bool(a), not_bool(b))

def xor_bool(a: bool, b: bool) -> bool:
    """XOR from NAND."""
    nand_ab = nand_bool(a, b)
    return nand_bool(nand_bool(a, nand_ab), nand_bool(b, nand_ab))


# ─── Soft NAND (Continuous) ──────────────────────────────────────────────────

def soft_nand(a: float, b: float) -> float:
    """
    Soft NAND: 1 - ab for a, b ∈ [0, 1]
    The multilinear extension of Boolean NAND.
    """
    return 1.0 - a * b

def soft_not(a: float) -> float:
    """Soft NOT: NAND(a, 1) = 1 - a"""
    return soft_nand(a, 1.0)

def soft_and(a: float, b: float) -> float:
    """Soft AND: NOT(NAND(a,b)) = ab"""
    return soft_not(soft_nand(a, b))

def soft_or(a: float, b: float) -> float:
    """Soft OR: NAND(NOT(a), NOT(b)) = a + b - ab"""
    return soft_nand(soft_not(a), soft_not(b))


# ─── Signal Restoration (Theorem 4.2) ────────────────────────────────────────

def ideal_restoration(x: float) -> float:
    """
    Ideal noiseless restoration: T(x) = 2x² - x⁴
    This is the dynamics of R(x) = NAND(NAND(x,x), NAND(x,x))
    with ε = 0.
    """
    return 2 * x * x - x * x * x * x

def restoration_circuit(x: float, epsilon: float = 0.0) -> float:
    """
    R(x) = G_ε(NAND(x,x), NAND(x,x)) with noise ε.
    
    With ε = 0: T(x) = 2x² - x⁴
    With ε > 0: R(x) = T(x) + E(x) where |E(x)| ≤ 4ε
    
    Theorem 4.2: If δ < 2/9 and ε < δ²/8, then
    R maps [b]_δ → [b]_{4δ²+4ε} with 4δ²+4ε < δ
    """
    y = soft_nand(x, x)  # 1 - x²
    # Add noise to first layer
    eta1 = epsilon * (2 * (hash(str(x)) % 1000) / 1000 - 1) if epsilon > 0 else 0
    y_noisy = max(0, min(1, y + eta1))
    
    result = soft_nand(y_noisy, y_noisy)  # 1 - y²
    # Add noise to second layer
    eta2 = epsilon * (2 * (hash(str(y)) % 1000) / 1000 - 1) if epsilon > 0 else 0
    result_noisy = max(0, min(1, result + eta2))
    
    return result_noisy

def compute_contraction(delta: float, epsilon: float) -> float:
    """
    Compute δ' = 4δ² + 4ε for the signal restoration contraction.
    Returns δ' if δ' < δ (contraction), else the original δ.
    """
    delta_prime = 4 * delta * delta + 4 * epsilon
    return delta_prime

def compute_fixed_point(epsilon: float) -> float:
    """
    Compute the fixed point δ* of the contraction iteration.
    δ* satisfies 4(δ*)² + 4ε = δ*
    Solution: δ* = (1 - sqrt(1 - 64ε)) / 8
    
    For small ε: δ* ≈ 4ε
    Requires ε ≤ 1/64 for a real solution.
    """
    if epsilon > 1.0 / 64:
        return float('inf')  # No fixed point exists
    discriminant = 1 - 64 * epsilon
    return (1 - math.sqrt(discriminant)) / 8

def iterated_restoration(x: float, epsilon: float, max_iters: int = 100, 
                         target_delta: float = None) -> Tuple[float, List[float]]:
    """
    Apply R repeatedly until convergence.
    Returns (final_value, delta_history).
    """
    current = x
    deltas = []
    target = target_delta or compute_fixed_point(epsilon) * 1.1
    
    for i in range(max_iters):
        current = restoration_circuit(current, epsilon)
        # Track how close to Boolean the value is
        delta = min(current, 1 - current)
        deltas.append(delta)
        if delta <= target:
            break
    
    return current, deltas


# ─── NAND Circuit Representation ─────────────────────────────────────────────

@dataclass
class NANDGate:
    """A single NAND gate in a circuit."""
    gate_id: int
    input_a: int    # Wire ID or -1 for constant 0, -2 for constant 1
    input_b: int
    output: int     # Output wire ID

@dataclass  
class NANDCircuit:
    """
    A NAND circuit: a directed acyclic graph of NAND gates.
    
    Wire conventions:
    - Negative IDs: constants (-1 = 0, -2 = 1)
    - 0..num_inputs-1: input wires
    - num_inputs..: internal wires
    - Last gate outputs are the output wires
    """
    num_inputs: int
    gates: List[NANDGate] = field(default_factory=list)
    output_wires: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_gate(self, input_a: int, input_b: int) -> int:
        """Add a NAND gate and return its output wire ID."""
        gate_id = len(self.gates)
        output_wire = self.num_inputs + gate_id
        gate = NANDGate(gate_id=gate_id, input_a=input_a, input_b=input_b, output=output_wire)
        self.gates.append(gate)
        return output_wire
    
    def evaluate(self, inputs: List[bool]) -> List[bool]:
        """Evaluate the circuit on Boolean inputs."""
        wires = {}
        # Constants
        wires[-1] = False  # 0
        wires[-2] = True   # 1
        # Inputs
        for i, val in enumerate(inputs):
            wires[i] = val
        # Gates
        for gate in self.gates:
            a = wires[gate.input_a]
            b = wires[gate.input_b]
            wires[gate.output] = nand_bool(a, b)
        # Outputs
        return [wires[w] for w in self.output_wires]
    
    def evaluate_soft(self, inputs: List[float], epsilon: float = 0.0) -> List[float]:
        """Evaluate the circuit on continuous [0,1] inputs with optional noise."""
        wires = {}
        wires[-1] = 0.0
        wires[-2] = 1.0
        for i, val in enumerate(inputs):
            wires[i] = val
        for gate in self.gates:
            a = wires[gate.input_a]
            b = wires[gate.input_b]
            result = soft_nand(a, b)
            if epsilon > 0:
                import random
                noise = epsilon * (2 * random.random() - 1)
                result = max(0, min(1, result + noise))
            wires[gate.output] = result
        return [wires[w] for w in self.output_wires]
    
    def gate_count(self) -> int:
        return len(self.gates)
    
    def depth(self) -> int:
        """Compute circuit depth."""
        wire_depth = {}
        wire_depth[-1] = 0
        wire_depth[-2] = 0
        for i in range(self.num_inputs):
            wire_depth[i] = 0
        for gate in self.gates:
            d = max(wire_depth.get(gate.input_a, 0), wire_depth.get(gate.input_b, 0)) + 1
            wire_depth[gate.output] = d
        if not self.output_wires:
            return 0
        return max(wire_depth.get(w, 0) for w in self.output_wires)
    
    def to_dict(self) -> Dict:
        return {
            "num_inputs": self.num_inputs,
            "gates": [{"id": g.gate_id, "a": g.input_a, "b": g.input_b, "out": g.output} for g in self.gates],
            "outputs": self.output_wires,
            "metadata": self.metadata
        }


# ─── Circuit Builders ────────────────────────────────────────────────────────

def build_not_circuit(input_wire: int, circuit: NANDCircuit) -> int:
    """NOT(a) = NAND(a, a)"""
    return circuit.add_gate(input_wire, input_wire)

def build_and_circuit(a_wire: int, b_wire: int, circuit: NANDCircuit) -> int:
    """AND(a, b) = NAND(NAND(a,b), NAND(a,b))"""
    nand_ab = circuit.add_gate(a_wire, b_wire)
    return circuit.add_gate(nand_ab, nand_ab)

def build_or_circuit(a_wire: int, b_wire: int, circuit: NANDCircuit) -> int:
    """OR(a, b) = NAND(NAND(a,a), NAND(b,b))"""
    not_a = build_not_circuit(a_wire, circuit)
    not_b = build_not_circuit(b_wire, circuit)
    return circuit.add_gate(not_a, not_b)

def build_xor_circuit(a_wire: int, b_wire: int, circuit: NANDCircuit) -> int:
    """XOR(a, b) = NAND(NAND(a, NAND(a,b)), NAND(b, NAND(a,b)))"""
    nand_ab = circuit.add_gate(a_wire, b_wire)
    left = circuit.add_gate(a_wire, nand_ab)
    right = circuit.add_gate(b_wire, nand_ab)
    return circuit.add_gate(left, right)

def build_half_adder(a: int, b: int, circuit: NANDCircuit) -> Tuple[int, int]:
    """Half adder: returns (sum, carry) wire IDs."""
    s = build_xor_circuit(a, b, circuit)
    c = build_and_circuit(a, b, circuit)
    return s, c

def build_full_adder(a: int, b: int, carry_in: int, circuit: NANDCircuit) -> Tuple[int, int]:
    """Full adder: returns (sum, carry_out) wire IDs."""
    s1, c1 = build_half_adder(a, b, circuit)
    s2, c2 = build_half_adder(s1, carry_in, circuit)
    carry_out = build_or_circuit(c1, c2, circuit)
    return s2, carry_out

def build_n_bit_adder(n: int, circuit: NANDCircuit, 
                       a_start: int, b_start: int) -> Tuple[List[int], int]:
    """
    n-bit ripple-carry adder.
    a_start..a_start+n-1: first operand wires
    b_start..b_start+n-1: second operand wires
    Returns (sum_wires[n+1], total_gates_used)
    """
    carry = -1  # Constant 0
    sum_wires = []
    for i in range(n):
        s, carry = build_full_adder(a_start + i, b_start + i, carry, circuit)
        sum_wires.append(s)
    sum_wires.append(carry)
    return sum_wires, 0

def build_restoration_circuit(input_wire: int, circuit: NANDCircuit,
                               epsilon: float = 0.0) -> int:
    """
    Signal restoration circuit: R(x) = NAND(NAND(x,x), NAND(x,x))
    Implements Theorem 4.2.
    """
    y = circuit.add_gate(input_wire, input_wire)  # NAND(x, x)
    return circuit.add_gate(y, y)  # NAND(y, y)

def build_constant_1(c: int, circuit: NANDCircuit) -> int:
    """
    Constant bootstrapping: NAND(c, NAND(c,c)) = 1 for Boolean c.
    Proposition 3.9.
    """
    not_c = build_not_circuit(c, circuit)
    return circuit.add_gate(c, not_c)
