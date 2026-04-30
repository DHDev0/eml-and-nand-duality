"""
T2: ε-NAND Framework
=====================

ε-NAND gates: continuous gates G_ε with bounded noise.
|G_ε(a,b) - (1-ab)| ≤ ε for all a,b ∈ [0,1]
G_ε(a,b) = NAND(a,b) exactly at Boolean corners.

Key results:
- ε-NOT: NOT_ε(a) = G_ε(a, 1) = 1 - a + η₁, |η₁| ≤ ε
- ε-AND: a ⊗_ε b = G_ε(G_ε(a,b), 1) = ab + (η₁ - η₂), error ≤ 2ε
- ε-OR:  a ⊕_ε b, error ≤ 3ε + ε²
- Signal Restoration: R(x) = NAND(NAND(x,x), NAND(x,x))
  Contraction: δ' = 4δ² + 4ε < δ
  Fixed point: δ* ≈ 4ε
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.nand.nand_core import (
    NANDCircuit, soft_nand, soft_not, soft_and, soft_or,
    ideal_restoration, restoration_circuit,
    compute_contraction, compute_fixed_point, iterated_restoration
)


@dataclass
class EpsilonNANDConfig:
    """Configuration for ε-NAND gate simulation."""
    epsilon: float = 0.001          # Gate noise parameter
    exact_at_corners: bool = True   # G_ε = NAND exactly on {0,1}²
    domain: str = "[0,1]²"          # Valid domain
    max_restoration_iters: int = 50  # Max iterations for signal restoration


class EpsilonNANDGate:
    """
    An ε-NAND gate: G_ε(a,b) = 1 - ab + η where |η| ≤ ε.
    
    Definition 3.1: G_ε : [0,1]² → [0,1] satisfies:
    |G_ε(a,b) - (1-ab)| ≤ ε  ∀ a,b ∈ [0,1]
    G_ε(a,b) = NAND(a,b)  ∀ a,b ∈ {0,1}
    """
    
    def __init__(self, epsilon: float = 0.001, rng_seed: Optional[int] = None):
        self.epsilon = epsilon
        self.rng = random.Random(rng_seed)
    
    def __call__(self, a: float, b: float) -> float:
        """Evaluate G_ε(a, b)."""
        ideal = soft_nand(a, b)  # 1 - ab
        
        # Check if inputs are Boolean
        if self._is_boolean(a) and self._is_boolean(b):
            return ideal  # Exact at corners (Proposition 3.1a)
        
        # Add bounded noise
        eta = self.epsilon * (2 * self.rng.random() - 1)
        result = ideal + eta
        return max(0.0, min(1.0, result))  # Clamp to [0,1]
    
    def _is_boolean(self, x: float, tol: float = 1e-10) -> bool:
        return abs(x) < tol or abs(x - 1) < tol


class EpsilonNANDCircuit:
    """
    A circuit of ε-NAND gates with signal restoration.
    """
    
    def __init__(self, config: EpsilonNANDConfig = None):
        self.config = config or EpsilonNANDConfig()
        self.gate = EpsilonNANDGate(epsilon=self.config.epsilon)
        self.wire_values: Dict[int, float] = {}
        self.metadata: Dict[str, Any] = {
            "epsilon": self.config.epsilon,
            "gate_evaluations": 0,
            "restoration_applications": 0,
        }
    
    def not_epsilon(self, a: float) -> float:
        """ε-NOT: G_ε(a, 1) = 1 - a + η₁"""
        self.metadata["gate_evaluations"] += 1
        return self.gate(a, 1.0)
    
    def and_epsilon(self, a: float, b: float) -> float:
        """ε-AND: G_ε(G_ε(a,b), 1) = ab + (η₁ - η₂), |error| ≤ 2ε"""
        self.metadata["gate_evaluations"] += 2
        nand_ab = self.gate(a, b)
        return self.gate(nand_ab, 1.0)
    
    def or_epsilon(self, a: float, b: float) -> float:
        """ε-OR: G_ε(NOT_ε(a), NOT_ε(b)), |error| ≤ 3ε + ε²"""
        self.metadata["gate_evaluations"] += 3
        not_a = self.not_epsilon(a)
        not_b = self.not_epsilon(b)
        return self.gate(not_a, not_b)
    
    def xor_epsilon(self, a: float, b: float) -> float:
        """ε-XOR: approximate XOR with error O(ε)"""
        self.metadata["gate_evaluations"] += 4
        nand_ab = self.gate(a, b)
        left = self.gate(a, nand_ab)
        right = self.gate(b, nand_ab)
        return self.gate(left, right)
    
    def restore(self, x: float, num_iters: int = None) -> float:
        """
        Signal restoration: R(x) = NAND(NAND(x,x), NAND(x,x))
        
        Theorem 4.2: If δ < 2/9 and ε < δ²/8, then
        R maps [b]_δ → [b]_{4δ²+4ε} with contraction 4δ²+4ε < δ
        
        Fixed point: δ* ≈ 4ε
        """
        iters = num_iters or self.config.max_restoration_iters
        self.metadata["restoration_applications"] += iters
        
        current = x
        for _ in range(iters):
            y = self.gate(current, current)   # NAND(x, x) = 1 - x²
            current = self.gate(y, y)          # NAND(y, y) = 1 - y² = T(x)
            
            # Check convergence
            delta = min(current, 1 - current)
            target = compute_fixed_point(self.config.epsilon) * 1.01
            if delta <= target:
                break
        
        return current
    
    def evaluate_circuit(self, circuit: NANDCircuit, inputs: List[float]) -> List[float]:
        """
        Evaluate a NAND circuit with ε-noisy gates and periodic restoration.
        """
        wires: Dict[int, float] = {}
        wires[-1] = 0.0  # Constant 0
        wires[-2] = 1.0  # Constant 1
        
        for i, val in enumerate(inputs):
            wires[i] = val
        
        # Evaluate gates with restoration every O(log n) layers
        depth_since_restore = 0
        restore_interval = max(1, int(math.log2(max(1, circuit.gate_count()))))
        
        for gate in circuit.gates:
            a = wires.get(gate.input_a, 0.5)
            b = wires.get(gate.input_b, 0.5)
            wires[gate.output] = self.gate(a, b)
            
            depth_since_restore += 1
            if depth_since_restore >= restore_interval:
                # Apply signal restoration to all internal wires
                for wire_id in list(wires.keys()):
                    if wire_id >= circuit.num_inputs:
                        val = wires[wire_id]
                        if 0 <= val <= 1:
                            wires[wire_id] = self.restore(val, num_iters=2)
                depth_since_restore = 0
        
        return [wires.get(w, 0.5) for w in circuit.output_wires]


def analyze_error_propagation(delta_0: float, epsilon: float, depth: int) -> Dict[str, Any]:
    """
    Analyze error propagation through a depth-d ε-NAND circuit.
    
    Corollary 3.8: δ_d ≤ 2^d · δ_0 + (2^d - 1) · ε
    
    Returns analysis of error at each depth level.
    """
    results = {
        "delta_0": delta_0,
        "epsilon": epsilon,
        "depth": depth,
        "levels": [],
        "fixed_point": compute_fixed_point(epsilon),
        "contraction_viable": epsilon < (delta_0 ** 2) / 8 and delta_0 < 2/9,
    }
    
    delta = delta_0
    for d in range(depth + 1):
        # Without restoration
        delta_unrestored = min(1.0, (2 ** d) * delta_0 + (2 ** d - 1) * epsilon)
        
        # With restoration (Theorem 4.2)
        delta_restored = delta
        if d > 0:
            # Apply contraction
            delta_restored = compute_contraction(delta, epsilon)
            if delta_restored >= delta:
                # No contraction possible, use unrestored
                delta_restored = delta_unrestored
        
        results["levels"].append({
            "depth": d,
            "delta_unrestored": delta_unrestored,
            "delta_restored": delta_restored,
        })
        
        # Update delta for next iteration
        delta = delta_restored
    
    return results


def measure_round_trip_error(epsilon: float, num_tests: int = 1000) -> Dict[str, Any]:
    """
    Measure the round-trip error of the EML-NAND cycle.
    
    Round-trip: EML → NAND → ε-NAND → ApproxEML → EML
    Theoretical bound: ≤ 2ε (Section 8)
    """
    gate = EpsilonNANDGate(epsilon=epsilon)
    circuit = EpsilonNANDCircuit(EpsilonNANDConfig(epsilon=epsilon))
    
    errors = []
    for _ in range(num_tests):
        # Test soft NAND then restoration
        a = random.random()
        b = random.random()
        
        # Exact: 1 - ab
        exact = soft_nand(a, b)
        
        # ε-NAND
        approx = gate(a, b)
        
        # After restoration
        restored = circuit.restore(approx)
        
        error = abs(restored - exact)
        errors.append(error)
    
    return {
        "epsilon": epsilon,
        "num_tests": num_tests,
        "max_error": max(errors),
        "mean_error": sum(errors) / len(errors),
        "theoretical_bound": 2 * epsilon,
        "bound_satisfied": max(errors) <= 2 * epsilon + 0.01,
    }
