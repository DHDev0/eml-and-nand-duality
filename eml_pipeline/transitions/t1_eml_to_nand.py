"""
T1: EML → NAND Transition (Theorem 2.6)
=========================================

The key bridge: soft NAND 1-ab = eml(0, e^{ab})

This transition converts an EML expression tree into a NAND circuit
by decomposing the EML computation into:
1. The multiplication ab = exp(ln(a) + ln(b))
2. The soft NAND = 1 - ab = eml(0, e^{ab})

At Boolean corners, this extends via continuous limit.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.eml.eml_core import (
    EMLNode, EMLNodeType, ONE, VAR, EML,
    eml_exp, eml_ln, eml_zero, eml_complement,
    eml_multiply, eml_soft_nand, eml_evaluate, eml_to_dict
)
from eml_pipeline.nand.nand_core import (
    NANDCircuit, NANDGate,
    build_not_circuit, build_and_circuit, build_or_circuit,
    build_xor_circuit, build_restoration_circuit,
    build_n_bit_adder, build_half_adder, build_full_adder,
    build_constant_1,
    soft_nand, soft_not, soft_and, soft_or,
    compute_contraction, compute_fixed_point
)


class EMLToNANDConverter:
    """
    Convert EML expression trees to NAND circuits.
    
    Strategy: 
    1. Evaluate the EML tree symbolically to extract the computation graph
    2. Map each EML primitive to its NAND circuit implementation
    3. Use fixed-point encoding for continuous values
    4. Apply Theorem 2.6 for the soft NAND bridge
    """
    
    def __init__(self, bit_width: int = 16, epsilon: float = 0.001):
        self.bit_width = bit_width
        self.epsilon = epsilon
        self.metadata: Dict[str, Any] = {
            "transition": "T1: EML → NAND",
            "theorem": "2.6",
            "bit_width": bit_width,
            "epsilon": epsilon,
        }
    
    def convert(self, eml_tree: EMLNode) -> Tuple[NANDCircuit, Dict]:
        """
        Convert an EML tree to a NAND circuit.
        
        Returns (circuit, metadata) where metadata tracks the
        conversion details for reverse reconstruction.
        """
        # Analyze the EML tree structure
        analysis = self._analyze_eml_tree(eml_tree)
        
        # Build NAND circuit
        num_inputs = len(analysis["variables"]) * self.bit_width
        circuit = NANDCircuit(num_inputs=num_inputs)
        circuit.metadata = {
            "eml_source": eml_to_dict(eml_tree),
            "variables": analysis["variables"],
            "bit_width": self.bit_width,
            "epsilon": self.epsilon,
        }
        
        # Convert based on EML tree pattern
        output_wires = self._convert_eml_node(eml_tree, circuit, {}, analysis)
        
        if isinstance(output_wires, int):
            circuit.output_wires = [output_wires]
        elif isinstance(output_wires, list):
            circuit.output_wires = output_wires
        else:
            circuit.output_wires = []
        
        meta = {
            **self.metadata,
            "circuit_gates": circuit.gate_count(),
            "circuit_depth": circuit.depth(),
            "analysis": analysis,
        }
        
        return circuit, meta
    
    def _analyze_eml_tree(self, node: EMLNode) -> Dict:
        """Analyze EML tree to extract variables, depth, patterns."""
        result = {
            "variables": [],
            "depth": node.depth(),
            "size": node.size(),
            "pattern": None,
        }
        
        # Check for known patterns
        name = node.metadata.get("name", "")
        if "exp" in name:
            result["pattern"] = "exp"
        elif "ln" in name or "log" in name:
            result["pattern"] = "ln"
        elif "sin" in name:
            result["pattern"] = "sin"
        elif "cos" in name:
            result["pattern"] = "cos"
        elif "NAND" in name:
            result["pattern"] = "soft_nand"
        elif "1-" in name:
            result["pattern"] = "complement"
        
        # Collect variables
        for var in node.variables():
            if var not in result["variables"] and not var.startswith("_"):
                result["variables"].append(var)
        
        return result
    
    def _convert_eml_node(self, node: EMLNode, circuit: NANDCircuit,
                          wire_map: Dict[str, int], analysis: Dict) -> int:
        """
        Recursively convert EML nodes to NAND circuit wires.
        
        Returns the output wire ID carrying the computed value.
        For multi-bit values, returns the starting wire of the bit array.
        """
        if node.node_type == EMLNodeType.ONE:
            # Constant 1: use NAND bootstrapping
            # NAND(c, NAND(c,c)) = 1 for Boolean c
            # For simplicity, use input wire 0 as seed
            const_1_wire = circuit.add_gate(-2, -2)  # NAND(1, 1) = 0... 
            # Actually NAND(1,1) = 0, so NOT(NAND(1,1)) = 1
            # AND(1,1) = NAND(NAND(1,1), NAND(1,1)) = 1
            nand_11 = circuit.add_gate(-2, -2)  # NAND(1,1) = 0
            const_1 = circuit.add_gate(nand_11, nand_11)  # NAND(0,0) = 1
            return const_1
        
        if node.node_type == EMLNodeType.VAR:
            var_name = node.var_name
            if var_name in wire_map:
                return wire_map[var_name]
            # Allocate input wires for this variable
            start_wire = len(wire_map) * self.bit_width
            wire_map[var_name] = start_wire
            return start_wire
        
        if node.node_type == EMLNodeType.EML:
            # eml(left, right) = e^left - ln(right)
            # This requires computing exp and ln via NAND arithmetic
            left_wire = self._convert_eml_node(node.left, circuit, wire_map, analysis)
            right_wire = self._convert_eml_node(node.right, circuit, wire_map, analysis)
            
            # The actual computation: eml = exp(left) - ln(right)
            # In NAND circuit, this becomes a sequence of arithmetic operations
            # on fixed-point encoded values
            
            # For the soft NAND pattern specifically:
            name = node.metadata.get("name", "")
            if node.metadata.get("is_soft_nand"):
                # Direct soft NAND construction
                return self._build_soft_nand_circuit(
                    left_wire, right_wire, circuit)
            
            # General EML: build exp circuit and ln circuit
            exp_wire = self._build_exp_circuit(left_wire, circuit)
            ln_wire = self._build_ln_circuit(right_wire, circuit)
            # Subtract: exp - ln
            result_wire = self._build_subtract_circuit(
                exp_wire, ln_wire, circuit)
            return result_wire
        
        return -2  # Default: constant 1
    
    def _build_soft_nand_circuit(self, a_wire: int, b_wire: int, 
                                  circuit: NANDCircuit) -> int:
        """
        Build the soft NAND circuit: 1 - ab
        
        This is the Theorem 2.6 construction.
        NAND_R(a,b) = eml(0, e^{ab}) = 1 - ab
        
        Steps:
        1. Multiply a × b (shift-and-add with NAND gates)
        2. Compute 1 - result (complement)
        """
        # Multiplication via NAND (simplified for demonstration)
        ab_wire = self._build_multiply_circuit(a_wire, b_wire, circuit)
        # Complement: 1 - ab
        # In NAND: NOT(ab) = NAND(ab, 1)
        # But we need 1 - ab (soft NOT), not 1 ⊕ ab
        # For Boolean: 1 - ab = NAND(a,b) when a,b ∈ {0,1}
        # For continuous: need analog subtraction
        complement = circuit.add_gate(ab_wire, -2)  # NAND(ab, 1)
        return complement
    
    def _build_multiply_circuit(self, a_wire: int, b_wire: int,
                                 circuit: NANDCircuit) -> int:
        """Build n-bit multiplier using NAND gates."""
        # Simplified: AND the two wires
        return build_and_circuit(a_wire, b_wire, circuit)
    
    def _build_exp_circuit(self, x_wire: int, circuit: NANDCircuit) -> int:
        """
        Build exp(x) circuit using Taylor series in NAND logic.
        
        exp(x) = 1 + x + x²/2 + x³/6 + ...
        
        For small x near 0, truncated Taylor series suffices.
        """
        # Simplified: pass through (in full impl, this would be
        # a full Taylor series computation in fixed-point arithmetic)
        # For now, identity + constant 1 (first two terms)
        one_wire = circuit.add_gate(-2, -2)  # NAND(1,1) = 0
        one_wire = circuit.add_gate(one_wire, one_wire)  # NOT(0) = 1
        # Add: result = x + 1 (via half-adder pattern)
        s, c = build_half_adder(x_wire, one_wire, circuit)
        return s
    
    def _build_ln_circuit(self, y_wire: int, circuit: NANDCircuit) -> int:
        """
        Build ln(y) circuit using artanh Taylor series in NAND logic.
        
        ln(y) = 2 * Σ (z^(2k+1) / (2k+1))  where z = (y-1)/(y+1)
        """
        # Simplified: for the pipeline, we use a lookup + interpolation
        # In full impl: CORDIC-style computation with NAND arithmetic
        return y_wire  # Placeholder
    
    def _build_subtract_circuit(self, a_wire: int, b_wire: int,
                                 circuit: NANDCircuit) -> int:
        """Build subtraction circuit: a - b using NAND gates."""
        # a - b = a + NOT(b) + 1 (two's complement)
        not_b = build_not_circuit(b_wire, circuit)
        s, c = build_half_adder(a_wire, not_b, circuit)
        # Add carry-in of 1
        one = circuit.add_gate(-2, -2)
        one = circuit.add_gate(one, one)
        result, _ = build_full_adder(s, one, -1, circuit)  # -1 = constant 0
        return result


def eml_to_nand(eml_tree: EMLNode, bit_width: int = 16, 
                epsilon: float = 0.001) -> Tuple[NANDCircuit, Dict]:
    """
    Convert an EML tree to a NAND circuit (T1 transition).
    
    Returns (circuit, metadata).
    """
    converter = EMLToNANDConverter(bit_width=bit_width, epsilon=epsilon)
    return converter.convert(eml_tree)
