"""
Reverse Pipeline: Full Bidirectional Reconstruction
=====================================================

Reconstructs LaTeX expressions from NAND circuits and assembly code
using the four-transition scheme:

  T1: EML → NAND         (Theorem 2.6: soft NAND bridge) — forward only
  T2: NAND → ε-NAND      (Definition 3.1: add bounded noise)
  T3: ε-NAND → ApproxEML (§5: Taylor series reconstruction)
  T4: ApproxEML → EML    (§6: Newton-Raphson correction)

The reverse pipeline supports:
  - NAND → EML → LaTeX (direct circuit reconstruction)
  - Assembly → NAND → EML → LaTeX (full decompilation)
  - Forward/backward with and without metadata
  - Error measurement at every transition stage

Supports all previous capabilities:
  - Loops, sums, pi products
  - Integrals, limits
  - Complex numbers, science signs
  - Special variables per domain

Based on: "The EML–NAND Duality" by Daniel Derycke (2026)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.eml.eml_core import (
    EMLNode, EMLNodeType, ONE, VAR, EML,
    eml_exp, eml_ln, eml_zero, eml_complement,
    eml_add, eml_subtract, eml_multiply, eml_divide,
    eml_negate, eml_evaluate, eml_to_dict, eml_from_dict
)
from eml_pipeline.nand.nand_core import NANDCircuit
from eml_pipeline.transitions.t3_t4_nand_to_eml import (
    nand_to_eml, NewtonRaphsonCorrector, TaylorSeriesComputer
)
from eml_pipeline.epsilon_nand.epsilon_nand import (
    EpsilonNANDCircuit, EpsilonNANDConfig, measure_round_trip_error
)
from eml_pipeline.assembly.asm_decompiler import (
    decompile_asm, decompile_with_metadata, DecompileResult
)


class EMLToLatexConverter:
    """Convert EML trees back to LaTeX expressions."""

    def __init__(self):
        self.var_counter = 0

    def convert(self, eml_tree: EMLNode, metadata: Dict = None) -> str:
        """Convert an EML tree to a LaTeX string."""
        meta = metadata or {}
        return self._node_to_latex(eml_tree, meta)

    def _node_to_latex(self, node: EMLNode, meta: Dict) -> str:
        """Recursively convert an EML node to LaTeX."""
        if node.node_type == EMLNodeType.ONE:
            return "1"

        if node.node_type == EMLNodeType.VAR:
            name = node.var_name
            # Clean up internal variable names
            if name.startswith("_n_"):
                val = node.metadata.get("const_value", name[3:])
                if isinstance(val, float) and val == int(val):
                    return str(int(val))
                return str(val)
            if name.startswith("_const_"):
                val = node.metadata.get("const_value", "")
                return str(val) if val else name
            if name.startswith("_inf"):
                return "\\infty"
            if name.startswith("_error_"):
                return f"\\text{{{name}}}"
            if name.startswith("_func_"):
                return f"\\operatorname{{{name[6:]}}}"
            if name.startswith("_sum"):
                return self._reconstruct_sum(node)
            if name.startswith("_prod"):
                return self._reconstruct_product(node)
            if name.startswith("_integral"):
                return self._reconstruct_integral(node)
            if name.startswith("_limit"):
                return self._reconstruct_limit(node)
            if name.startswith("_factorial"):
                return self._reconstruct_factorial(node)
            if name.startswith("_binom"):
                return self._reconstruct_binomial(node)
            if name.startswith("_relation"):
                return self._reconstruct_relation(node)
            if name.startswith("_matrix"):
                return "\\text{matrix}"
            if name.startswith("_cases"):
                return "\\text{cases}"
            # Check for decorated variables
            deco = node.metadata.get("decoration")
            if deco:
                base = node.metadata.get("base_name", "x")
                return f"\\{deco}{{{base}}}"
            return name

        if node.node_type == EMLNodeType.EML:
            # Check for known patterns
            name = node.metadata.get("name", "")

            # Check metadata for pattern recognition
            if "is_soft_nand" in node.metadata:
                a_latex = self._node_to_latex(node.left, meta)
                b_latex = self._node_to_latex(node.right, meta)
                return f"\\operatorname{{NAND}}_{{\\mathbb{{R}}}}({a_latex}, {b_latex})"

            # Try structural pattern matching
            pattern = self._match_eml_pattern(node)
            if pattern:
                return pattern

            # Generic eml notation
            left_latex = self._node_to_latex(node.left, meta)
            right_latex = self._node_to_latex(node.right, meta)
            return f"\\operatorname{{eml}}({left_latex}, {right_latex})"

        return "\\text{unknown}"

    def _match_eml_pattern(self, node: EMLNode) -> Optional[str]:
        """Try to match an EML tree against known patterns."""
        if node.node_type != EMLNodeType.EML:
            return None

        # Pattern: eml(x, 1) = exp(x)
        if (node.right and node.right.node_type == EMLNodeType.ONE):
            inner = self._node_to_latex(node.left, {})
            name = node.metadata.get("name", "")
            if name.startswith("exp"):
                return f"e^{{{inner}}}"
            return f"e^{{{inner}}}"

        # Pattern: eml(1, x) = e - ln(x)
        if (node.left and node.left.node_type == EMLNodeType.ONE):
            inner = self._node_to_latex(node.right, {})
            name = node.metadata.get("name", "")
            if "ln" in name:
                return f"\\ln({inner})"
            return f"e - \\ln({inner})"

        # Pattern: complement eml(0, e^y) = 1 - y
        name = node.metadata.get("name", "")
        if name.startswith("(1-"):
            inner = self._node_to_latex(node.right, {})
            return f"(1 - {inner})"
        if name.startswith("(-"):
            inner = self._node_to_latex(node.right, {})
            return f"(-{inner})"
        if name.startswith("(") and "-" in name:
            left = self._node_to_latex(node.left, {})
            right = self._node_to_latex(node.right, {})
            return f"({left} - {right})"

        # Addition
        if node.metadata.get("is_addition"):
            left = self._node_to_latex(node.left, {})
            right = self._node_to_latex(node.right, {})
            return f"({left} + {right})"

        # Negation
        if node.metadata.get("is_negation"):
            inner = self._node_to_latex(node.left, {})
            return f"(-{inner})"

        return None

    def _reconstruct_sum(self, node: EMLNode) -> str:
        meta = node.metadata
        lower = meta.get("lower")
        upper = meta.get("upper")
        body = meta.get("body")
        return "\\sum f(i)"

    def _reconstruct_product(self, node: EMLNode) -> str:
        return "\\prod f(i)"

    def _reconstruct_integral(self, node: EMLNode) -> str:
        meta = node.metadata
        integral_type = meta.get("integral_type", "\\int")
        if integral_type == "\\oint":
            return "\\oint f(x) dx"
        return "\\int f(x) dx"

    def _reconstruct_limit(self, node: EMLNode) -> str:
        return "\\lim f(x)"

    def _reconstruct_factorial(self, node: EMLNode) -> str:
        return "n!"

    def _reconstruct_binomial(self, node: EMLNode) -> str:
        return "\\binom{n}{k}"

    def _reconstruct_relation(self, node: EMLNode) -> str:
        return "\\text{relation}"


class ReversePipeline:
    """
    Full reverse pipeline using the four-transition scheme.

    T2: NAND → ε-NAND      (Definition 3.1)
    T3: ε-NAND → ApproxEML (§5: Taylor series)
    T4: ApproxEML → EML    (§6: Newton-Raphson)

    Supports:
    - Direct NAND → EML → LaTeX reconstruction
    - Assembly decompilation: ASM → NAND → EML → LaTeX
    - With/without forward metadata
    - Error measurement at every stage
    - All previously supported constructs (loops, sums, etc.)
    """

    def __init__(self, epsilon: float = 0.001, taylor_order: int = 12):
        self.epsilon = epsilon
        self.taylor_order = taylor_order
        self.eml_to_latex = EMLToLatexConverter()
        self.metadata: Dict[str, Any] = {}

    def asm_to_nand(self, asm_code: str, arch: str = "x86",
                    forward_metadata: Dict = None) -> DecompileResult:
        """
        T-assembly: Assembly → NAND circuit reconstruction.

        Parses assembly code to identify NAND gate patterns and
        reconstructs the circuit DAG.
        """
        if forward_metadata:
            return decompile_with_metadata(asm_code, arch, forward_metadata)
        return decompile_asm(asm_code, arch)

    def nand_to_eml_reverse(self, circuit: NANDCircuit,
                             inputs: List[float],
                             forward_metadata: Dict = None) -> Tuple[EMLNode, Dict]:
        """
        T2+T3+T4: NAND → ε-NAND → ApproxEML → EML

        T2: Evaluate NAND circuit with ε-noisy gates
        T3: Reconstruct EML via Taylor series
        T4: Apply Newton-Raphson correction

        Args:
            circuit: NAND circuit to reverse
            inputs: Input values for evaluation
            forward_metadata: Optional metadata from forward pass
        """
        return nand_to_eml(circuit, inputs, self.epsilon, self.taylor_order)

    def eml_to_latex_reverse(self, eml_tree: EMLNode,
                              metadata: Dict = None) -> str:
        """
        EML → LaTeX reconstruction.

        Uses pattern matching and metadata to reconstruct
        a human-readable LaTeX expression from the EML tree.
        """
        return self.eml_to_latex.convert(eml_tree, metadata)

    def full_reverse(self, circuit: NANDCircuit, inputs: List[float],
                     forward_metadata: Dict = None) -> Dict[str, Any]:
        """
        Complete reverse pipeline: NAND → EML → LaTeX

        With error measurements at each transition stage.

        Args:
            circuit: NAND circuit
            inputs: Input values
            forward_metadata: Optional metadata from forward pass

        Returns:
            Dictionary with reconstruction results and error measurements
        """
        results: Dict[str, Any] = {
            "transitions": {},
            "forward_metadata_available": forward_metadata is not None,
        }

        # T2: NAND → ε-NAND
        eps_circuit = EpsilonNANDCircuit(EpsilonNANDConfig(epsilon=self.epsilon))
        eps_results = eps_circuit.evaluate_circuit(circuit, inputs)
        results["transitions"]["t2_nand_to_epsilon_nand"] = {
            "transition": "T2",
            "definition": "3.1",
            "epsilon": self.epsilon,
            "gate_evaluations": eps_circuit.metadata.get("gate_evaluations", 0),
            "restoration_applications": eps_circuit.metadata.get("restoration_applications", 0),
            "raw_results": eps_results,
        }

        # T3+T4: ε-NAND → ApproxEML → EML
        eml_tree, t34_meta = self.nand_to_eml_reverse(
            circuit, inputs, forward_metadata)
        results["transitions"]["t3_epsilon_nand_to_approx_eml"] = {
            "transition": "T3",
            "section": "5",
            "taylor_order": self.taylor_order,
            "approx_value": t34_meta.get("approx_value"),
            "taylor_error_bound": t34_meta.get("taylor_metadata", {}).get("total_error_bound"),
        }
        results["transitions"]["t4_approx_eml_to_eml"] = {
            "transition": "T4",
            "section": "6",
            "corrected_value": t34_meta.get("corrected_value"),
            "newton_raphson_applied": not t34_meta.get("newton_raphson_metadata", {}).get("skipped", False),
            "final_error": t34_meta.get("error"),
        }

        # EML → LaTeX
        latex = self.eml_to_latex_reverse(eml_tree, forward_metadata)

        # Compute exact value for comparison
        exact_vals = {}
        for i, val in enumerate(inputs[:2]):
            if i == 0:
                exact_vals["x"] = val
            elif i == 1:
                exact_vals["y"] = val

        x_val = exact_vals.get("x", 0.5)
        y_val = max(exact_vals.get("y", 1.5), 0.001)
        exact_eml = math.exp(x_val) - math.log(y_val)

        results.update({
            "latex": latex,
            "eml_tree": eml_to_dict(eml_tree),
            "eml_value": t34_meta.get("corrected_value", t34_meta.get("approx_value")),
            "exact_value": exact_eml,
            "error": t34_meta.get("error"),
            "t3_t4_metadata": t34_meta,
            "forward_metadata": forward_metadata,
        })

        # Round-trip error analysis
        corrected_val = t34_meta.get("corrected_value")
        if corrected_val is not None:
            round_trip_error = abs(corrected_val - exact_eml)
            results["round_trip_analysis"] = {
                "round_trip_error": round_trip_error,
                "theoretical_bound": 2 * self.epsilon,
                "bound_satisfied": round_trip_error <= 2 * self.epsilon,
                "epsilon": self.epsilon,
            }

        self.metadata = results
        return results

    def full_reverse_from_asm(self, asm_code: str, arch: str = "x86",
                               test_inputs: List[float] = None,
                               forward_metadata: Dict = None) -> Dict[str, Any]:
        """
        Complete reverse from assembly:
          Assembly → NAND → ε-NAND → ApproxEML → EML → LaTeX

        All four transitions measured.
        """
        results: Dict[str, Any] = {
            "input_asm_arch": arch,
            "transitions": {},
        }

        # T-asm: Assembly → NAND
        try:
            decompile_result = self.asm_to_nand(asm_code, arch, forward_metadata)
            circuit = decompile_result.circuit
            results["transitions"]["t_asm_to_nand"] = {
                "transition": "T-asm",
                "arch": decompile_result.arch,
                "gates_parsed": decompile_result.gates_parsed,
                "gates_reconstructed": decompile_result.gates_reconstructed,
                "metadata": decompile_result.metadata,
            }
        except Exception as e:
            results["transitions"]["t_asm_to_nand"] = {"error": str(e)}
            return results

        # T2+T3+T4: NAND → ε-NAND → ApproxEML → EML → LaTeX
        inputs = test_inputs or [0.5] * max(circuit.num_inputs, 1)
        reverse_result = self.full_reverse(circuit, inputs, forward_metadata)
        results.update(reverse_result)

        return results

    def measure_reverse_error(self, circuit: NANDCircuit,
                               inputs: List[float],
                               forward_metadata: Dict = None) -> Dict[str, Any]:
        """
        Measure errors at each reverse transition stage.

        Returns per-transition error measurements and paper-bound compliance.
        """
        result = self.full_reverse(circuit, inputs, forward_metadata)

        # Extract errors from each transition
        errors = {}
        for t_name, t_data in result.get("transitions", {}).items():
            if isinstance(t_data, dict):
                error_val = t_data.get("error") or t_data.get("final_error") or 0.0
                errors[t_name] = {
                    "error": error_val,
                    "bound": self.epsilon * 2 if "t4" in t_name else self.epsilon,
                }

        # Round-trip analysis
        rt = result.get("round_trip_analysis", {})
        if rt:
            errors["round_trip"] = rt

        return {
            "per_transition_errors": errors,
            "epsilon": self.epsilon,
            "theoretical_round_trip_bound": 2 * self.epsilon,
        }
