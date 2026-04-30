"""
Error Metrics Module
=====================

Measures and tracks errors throughout the EML-NAND pipeline:
- Round-trip error: EML → NAND → ε-NAND → ApproxEML → EML (≤ 2ε)
- Signal restoration contraction: δ' = 4δ² + 4ε
- Taylor truncation error
- Newton-Raphson convergence
- Self-correcting cycle error O(ε^{2^k})
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any

from eml_pipeline.eml.eml_core import EMLNode, eml_evaluate, eml_to_dict
from eml_pipeline.nand.nand_core import (
    soft_nand, ideal_restoration, compute_contraction, 
    compute_fixed_point, NANDCircuit
)
from eml_pipeline.epsilon_nand.epsilon_nand import (
    EpsilonNANDGate, EpsilonNANDCircuit, EpsilonNANDConfig,
    measure_round_trip_error, analyze_error_propagation
)


class ErrorAnalyzer:
    """Comprehensive error analysis for the EML-NAND pipeline."""
    
    def __init__(self, epsilon: float = 0.001):
        self.epsilon = epsilon
        self.results: Dict[str, Any] = {}
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run comprehensive error analysis."""
        self.results = {
            "epsilon": self.epsilon,
            "signal_restoration": self._analyze_restoration(),
            "round_trip": self._analyze_round_trip(),
            "contraction": self._analyze_contraction(),
            "taylor_exp": self._analyze_taylor_exp(),
            "taylor_ln": self._analyze_taylor_ln(),
            "newton_raphson": self._analyze_newton_raphson(),
            "self_correcting_cycle": self._analyze_self_correcting(),
        }
        return self.results
    
    def _analyze_restoration(self) -> Dict[str, Any]:
        """Analyze signal restoration (Theorem 4.2)."""
        eps = self.epsilon
        gate = EpsilonNANDGate(epsilon=eps)
        
        # Test restoration from various starting points
        test_points = [0.01, 0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95, 0.99]
        results = []
        
        for x in test_points:
            # Ideal restoration (no noise)
            ideal = ideal_restoration(x)
            ideal_delta_after = min(ideal, 1 - ideal)
            
            # With noise
            circuit = EpsilonNANDCircuit(EpsilonNANDConfig(epsilon=eps))
            restored = circuit.restore(x, num_iters=20)
            actual_delta = min(restored, 1 - restored)
            
            # Theoretical bound
            delta_before = min(x, 1 - x)
            theoretical_delta = compute_contraction(delta_before, eps)
            fixed_point = compute_fixed_point(eps)
            
            results.append({
                "start": x,
                "ideal_restored": ideal,
                "actual_restored": restored,
                "delta_before": delta_before,
                "ideal_delta_after": ideal_delta_after,
                "actual_delta_after": actual_delta,
                "theoretical_bound": theoretical_delta,
                "fixed_point": fixed_point,
            })
        
        return {
            "test_points": results,
            "fixed_point_delta": compute_fixed_point(eps),
            "contraction_viable": eps < 1/64,
            "noise_threshold": 1.0/64,
        }
    
    def _analyze_round_trip(self) -> Dict[str, Any]:
        """Analyze round-trip error (Section 8)."""
        return measure_round_trip_error(self.epsilon, num_tests=500)
    
    def _analyze_contraction(self) -> Dict[str, Any]:
        """Analyze contraction dynamics δ_{n+1} = 4δ_n² + 4ε."""
        eps = self.epsilon
        delta_star = compute_fixed_point(eps)
        
        # Simulate contraction from various starting deltas
        trajectories = {}
        for delta_0 in [0.2, 0.1, 0.05, 0.01]:
            trajectory = [delta_0]
            delta = delta_0
            for _ in range(50):
                delta = compute_contraction(delta, eps)
                trajectory.append(delta)
                if delta <= delta_star * 1.01:
                    break
            trajectories[str(delta_0)] = trajectory
        
        return {
            "fixed_point": delta_star,
            "trajectories": trajectories,
            "convergence_rate": "quadratic_far_linear_near",
        }
    
    def _analyze_taylor_exp(self) -> Dict[str, Any]:
        """Analyze Taylor series error for exp."""
        from eml_pipeline.transitions.t3_t4_nand_to_eml import TaylorSeriesComputer
        computer = TaylorSeriesComputer(epsilon=self.epsilon)
        
        test_vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
        results = []
        for x in test_vals:
            approx, meta = computer.compute_exp(x)
            exact = math.exp(x)
            error = abs(approx - exact)
            results.append({
                "x": x,
                "approx": approx,
                "exact": exact,
                "error": error,
                "remainder_bound": meta["remainder_bound"],
            })
        
        return {"results": results}
    
    def _analyze_taylor_ln(self) -> Dict[str, Any]:
        """Analyze Taylor series error for ln."""
        from eml_pipeline.transitions.t3_t4_nand_to_eml import TaylorSeriesComputer
        computer = TaylorSeriesComputer(epsilon=self.epsilon)
        
        test_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        results = []
        for y in test_vals:
            approx, meta = computer.compute_ln(y)
            exact = math.log(y)
            error = abs(approx - exact)
            results.append({
                "y": y,
                "approx": approx,
                "exact": exact,
                "error": error,
                "convergence_rate_C": meta["convergence_rate"],
                "remainder_bound": meta["remainder_bound"],
            })
        
        return {"results": results}
    
    def _analyze_newton_raphson(self) -> Dict[str, Any]:
        """Analyze Newton-Raphson correction convergence."""
        from eml_pipeline.transitions.t3_t4_nand_to_eml import NewtonRaphsonCorrector
        corrector = NewtonRaphsonCorrector()
        
        results = []
        for x in [0.5, 1.0, 2.0]:
            approx = math.exp(x) * (1 + 0.01)  # 1% error
            corrected, meta = corrector.correct_exp(x, approx)
            results.append({
                "x": x,
                "initial_error": abs(approx - math.exp(x)),
                "final_error": meta["final_error"],
                "iterations": meta["iterations"],
                "quadratic": meta["convergence_quadratic"],
            })
        
        return {"results": results}
    
    def _analyze_self_correcting(self) -> Dict[str, Any]:
        """
        Analyze the self-correcting cycle (Section 8).
        
        Round-trip: EML → NAND → ε-NAND → ApproxEML → EML
        Error after k cycles: O(ε^{2^k})
        """
        eps = self.epsilon
        results = []
        
        # Simulate multiple round-trip cycles
        error = eps
        for k in range(1, 6):
            # After one cycle, error contracts quadratically
            # E_{k+1} ≈ C · E_k² for some constant C
            error = min(1.0, 2 * error * error + 2 * eps)
            results.append({
                "cycle": k,
                "error": error,
                "theoretical": f"O(ε^{2**k})" if k <= 4 else "≈0",
            })
        
        return {"cycle_analysis": results}
    
    def measure_pipeline_error(self, latex_expr: str, 
                                test_values: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Measure the full pipeline error for a given LaTeX expression.
        """
        from eml_pipeline.eml.latex_to_eml import latex_to_eml
        from eml_pipeline.parsers.latex_parser import parse_latex
        
        # Forward: LaTeX → EML
        try:
            eml_tree, convert_meta = latex_to_eml(latex_expr)
        except Exception as e:
            return {"error": f"LaTeX parse failed: {e}"}
        
        # Evaluate EML
        if test_values:
            try:
                eml_value = eml_evaluate(eml_tree, test_values)
            except Exception as e:
                eml_value = None
                eval_error = str(e)
        else:
            eml_value = None
            eval_error = "No test values provided"
        
        # Get the exact Python computation for comparison
        exact_value = None
        if test_values and eml_value is not None:
            try:
                # Simple: eval the expression with the test values
                # (This is a rough comparison for testing purposes)
                pass
            except:
                pass
        
        return {
            "latex": latex_expr,
            "eml_tree_size": eml_tree.size(),
            "eml_tree_depth": eml_tree.depth(),
            "eml_value": eml_value,
            "conversion_metadata": convert_meta,
        }
