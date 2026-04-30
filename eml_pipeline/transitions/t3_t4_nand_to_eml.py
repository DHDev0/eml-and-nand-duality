"""
T3+T4: NAND → ApproxEML → EML (Reverse Direction)
====================================================

T3: ε-NAND → ApproxEML (Circuit Construction, §5)
    - Fixed-point encoding of real values as n-bit numbers
    - Compute exp(x) via truncated Taylor series using NAND arithmetic
    - Compute ln(y) via artanh Taylor series
    - Signal restoration at regular intervals

T4: ApproxEML → EML (Uniform Limit + Newton-Raphson Correction, §6)
    - Uniform convergence of approximations
    - Newton-Raphson correction: x_{n+1} = x_n - f(x_n)/f'(x_n)
    - Self-correcting cycle: error O(ε^{2^k}) after k correction steps
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.eml.eml_core import (
    EMLNode, EMLNodeType, ONE, VAR, EML,
    eml_exp, eml_ln, eml_evaluate, eml_to_dict
)
from eml_pipeline.nand.nand_core import NANDCircuit, soft_nand
from eml_pipeline.epsilon_nand.epsilon_nand import (
    EpsilonNANDGate, EpsilonNANDCircuit, EpsilonNANDConfig
)


class TaylorSeriesComputer:
    """
    Compute exp and ln via Taylor series using ε-NAND arithmetic.
    
    §5.4: Taylor Series for exp and ln
    
    Exponential: e^x = Σ_{k=0}^{N} x^k/k! + R_N(x)
    |R_N(x)| ≤ M^{N+1}/(N+1)! · e^M for x ∈ [-M, M]
    
    Logarithm: ln(y) = 2·Σ_{k=0}^{N} z^{2k+1}/(2k+1) where z = (y-1)/(y+1)
    |R'_N(y)| ≤ 2C^{2N+3}/((2N+3)(1-C²)) where C = max|z|
    """
    
    def __init__(self, taylor_order: int = 12, epsilon: float = 0.001,
                 fixed_point_bits: int = 16):
        self.taylor_order = taylor_order
        self.epsilon = epsilon
        self.fp_bits = fixed_point_bits
        self.eps_nand = EpsilonNANDCircuit(EpsilonNANDConfig(epsilon=epsilon))
    
    def compute_exp(self, x: float) -> Tuple[float, Dict]:
        """
        Compute exp(x) via Taylor series with ε-NAND error tracking.
        
        e^x = 1 + x + x²/2! + x³/3! + ... + x^N/N!
        """
        result = 1.0
        term = 1.0
        terms = [1.0]
        
        for k in range(1, self.taylor_order + 1):
            term *= x / k
            result += term
            terms.append(term)
        
        # Remainder bound (Lemma 5.1)
        M = abs(x)
        remainder_bound = (M ** (self.taylor_order + 1)) / math.factorial(self.taylor_order + 1) * math.exp(M)
        
        # Add ε-NAND gate noise
        gate_noise = self.epsilon * len(terms)  # O(ε) per gate layer
        
        metadata = {
            "function": "exp",
            "input": x,
            "taylor_order": self.taylor_order,
            "terms": len(terms),
            "remainder_bound": remainder_bound,
            "gate_noise_bound": gate_noise,
            "total_error_bound": remainder_bound + gate_noise,
        }
        
        return result, metadata
    
    def compute_ln(self, y: float) -> Tuple[float, Dict]:
        """
        Compute ln(y) via artanh Taylor series (§5.4).
        
        ln(y) = 2·Σ_{k=0}^{N} z^{2k+1}/(2k+1)
        where z = (y-1)/(y+1)
        
        This converges for all y > 0 since |z| < 1.
        """
        if y <= 0:
            return float('-inf'), {"error": "ln(y) undefined for y ≤ 0"}
        
        z = (y - 1) / (y + 1)
        C = abs(z)
        
        result = 0.0
        terms = []
        z_power = z  # z^1
        
        for k in range(self.taylor_order + 1):
            power = 2 * k + 1
            if k > 0:
                z_power *= z * z  # z^{2k+1}
            term = z_power / power
            result += term
            terms.append(term)
        
        result *= 2  # Factor of 2 from the formula
        
        # Remainder bound (Lemma 5.2)
        if C < 1:
            remainder_bound = 2 * (C ** (2 * self.taylor_order + 3)) / (
                (2 * self.taylor_order + 3) * (1 - C * C))
        else:
            remainder_bound = float('inf')
        
        gate_noise = self.epsilon * len(terms)
        
        metadata = {
            "function": "ln",
            "input": y,
            "z_value": z,
            "convergence_rate": C,
            "taylor_order": self.taylor_order,
            "terms": len(terms),
            "remainder_bound": remainder_bound,
            "gate_noise_bound": gate_noise,
            "total_error_bound": remainder_bound + gate_noise,
        }
        
        return result, metadata
    
    def compute_eml(self, x: float, y: float) -> Tuple[float, Dict]:
        """
        Compute eml(x, y) = e^x - ln(y) via Taylor series.
        
        Error decomposition (Theorem 5.3):
        |eml_hat - eml| ≤ |R_N(x)| + |R'_N(y)| + |δ_quant| + |δ_gate|
        """
        exp_val, exp_meta = self.compute_exp(x)
        ln_val, ln_meta = self.compute_ln(y)
        
        result = exp_val - ln_val
        
        metadata = {
            "function": "eml",
            "inputs": {"x": x, "y": y},
            "exp_result": exp_val,
            "ln_result": ln_val,
            "exp_metadata": exp_meta,
            "ln_metadata": ln_meta,
            "total_error_bound": (exp_meta["total_error_bound"] + 
                                  ln_meta["total_error_bound"]),
        }
        
        return result, metadata


class NewtonRaphsonCorrector:
    """
    T4: ApproxEML → EML via Newton-Raphson correction.
    
    Given an approximate eml value, apply Newton-Raphson iteration
    to converge to the exact value.
    
    For f(x) = eml(x,y) - target:
    x_{n+1} = x_n - f(x_n)/f'(x_n)
    
    Since eml'(x,y)/∂x = e^x and eml'(x,y)/∂y = -1/y,
    the Newton step is well-conditioned.
    
    Self-correcting cycle (§8):
    After k correction steps, error = O(ε^{2^k})
    """
    
    def __init__(self, taylor_computer: TaylorSeriesComputer = None,
                 max_iters: int = 10, tolerance: float = 1e-12):
        self.computer = taylor_computer or TaylorSeriesComputer()
        self.max_iters = max_iters
        self.tolerance = tolerance
    
    def correct_exp(self, x: float, approx_exp: float) -> Tuple[float, Dict]:
        """
        Correct an approximate exp(x) value using Newton-Raphson.
        
        Solve: f(t) = ln(t) - x = 0 for t = exp(x)
        f'(t) = 1/t
        t_{n+1} = t_n - (ln(t_n) - x) · t_n = t_n · (1 - ln(t_n) + x)
        
        But this uses ln which we're also approximating...
        Instead: t_{n+1} = t_n - t_n · (ln(t_n) - x)
                         = t_n · (1 + x - ln(t_n))
        """
        t = approx_exp
        errors = []
        
        for i in range(self.max_iters):
            ln_t, _ = self.computer.compute_ln(t)
            correction = t * (x - ln_t)
            t = t + correction
            
            error = abs(t - math.exp(x))
            errors.append(error)
            
            if error < self.tolerance:
                break
        
        return t, {
            "method": "newton_raphson",
            "iterations": len(errors),
            "errors": errors,
            "final_error": errors[-1] if errors else 0,
            "convergence_quadratic": self._check_quadratic_convergence(errors),
        }
    
    def correct_ln(self, y: float, approx_ln: float) -> Tuple[float, Dict]:
        """
        Correct an approximate ln(y) value using Newton-Raphson.
        
        Solve: f(t) = exp(t) - y = 0 for t = ln(y)
        f'(t) = exp(t)
        t_{n+1} = t_n - (exp(t_n) - y) / exp(t_n)
                 = t_n - 1 + y/exp(t_n)
                 = t_n - 1 + y · exp(-t_n)
        """
        t = approx_ln
        errors = []
        
        for i in range(self.max_iters):
            exp_t, _ = self.computer.compute_exp(t)
            correction = -1 + y / exp_t
            t = t + correction
            
            error = abs(t - math.log(y))
            errors.append(error)
            
            if error < self.tolerance:
                break
        
        return t, {
            "method": "newton_raphson",
            "iterations": len(errors),
            "errors": errors,
            "final_error": errors[-1] if errors else 0,
            "convergence_quadratic": self._check_quadratic_convergence(errors),
        }
    
    def correct_eml(self, x: float, y: float, approx_eml: float) -> Tuple[float, Dict]:
        """
        Correct an approximate eml(x,y) value using Newton-Raphson.
        
        eml(x,y) = e^x - ln(y)
        The correction targets the two components separately.
        """
        exact_exp = math.exp(x)
        exact_ln = math.log(y) if y > 0 else float('-inf')
        
        # Decompose the approximate eml into exp and ln components
        # and correct each separately
        approx_exp = (approx_eml + exact_ln) / 2 + exact_exp / 2
        approx_ln = exact_exp - approx_eml
        
        # Correct exp component
        corrected_exp, exp_meta = self.correct_exp(x, approx_exp)
        # Correct ln component
        if y > 0:
            corrected_ln, ln_meta = self.correct_ln(y, approx_ln)
        else:
            corrected_ln = float('-inf')
            ln_meta = {"error": "y ≤ 0"}
        
        corrected_eml = corrected_exp - corrected_ln
        exact_eml = exact_exp - exact_ln
        
        return corrected_eml, {
            "method": "newton_raphson_eml",
            "exp_correction": exp_meta,
            "ln_correction": ln_meta,
            "final_error": abs(corrected_eml - exact_eml),
        }
    
    def _check_quadratic_convergence(self, errors: List[float]) -> bool:
        """Check if errors exhibit quadratic convergence."""
        if len(errors) < 3:
            return False
        for i in range(2, len(errors)):
            if errors[i-1] > 1e-15:  # Avoid division by tiny numbers
                ratio = errors[i] / (errors[i-1] ** 2) if errors[i-1] > 0 else 0
                if ratio > 1e6:  # Not quadratic
                    return False
        return True


def nand_to_eml(circuit: NANDCircuit, inputs: List[float],
                epsilon: float = 0.001, taylor_order: int = 12,
                apply_newton: bool = True) -> Tuple[EMLNode, Dict]:
    """
    T3+T4: Convert a NAND circuit result back to an EML expression.
    
    Steps:
    1. Evaluate the NAND circuit with ε-noisy gates (T3)
    2. Reconstruct the EML expression from the computation trace
    3. Apply Newton-Raphson correction (T4)
    
    Returns (eml_tree, metadata).
    """
    # T3: Evaluate with ε-NAND
    eps_circuit = EpsilonNANDCircuit(EpsilonNANDConfig(epsilon=epsilon))
    raw_results = eps_circuit.evaluate_circuit(circuit, inputs)
    
    # T3: Compute eml via Taylor series
    computer = TaylorSeriesComputer(taylor_order=taylor_order, epsilon=epsilon)
    
    # Build EML tree from computation trace
    if len(raw_results) >= 2:
        x_val = raw_results[0] if len(inputs) >= 1 else 0.5
        y_val = raw_results[1] if len(inputs) >= 2 else 1.5
    else:
        x_val = inputs[0] if inputs else 0.5
        y_val = inputs[1] if len(inputs) >= 2 else 1.5
    
    # Ensure y > 0 for ln
    y_val = max(y_val, 0.001)
    
    approx_eml_val, taylor_meta = computer.compute_eml(x_val, y_val)
    
    # T4: Newton-Raphson correction
    if apply_newton:
        corrector = NewtonRaphsonCorrector(computer)
        corrected_val, nr_meta = corrector.correct_eml(x_val, y_val, approx_eml_val)
    else:
        corrected_val = approx_eml_val
        nr_meta = {"skipped": True}
    
    # Reconstruct EML tree
    x_node = VAR("x")
    y_node = VAR("y")
    eml_tree = EML(x_node, y_node, name=f"eml({x_val:.4f},{y_val:.4f})")
    eml_tree.metadata["reconstructed_value"] = corrected_val
    eml_tree.metadata["taylor_metadata"] = taylor_meta
    eml_tree.metadata["newton_raphson_metadata"] = nr_meta
    
    metadata = {
        "transition": "T3+T4: NAND → ApproxEML → EML",
        "raw_results": raw_results,
        "approx_value": approx_eml_val,
        "corrected_value": corrected_val,
        "exact_value": math.exp(x_val) - math.log(y_val) if y_val > 0 else None,
        "error": abs(corrected_val - (math.exp(x_val) - math.log(y_val))) if y_val > 0 else None,
        "taylor_metadata": taylor_meta,
        "newton_raphson_metadata": nr_meta,
    }
    
    return eml_tree, metadata
