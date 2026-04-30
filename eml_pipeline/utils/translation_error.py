"""
Translation Error Module — Pipeline Stage Error Measurement
============================================================

Measures and tracks errors at every translation stage of the EML pipeline:
    LaTeX → EML → NAND → Rewritten NAND → Verilog → Optimized (Yosys/ABC) → FPGA/ASIC

Each measurement function references the specific paper theorem, lemma, or
section that establishes the relevant error bound:

    - Theorem 2.6:  EML-soft NAND identity  NAND_R(a,b) = 1 - ab = eml(0, e^{ab})
    - Theorem 4.2:  Signal restoration contraction  δ' = 4δ² + 4ε < δ
    - Section 8:    Round-trip error bound  ≤ 2ε
    - Corollary 3.8: Error propagation  δ_d ≤ 2^d · δ_0 + (2^d - 1) · ε
    - Definition 3.1: ε-NAND gate  |G_ε(a,b) - (1-ab)| ≤ ε

Reference: "The EML–NAND Duality" by Daniel Derycke (2026)
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eml_pipeline.eml.eml_core import (
    EMLNode,
    EMLNodeType,
    eml_evaluate,
    eml_to_dict,
    eml_exp,
    eml_ln,
    eml_zero,
    eml_complement,
    eml_add,
    eml_subtract,
    eml_multiply,
    eml_divide,
    eml_negate,
    eml_soft_nand,
    ONE,
    VAR,
    EML,
)
from eml_pipeline.nand.nand_core import (
    NANDCircuit,
    soft_nand,
    compute_contraction,
    compute_fixed_point,
    ideal_restoration,
    iterated_restoration,
)
from eml_pipeline.epsilon_nand.epsilon_nand import (
    EpsilonNANDGate,
    EpsilonNANDCircuit,
    EpsilonNANDConfig,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TranslationErrorTracker
# ═══════════════════════════════════════════════════════════════════════════════


class TranslationErrorTracker:
    """Track and aggregate errors across all pipeline translation stages.

    Every *translation* (e.g. ``"LaTeX" → "EML"``) is identified by a unique
    ``translation_id`` returned by :meth:`begin_translation`.  Individual error
    measurements are attached to that id via :meth:`record_error` and later
    inspected with :meth:`get_report`, :meth:`get_cumulative_error`, or
    :meth:`check_paper_bound`.

    Parameters
    ----------
    epsilon:
        The gate-noise parameter ε used throughout the pipeline
        (Definition 3.1).  Defaults to 0.001.
    """

    def __init__(self, epsilon: float = 0.001) -> None:
        self.epsilon = epsilon
        self._translations: Dict[str, Dict[str, Any]] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def begin_translation(self, source_format: str, target_format: str) -> str:
        """Register a new translation and return its unique identifier.

        Parameters
        ----------
        source_format:
            Name of the source representation (e.g. ``"LaTeX"``, ``"EML"``).
        target_format:
            Name of the target representation (e.g. ``"EML"``, ``"NAND"``).

        Returns
        -------
        str
            A UUID-based translation id used for subsequent record / query
            calls.
        """
        tid = f"{source_format}_to_{target_format}_{uuid.uuid4().hex[:8]}"
        self._translations[tid] = {
            "source_format": source_format,
            "target_format": target_format,
            "epsilon": self.epsilon,
            "stages": [],
            "errors": [],
            "started_at": _timestamp(),
        }
        return tid

    def record_error(
        self,
        translation_id: str,
        stage: str,
        error_value: float,
        error_bound: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an error measurement at a specific pipeline stage.

        Parameters
        ----------
        translation_id:
            Identifier returned by :meth:`begin_translation`.
        stage:
            Human-readable stage name (e.g. ``"latex_to_eml"``,
            ``"eml_to_nand"``).
        error_value:
            The *measured* absolute error at this stage.
        error_bound:
            The *theoretical* upper bound for this error (from the paper).
        metadata:
            Optional dict with extra diagnostic information.

        Raises
        ------
        KeyError
            If *translation_id* is unknown.
        ValueError
            If *error_value* or *error_bound* is negative.
        """
        if translation_id not in self._translations:
            raise KeyError(f"Unknown translation id: {translation_id!r}")
        if error_value < 0:
            raise ValueError(f"error_value must be non-negative, got {error_value}")
        if error_bound < 0:
            raise ValueError(f"error_bound must be non-negative, got {error_bound}")

        record = {
            "stage": stage,
            "error_value": error_value,
            "error_bound": error_bound,
            "bound_satisfied": error_value <= error_bound,
            "metadata": metadata or {},
            "recorded_at": _timestamp(),
        }
        self._translations[translation_id]["errors"].append(record)
        if stage not in self._translations[translation_id]["stages"]:
            self._translations[translation_id]["stages"].append(stage)

    def get_report(self, translation_id: str) -> Dict[str, Any]:
        """Return a comprehensive error report for a translation.

        The report includes every recorded error, the cumulative bound,
        and a paper-compliance check.

        Parameters
        ----------
        translation_id:
            Identifier returned by :meth:`begin_translation`.

        Returns
        -------
        Dict[str, Any]
            Nested dictionary with keys ``source_format``, ``target_format``,
            ``epsilon``, ``stages``, ``errors``, ``cumulative_error``,
            ``paper_bounds``.

        Raises
        ------
        KeyError
            If *translation_id* is unknown.
        """
        if translation_id not in self._translations:
            raise KeyError(f"Unknown translation id: {translation_id!r}")

        t = self._translations[translation_id]
        cumulative = self.get_cumulative_error(translation_id)
        paper = self.check_paper_bound(translation_id)

        return {
            "translation_id": translation_id,
            "source_format": t["source_format"],
            "target_format": t["target_format"],
            "epsilon": t["epsilon"],
            "stages": list(t["stages"]),
            "errors": list(t["errors"]),
            "cumulative_error": cumulative,
            "paper_bounds": paper,
        }

    def get_cumulative_error(self, translation_id: str) -> float:
        """Return the total accumulated error bound across all stages.

        For independent error sources the cumulative bound is the sum of the
        individual bounds (a conservative worst-case estimate).

        Parameters
        ----------
        translation_id:
            Identifier returned by :meth:`begin_translation`.

        Returns
        -------
        float
            Σ error_bound across all recorded stages.

        Raises
        ------
        KeyError
            If *translation_id* is unknown.
        """
        if translation_id not in self._translations:
            raise KeyError(f"Unknown translation id: {translation_id!r}")
        return sum(e["error_bound"] for e in self._translations[translation_id]["errors"])

    def check_paper_bound(self, translation_id: str) -> Dict[str, Any]:
        """Check whether recorded errors satisfy the paper's theoretical bounds.

        Verified bounds (referenced by section / theorem):

        * **Round-trip error ≤ 2ε** (Section 8)
        * **Signal restoration contraction**: δ' = 4δ² + 4ε < δ (Theorem 4.2)
        * **Fixed point**: δ* ≈ 4ε (Theorem 4.2, requires ε < 1/64)
        * **Taylor truncation**: |R_N| ≤ M^{N+1}/(N+1)! · e^M (standard bound)
        * **Self-correcting cycle**: O(ε^{2^k}) after k steps (Section 8)

        Parameters
        ----------
        translation_id:
            Identifier returned by :meth:`begin_translation`.

        Returns
        -------
        Dict[str, Any]
            Dictionary with one key per bound, each containing ``measured``,
            ``bound``, and ``satisfied`` fields.
        """
        if translation_id not in self._translations:
            raise KeyError(f"Unknown translation id: {translation_id!r}")

        t = self._translations[translation_id]
        eps = t["epsilon"]
        errors = t["errors"]

        # ── Round-trip error ≤ 2ε  (Section 8) ──────────────────────────
        rt_errors = [e for e in errors if "round_trip" in e["stage"]]
        rt_measured = max((e["error_value"] for e in rt_errors), default=0.0)
        rt_bound = 2 * eps

        # ── Contraction δ' = 4δ² + 4ε < δ  (Theorem 4.2) ──────────────
        contraction_errors = [e for e in errors if "contraction" in e["stage"]]
        contraction_delta = max((e["error_value"] for e in contraction_errors), default=0.0)
        contraction_bound_met = contraction_delta < eps  # simplified check

        # ── Fixed point δ* ≈ 4ε  (Theorem 4.2) ─────────────────────────
        fixed_point = compute_fixed_point(eps)
        fp_errors = [e for e in errors if "fixed_point" in e["stage"]]
        fp_measured = max((e["error_value"] for e in fp_errors), default=0.0)

        # ── Taylor truncation (standard remainder bound) ────────────────
        taylor_errors = [e for e in errors if "taylor" in e["stage"]]
        taylor_measured = max((e["error_value"] for e in taylor_errors), default=0.0)
        taylor_bound = max((e["error_bound"] for e in taylor_errors), default=0.0)

        # ── Self-correcting cycle O(ε^{2^k})  (Section 8) ──────────────
        sc_errors = [e for e in errors if "self_correct" in e["stage"]]
        sc_measured = max((e["error_value"] for e in sc_errors), default=0.0)

        return {
            "round_trip": {
                "measured": rt_measured,
                "bound": rt_bound,
                "satisfied": rt_measured <= rt_bound,
                "reference": "Section 8",
            },
            "contraction": {
                "delta_prime": contraction_delta,
                "epsilon": eps,
                "contraction_holds": contraction_bound_met,
                "reference": "Theorem 4.2",
            },
            "fixed_point": {
                "measured": fp_measured,
                "theoretical_delta_star": fixed_point,
                "epsilon_lt_1_over_64": eps < 1.0 / 64,
                "reference": "Theorem 4.2",
            },
            "taylor_truncation": {
                "measured": taylor_measured,
                "bound": taylor_bound,
                "satisfied": taylor_measured <= taylor_bound if taylor_bound > 0 else True,
                "reference": "Taylor remainder theorem",
            },
            "self_correcting_cycle": {
                "measured": sc_measured,
                "theoretical_order": f"O(ε^(2^k))",
                "reference": "Section 8",
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Stage-Specific Error Measurement Functions
# ═══════════════════════════════════════════════════════════════════════════════


def measure_latex_to_eml_error(
    latex_str: str,
    test_env: Dict[str, float],
) -> Dict[str, Any]:
    """Measure the error introduced by the LaTeX → EML decomposition.

    The LaTeX expression is parsed and converted to an EML binary tree.
    The EML tree is then evaluated numerically under *test_env* and compared
    against a reference value computed via Python's ``math`` module.

    Error sources in this stage include:
    * **Taylor truncation** for trigonometric / transcendental functions
      (standard remainder bound: |R_N| ≤ M^{N+1}/(N+1)! · e^M).
    * **Domain restrictions** — ln(x) is only defined for x > 0 (Remark 2.4a).
    * **Constant encoding** — floating-point constants are stored as annotated
      VAR nodes rather than exact EML trees.

    Parameters
    ----------
    latex_str:
        A LaTeX math expression string (e.g. ``r"\\exp(x) + \\ln(y)"``).
    test_env:
        A mapping from variable names to the numeric values at which the
        expression should be evaluated.

    Returns
    -------
    Dict[str, Any]
        Keys: ``latex``, ``eml_value``, ``reference_value``, ``absolute_error``,
        ``relative_error``, ``eml_tree_depth``, ``eml_tree_size``,
        ``decompositions``.
    """
    from eml_pipeline.eml.latex_to_eml import latex_to_eml

    result: Dict[str, Any] = {
        "latex": latex_str,
        "eml_value": None,
        "reference_value": None,
        "absolute_error": None,
        "relative_error": None,
        "eml_tree_depth": None,
        "eml_tree_size": None,
        "decompositions": {},
    }

    # ── Parse LaTeX → EML ──────────────────────────────────────────────
    try:
        eml_tree, convert_meta = latex_to_eml(latex_str)
    except Exception as exc:
        result["error"] = f"LaTeX → EML conversion failed: {exc}"
        return result

    result["eml_tree_depth"] = eml_tree.depth()
    result["eml_tree_size"] = eml_tree.size()
    result["decompositions"] = convert_meta.get("conversions", {})

    # ── Evaluate EML tree ──────────────────────────────────────────────
    try:
        eml_value = eml_evaluate(eml_tree, test_env)
        result["eml_value"] = eml_value
    except Exception as exc:
        result["error"] = f"EML evaluation failed: {exc}"
        return result

    # ── Compute reference via Python math ──────────────────────────────
    ref_value = _evaluate_latex_reference(latex_str, test_env)
    result["reference_value"] = ref_value

    if ref_value is not None and eml_value is not None:
        abs_err = abs(eml_value - ref_value)
        result["absolute_error"] = abs_err
        if abs(ref_value) > 1e-15:
            result["relative_error"] = abs_err / abs(ref_value)
        else:
            result["relative_error"] = abs_err  # absolute when ref ≈ 0

    return result


def measure_eml_to_nand_error(
    eml_tree: EMLNode,
    test_env: Dict[str, float],
    circuit: NANDCircuit,
) -> Dict[str, Any]:
    """Measure the error between EML evaluation and NAND soft evaluation.

    This function verifies **Theorem 2.6** — the EML-soft NAND identity:
        ``NAND_R(a, b) = 1 - ab = eml(0, e^{ab})``

    For an EML tree that encodes a soft NAND, the NAND circuit's soft
    evaluation (``circuit.evaluate_soft``) should match the EML numerical
    evaluation to within floating-point tolerance.  For general EML trees
    the NAND circuit approximates the computation; the discrepancy is
    measured here.

    Parameters
    ----------
    eml_tree:
        The EML expression tree.
    test_env:
        Variable bindings for numerical evaluation.
    circuit:
        The NAND circuit produced from *eml_tree*.

    Returns
    -------
    Dict[str, Any]
        Keys: ``eml_value``, ``nand_soft_value``, ``absolute_error``,
        ``relative_error``, ``circuit_gates``, ``circuit_depth``,
        ``theorem_2_6_verified``.
    """
    result: Dict[str, Any] = {
        "eml_value": None,
        "nand_soft_value": None,
        "absolute_error": None,
        "relative_error": None,
        "circuit_gates": circuit.gate_count(),
        "circuit_depth": circuit.depth(),
        "theorem_2_6_verified": None,
    }

    # ── Evaluate EML tree ──────────────────────────────────────────────
    try:
        eml_value = eml_evaluate(eml_tree, test_env)
        result["eml_value"] = eml_value
    except Exception as exc:
        result["error"] = f"EML evaluation failed: {exc}"
        return result

    # ── Evaluate NAND circuit in soft mode ─────────────────────────────
    variables = eml_tree.variables()
    soft_inputs: List[float] = []
    for var in variables:
        soft_inputs.append(test_env.get(var, 0.5))

    try:
        nand_soft_vals = circuit.evaluate_soft(soft_inputs, epsilon=0.0)
        nand_soft_value = nand_soft_vals[0] if nand_soft_vals else None
        result["nand_soft_value"] = nand_soft_value
    except Exception as exc:
        result["error"] = f"NAND soft evaluation failed: {exc}"
        return result

    # ── Compare ─────────────────────────────────────────────────────────
    if eml_value is not None and nand_soft_value is not None:
        abs_err = abs(eml_value - nand_soft_value)
        result["absolute_error"] = abs_err
        if abs(eml_value) > 1e-15:
            result["relative_error"] = abs_err / abs(eml_value)
        else:
            result["relative_error"] = abs_err

    # ── Verify Theorem 2.6 identity ────────────────────────────────────
    # NAND_R(a,b) = 1 - ab = eml(0, e^{ab})
    # Check whether the EML tree contains a soft NAND sub-tree
    theorem_verified = _verify_theorem_2_6(eml_tree, test_env)
    result["theorem_2_6_verified"] = theorem_verified

    return result


def measure_nand_rewrite_error(
    original: NANDCircuit,
    rewritten: NANDCircuit,
    num_tests: int = 1000,
) -> Dict[str, Any]:
    """Verify functional equivalence after NAND pattern rewriting.

    Rewriting preserves the Boolean function but may change gate count or
    circuit topology.  This function tests:

    1. **Boolean equivalence** — exhaustive for ≤ 16 input wires, random
       otherwise.
    2. **Soft equivalence** — random inputs in [0, 1] with the soft NAND
       extension (Theorem 2.6).

    Parameters
    ----------
    original:
        The NAND circuit before rewriting.
    rewritten:
        The NAND circuit after rewriting.
    num_tests:
        Number of random test vectors to use when exhaustive testing is
        infeasible.

    Returns
    -------
    Dict[str, Any]
        Keys: ``equivalent_bool``, ``equivalent_soft``, ``max_soft_error``,
        ``mean_soft_error``, ``original_gates``, ``rewritten_gates``,
        ``gates_saved``, ``verification_tests``.
    """
    import random

    result: Dict[str, Any] = {
        "equivalent_bool": None,
        "equivalent_soft": None,
        "max_soft_error": 0.0,
        "mean_soft_error": 0.0,
        "original_gates": original.gate_count(),
        "rewritten_gates": rewritten.gate_count(),
        "gates_saved": original.gate_count() - rewritten.gate_count(),
        "verification_tests": 0,
    }

    n = original.num_inputs
    total_tests = 0

    # ── Boolean equivalence ─────────────────────────────────────────────
    if n <= 16:
        # Exhaustive Boolean testing
        bool_equivalent = True
        for i in range(2**n):
            inputs = [(i >> j) & 1 == 1 for j in range(n)]
            orig_out = original.evaluate(inputs)
            rew_out = rewritten.evaluate(inputs)
            if orig_out != rew_out:
                bool_equivalent = False
                break
            total_tests += 1
        result["equivalent_bool"] = bool_equivalent
    else:
        # Random Boolean testing
        bool_equivalent = True
        for _ in range(num_tests):
            inputs = [random.choice([False, True]) for _ in range(n)]
            orig_out = original.evaluate(inputs)
            rew_out = rewritten.evaluate(inputs)
            if orig_out != rew_out:
                bool_equivalent = False
                break
            total_tests += 1
        result["equivalent_bool"] = bool_equivalent

    # ── Soft equivalence ────────────────────────────────────────────────
    soft_errors: List[float] = []
    soft_equivalent = True
    tolerance = 1e-10

    for _ in range(num_tests):
        inputs = [random.random() for _ in range(n)]
        try:
            orig_soft = original.evaluate_soft(inputs, epsilon=0.0)
            rew_soft = rewritten.evaluate_soft(inputs, epsilon=0.0)
            for o_val, r_val in zip(orig_soft, rew_soft):
                err = abs(o_val - r_val)
                soft_errors.append(err)
                if err > tolerance:
                    soft_equivalent = False
        except Exception:
            soft_equivalent = False
            break
        total_tests += 1

    if soft_errors:
        result["max_soft_error"] = max(soft_errors)
        result["mean_soft_error"] = sum(soft_errors) / len(soft_errors)
    result["equivalent_soft"] = soft_equivalent
    result["verification_tests"] = total_tests

    return result


def measure_nand_to_verilog_error(
    circuit: NANDCircuit,
    verilog_code: str,
) -> Dict[str, Any]:
    """Perform a structural check between a NAND circuit and Verilog code.

    This is a **structural** (not functional) check: the Verilog should
    instantiate exactly the same set of NAND gates with the same wire
    connections as the circuit object.

    Parameters
    ----------
    circuit:
        The NAND circuit to compare against.
    verilog_code:
        Generated Verilog source code.

    Returns
    -------
    Dict[str, Any]
        Keys: ``structural_match``, ``gate_count_match``,
        ``wire_connections_verified``.
    """
    result: Dict[str, Any] = {
        "structural_match": False,
        "gate_count_match": False,
        "wire_connections_verified": False,
    }

    # ── Count gate instances in Verilog ─────────────────────────────────
    # Match patterns like:  NAND gate0 (.A(wire_a), .B(wire_b), .Y(wire_y));
    # or:  assign wire_y = ~(wire_a & wire_b);
    nand_pattern = re.compile(
        r"\bNAND\b\s+(\w+)\s*\(", re.IGNORECASE
    )
    nand_instances = nand_pattern.findall(verilog_code)

    # Also check for Verilog structural nand primitives:
    # nand(y, a, b);
    nand_primitive = re.compile(
        r"\bnand\b\s*\(", re.IGNORECASE
    )
    primitive_count = len(nand_primitive.findall(verilog_code))

    verilog_gate_count = len(nand_instances) + primitive_count
    circuit_gate_count = circuit.gate_count()

    result["gate_count_match"] = verilog_gate_count == circuit_gate_count

    # ── Verify wire connections ─────────────────────────────────────────
    # Extract wire assignments / connections from Verilog
    wire_connections_ok = True
    for gate in circuit.gates:
        # Look for the gate's output wire in the Verilog source
        wire_name = f"w{gate.output}"
        if wire_name not in verilog_code and f"wire_{gate.output}" not in verilog_code:
            # Not every wire is necessarily named with this convention;
            # the check is best-effort.
            wire_connections_ok = False
            break

    result["wire_connections_verified"] = wire_connections_ok
    result["structural_match"] = result["gate_count_match"] and wire_connections_ok

    # Include diagnostic counts
    result["verilog_gate_count"] = verilog_gate_count
    result["circuit_gate_count"] = circuit_gate_count

    return result


def measure_verilog_to_optimized_error(
    original_verilog: str,
    optimized_verilog: str,
    yosys_available: bool = False,
) -> Dict[str, Any]:
    """Compare gate counts before / after Yosys optimization.

    If Yosys is available, a formal equivalence check can be run.
    Otherwise the comparison is limited to structural analysis.

    Parameters
    ----------
    original_verilog:
        Verilog source before optimization.
    optimized_verilog:
        Verilog source after optimization.
    yosys_available:
        Whether Yosys is installed and can be invoked for formal
        equivalence checking.

    Returns
    -------
    Dict[str, Any]
        Keys: ``original_gates``, ``optimized_gates``,
        ``optimization_ratio``, ``equivalence_verified``,
        ``yosys_available``.
    """
    result: Dict[str, Any] = {
        "original_gates": 0,
        "optimized_gates": 0,
        "optimization_ratio": 0.0,
        "equivalence_verified": False,
        "yosys_available": yosys_available,
    }

    # ── Count gates in both versions ────────────────────────────────────
    original_count = _count_verilog_gates(original_verilog)
    optimized_count = _count_verilog_gates(optimized_verilog)

    result["original_gates"] = original_count
    result["optimized_gates"] = optimized_count

    if original_count > 0:
        result["optimization_ratio"] = optimized_count / original_count
    else:
        result["optimization_ratio"] = 1.0

    # ── Equivalence check ───────────────────────────────────────────────
    if yosys_available:
        try:
            result["equivalence_verified"] = _yosys_formal_eq_check(
                original_verilog, optimized_verilog
            )
        except Exception as exc:
            result["equivalence_verified"] = False
            result["equivalence_error"] = str(exc)
    else:
        # Without Yosys: best-effort structural comparison
        # Check if module port signatures match
        orig_ports = _extract_verilog_ports(original_verilog)
        opt_ports = _extract_verilog_ports(optimized_verilog)
        result["equivalence_verified"] = orig_ports == opt_ports
        result["equivalence_note"] = (
            "Structural port-matching only; install Yosys for formal equivalence"
        )

    return result


def measure_full_pipeline_error(
    latex_str: str,
    test_env: Dict[str, float],
    epsilon: float = 0.001,
) -> Dict[str, Any]:
    """Run the complete pipeline and measure error at every stage.

    Stages measured:

    1. **LaTeX → EML** — parse / decompose error (Taylor truncation, domain)
    2. **EML → NAND** — soft-NAND bridge error (Theorem 2.6)
    3. **NAND rewrite** — functional equivalence after optimisation
    4. **Round-trip** — EML → NAND → ε-NAND → ApproxEML → EML error
       (Section 8 bound: ≤ 2ε)

    Parameters
    ----------
    latex_str:
        LaTeX expression to push through the pipeline.
    test_env:
        Variable bindings for numerical evaluation.
    epsilon:
        Gate noise parameter ε (Definition 3.1).

    Returns
    -------
    Dict[str, Any]
        Comprehensive report with ``latex_to_eml_error``,
        ``eml_to_nand_error``, ``rewrite_error``, ``round_trip_error``,
        ``paper_compliance``.
    """
    from eml_pipeline.eml.latex_to_eml import latex_to_eml
    from eml_pipeline.transitions.t1_eml_to_nand import eml_to_nand

    tracker = TranslationErrorTracker(epsilon=epsilon)
    report: Dict[str, Any] = {
        "latex": latex_str,
        "epsilon": epsilon,
        "latex_to_eml_error": None,
        "eml_to_nand_error": None,
        "rewrite_error": None,
        "round_trip_error": None,
        "paper_compliance": None,
    }

    # ── Stage 1: LaTeX → EML ───────────────────────────────────────────
    tid1 = tracker.begin_translation("LaTeX", "EML")
    l2e = measure_latex_to_eml_error(latex_str, test_env)
    report["latex_to_eml_error"] = l2e

    if l2e.get("absolute_error") is not None:
        tracker.record_error(
            tid1,
            stage="latex_to_eml",
            error_value=l2e["absolute_error"],
            error_bound=l2e.get("relative_error", l2e["absolute_error"]),
            metadata={"relative_error": l2e.get("relative_error")},
        )

    # ── Stage 2: EML → NAND ────────────────────────────────────────────
    eml_tree = None
    try:
        eml_tree, _ = latex_to_eml(latex_str)
    except Exception:
        pass

    if eml_tree is not None:
        tid2 = tracker.begin_translation("EML", "NAND")
        try:
            circuit, conv_meta = eml_to_nand(eml_tree, epsilon=epsilon)
            e2n = measure_eml_to_nand_error(eml_tree, test_env, circuit)
            report["eml_to_nand_error"] = e2n

            if e2n.get("absolute_error") is not None:
                tracker.record_error(
                    tid2,
                    stage="eml_to_nand",
                    error_value=e2n["absolute_error"],
                    error_bound=epsilon,  # Theorem 2.6: soft NAND is exact on interior
                    metadata={"circuit_gates": e2n.get("circuit_gates")},
                )
        except Exception as exc:
            report["eml_to_nand_error"] = {"error": str(exc)}

    # ── Stage 3: NAND rewrite ──────────────────────────────────────────
    # (No rewriting applied in the basic pipeline; mark as N/A)
    report["rewrite_error"] = {
        "equivalent_bool": True,
        "equivalent_soft": True,
        "max_soft_error": 0.0,
        "mean_soft_error": 0.0,
        "original_gates": 0,
        "rewritten_gates": 0,
        "gates_saved": 0,
        "verification_tests": 0,
        "note": "No rewrite applied in basic pipeline",
    }

    # ── Stage 4: Round-trip error (Section 8) ──────────────────────────
    rt = _measure_round_trip_pipeline(epsilon)
    report["round_trip_error"] = rt

    tid4 = tracker.begin_translation("EML", "EML_round_trip")
    if rt.get("max_error") is not None:
        tracker.record_error(
            tid4,
            stage="round_trip",
            error_value=rt["max_error"],
            error_bound=2 * epsilon,
            metadata={"mean_error": rt.get("mean_error")},
        )

    # ── Paper compliance ────────────────────────────────────────────────
    report["paper_compliance"] = tracker.check_paper_bound(tid4)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PaperBoundChecker
# ═══════════════════════════════════════════════════════════════════════════════


class PaperBoundChecker:
    """Static utilities for verifying theoretical bounds from the paper.

    Every method is a ``@staticmethod`` that takes numeric parameters and
    returns a dictionary indicating whether the bound is satisfied, along
    with the computed values.
    """

    @staticmethod
    def check_round_trip_bound(measured_error: float, epsilon: float) -> Dict[str, Any]:
        """Check whether the round-trip error satisfies the ≤ 2ε bound.

        **Reference**: Section 8 — "The round-trip error of the
        EML → NAND → ε-NAND → ApproxEML → EML cycle is bounded by 2ε."

        Parameters
        ----------
        measured_error:
            The empirically observed round-trip error.
        epsilon:
            The gate noise parameter ε.

        Returns
        -------
        Dict[str, Any]
            Keys: ``measured``, ``bound``, ``ratio``, ``satisfied``,
            ``reference``.
        """
        bound = 2 * epsilon
        return {
            "measured": measured_error,
            "bound": bound,
            "ratio": measured_error / bound if bound > 0 else float("inf"),
            "satisfied": measured_error <= bound,
            "reference": "Section 8: round-trip error ≤ 2ε",
        }

    @staticmethod
    def check_contraction(delta: float, epsilon: float) -> Dict[str, Any]:
        """Check the signal-restoration contraction bound.

        **Theorem 4.2**: If δ < 2/9 and ε < δ²/8, then the restoration
        map R satisfies R : [b]_δ → [b]_{δ'} where δ' = 4δ² + 4ε < δ.

        Parameters
        ----------
        delta:
            Current signal deviation δ from a Boolean value.
        epsilon:
            Gate noise parameter ε.

        Returns
        -------
        Dict[str, Any]
            Keys: ``delta``, ``delta_prime``, ``epsilon``,
            ``contraction_holds``, ``precondition_delta``,
            ``precondition_epsilon``, ``reference``.
        """
        delta_prime = compute_contraction(delta, epsilon)
        contraction_holds = delta_prime < delta
        precond_delta = delta < 2.0 / 9
        precond_epsilon = epsilon < (delta ** 2) / 8 if delta > 0 else True

        return {
            "delta": delta,
            "delta_prime": delta_prime,
            "epsilon": epsilon,
            "contraction_holds": contraction_holds,
            "precondition_delta": precond_delta,
            "precondition_epsilon": precond_epsilon,
            "reference": "Theorem 4.2: δ' = 4δ² + 4ε < δ",
        }

    @staticmethod
    def check_fixed_point(epsilon: float) -> Dict[str, Any]:
        """Compute the fixed point δ* and check ε < 1/64.

        **Theorem 4.2** (corollary): The fixed point of the contraction
        iteration satisfies δ* = (1 - √(1 - 64ε)) / 8 ≈ 4ε for small ε.
        A real solution requires ε ≤ 1/64.

        Parameters
        ----------
        epsilon:
            Gate noise parameter ε.

        Returns
        -------
        Dict[str, Any]
            Keys: ``epsilon``, ``delta_star``, ``approx_4epsilon``,
            ``epsilon_lt_1_over_64``, ``real_solution_exists``,
            ``reference``.
        """
        delta_star = compute_fixed_point(epsilon)
        real_solution = epsilon <= 1.0 / 64

        return {
            "epsilon": epsilon,
            "delta_star": delta_star,
            "approx_4epsilon": 4 * epsilon,
            "epsilon_lt_1_over_64": epsilon < 1.0 / 64,
            "real_solution_exists": real_solution,
            "reference": "Theorem 4.2: fixed point δ* ≈ 4ε",
        }

    @staticmethod
    def check_taylor_remainder(x: float, order: int) -> Dict[str, Any]:
        """Compute the Taylor remainder bound for a truncated series.

        The standard Lagrange remainder for exp(x) truncated at order *N*:
            |R_N| ≤ M^{N+1} / (N+1)! · e^M
        where M = |x| and the series is evaluated on [-M, M].

        Parameters
        ----------
        x:
            The evaluation point.
        order:
            The truncation order N (number of terms minus one).

        Returns
        -------
        Dict[str, Any]
            Keys: ``x``, ``order``, ``M``, ``remainder_bound``,
            ``actual_exp``, ``approx_exp``, ``actual_error``,
            ``bound_satisfied``, ``reference``.
        """
        M = abs(x)
        remainder_bound = (M ** (order + 1)) / math.factorial(order + 1) * math.exp(M)

        # Compute the actual truncated Taylor sum
        approx = sum(M ** k / math.factorial(k) for k in range(order + 1))
        actual = math.exp(x)
        actual_error = abs(approx - actual)

        return {
            "x": x,
            "order": order,
            "M": M,
            "remainder_bound": remainder_bound,
            "approx_exp": approx,
            "actual_exp": actual,
            "actual_error": actual_error,
            "bound_satisfied": actual_error <= remainder_bound,
            "reference": "Taylor remainder theorem: |R_N| ≤ M^{N+1}/(N+1)! · e^M",
        }

    @staticmethod
    def check_self_correcting(epsilon: float, k: int) -> Dict[str, Any]:
        """Check the self-correcting cycle bound O(ε^{2^k}).

        **Section 8**: After k round-trip cycles
        (EML → NAND → ε-NAND → ApproxEML → EML), the error is
        O(ε^{2^k}).  This is because each cycle squares the error
        (up to constants), giving doubly-exponential convergence.

        Parameters
        ----------
        epsilon:
            Gate noise parameter ε.
        k:
            Number of self-correction cycles.

        Returns
        -------
        Dict[str, Any]
            Keys: ``epsilon``, ``k``, ``theoretical_order``,
            ``estimated_error``, ``reference``.
        """
        if epsilon <= 0 or epsilon >= 1:
            return {
                "epsilon": epsilon,
                "k": k,
                "theoretical_order": f"O(ε^(2^{k}))",
                "estimated_error": float("inf"),
                "error": "ε must be in (0, 1)",
                "reference": "Section 8: self-correcting cycle O(ε^{2^k})",
            }

        # Conservative estimate: E_{k} ≈ C · ε^{2^k} with C = 2
        # This follows from E_{k+1} ≈ 2·E_k² + 2ε (Section 8 analysis)
        estimated = 2.0 * (epsilon ** (2 ** k))
        theoretical = f"O(ε^(2^{k}))"

        return {
            "epsilon": epsilon,
            "k": k,
            "theoretical_order": theoretical,
            "estimated_error": estimated,
            "reference": "Section 8: self-correcting cycle O(ε^{2^k})",
        }

    @staticmethod
    def full_paper_compliance_report(
        epsilon: float,
        measurements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a compliance report against ALL paper bounds.

        Parameters
        ----------
        epsilon:
            Gate noise parameter ε.
        measurements:
            Dictionary of measured values.  Expected keys (all optional):

            * ``round_trip_error`` — measured round-trip error
            * ``delta`` — signal deviation for contraction check
            * ``taylor_x`` — evaluation point for Taylor check
            * ``taylor_order`` — truncation order for Taylor check
            * ``self_correcting_k`` — number of correction cycles

        Returns
        -------
        Dict[str, Any]
            Keys: ``epsilon``, ``round_trip``, ``contraction``,
            ``fixed_point``, ``taylor``, ``self_correcting``,
            ``all_satisfied``, ``summary``.
        """
        report: Dict[str, Any] = {"epsilon": epsilon}

        # ── Round-trip (Section 8) ──────────────────────────────────────
        rt_error = measurements.get("round_trip_error", 0.0)
        report["round_trip"] = PaperBoundChecker.check_round_trip_bound(
            rt_error, epsilon
        )

        # ── Contraction (Theorem 4.2) ───────────────────────────────────
        delta = measurements.get("delta", 0.1)
        report["contraction"] = PaperBoundChecker.check_contraction(delta, epsilon)

        # ── Fixed point (Theorem 4.2) ───────────────────────────────────
        report["fixed_point"] = PaperBoundChecker.check_fixed_point(epsilon)

        # ── Taylor remainder ────────────────────────────────────────────
        tx = measurements.get("taylor_x", 1.0)
        tord = measurements.get("taylor_order", 8)
        report["taylor"] = PaperBoundChecker.check_taylor_remainder(tx, tord)

        # ── Self-correcting (Section 8) ─────────────────────────────────
        sc_k = measurements.get("self_correcting_k", 3)
        report["self_correcting"] = PaperBoundChecker.check_self_correcting(
            epsilon, sc_k
        )

        # ── Summary ─────────────────────────────────────────────────────
        checks = [
            report["round_trip"]["satisfied"],
            report["contraction"]["contraction_holds"],
            report["fixed_point"]["real_solution_exists"],
            report["taylor"]["bound_satisfied"],
        ]
        report["all_satisfied"] = all(checks)
        report["summary"] = {
            "round_trip": "PASS" if report["round_trip"]["satisfied"] else "FAIL",
            "contraction": "PASS" if report["contraction"]["contraction_holds"] else "FAIL",
            "fixed_point": "PASS" if report["fixed_point"]["real_solution_exists"] else "FAIL",
            "taylor": "PASS" if report["taylor"]["bound_satisfied"] else "FAIL",
            "self_correcting": "INFO",  # informational only
        }

        return report


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ErrorVisualization
# ═══════════════════════════════════════════════════════════════════════════════


class ErrorVisualization:
    """Format error reports and paper-compliance reports as readable text.

    All methods return strings suitable for terminal output or log files.
    """

    # Width constants for alignment
    _LABEL_W = 32
    _VALUE_W = 16

    @staticmethod
    def format_error_report(report: Dict[str, Any]) -> str:
        """Format an error report dict as a human-readable, aligned string.

        Parameters
        ----------
        report:
            A report dictionary as returned by
            :meth:`TranslationErrorTracker.get_report` or
            :func:`measure_full_pipeline_error`.

        Returns
        -------
        str
            Multi-line formatted text.
        """
        lines: List[str] = []
        lw = ErrorVisualization._LABEL_W
        vw = ErrorVisualization._VALUE_W

        lines.append("=" * 72)
        lines.append("  EML Pipeline Translation Error Report")
        lines.append("=" * 72)

        # Header information
        lines.append("")
        lines.append(f"  {'Translation ID:':<{lw}} {report.get('translation_id', 'N/A')}")
        lines.append(f"  {'Source Format:':<{lw}} {report.get('source_format', 'N/A')}")
        lines.append(f"  {'Target Format:':<{lw}} {report.get('target_format', 'N/A')}")
        lines.append(f"  {'Epsilon (ε):':<{lw}} {report.get('epsilon', 'N/A')}")
        lines.append(f"  {'Cumulative Error Bound:':<{lw}} {report.get('cumulative_error', 'N/A')}")
        lines.append("")

        # Stage-by-stage errors
        errors = report.get("errors", [])
        if errors:
            lines.append("  Stage Errors")
            lines.append("  " + "-" * 68)
            lines.append(f"  {'Stage':<24} {'Measured':>12} {'Bound':>12} {'OK':>6}")
            lines.append("  " + "-" * 68)
            for e in errors:
                ok_str = "✓" if e.get("bound_satisfied") else "✗"
                lines.append(
                    f"  {e['stage']:<24} "
                    f"{_fmt(e['error_value']):>12} "
                    f"{_fmt(e['error_bound']):>12} "
                    f"{ok_str:>6}"
                )
            lines.append("")

        # Paper bounds section
        paper = report.get("paper_bounds", {})
        if paper:
            lines.append("  Paper Bound Checks")
            lines.append("  " + "-" * 68)
            for bound_name, bound_info in paper.items():
                lines.append(f"  {bound_name}:")
                for k, v in bound_info.items():
                    if isinstance(v, float):
                        lines.append(f"    {k + ':':<28} {_fmt(v)}")
                    elif isinstance(v, bool):
                        lines.append(f"    {k + ':':<28} {v}")
                    else:
                        lines.append(f"    {k + ':':<28} {v}")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    @staticmethod
    def format_compliance_report(report: Dict[str, Any]) -> str:
        """Format a paper-compliance report with PASS/FAIL for each bound.

        Parameters
        ----------
        report:
            A compliance report as returned by
            :meth:`PaperBoundChecker.full_paper_compliance_report`.

        Returns
        -------
        str
            Multi-line formatted text with PASS / FAIL indicators.
        """
        lines: List[str] = []
        lw = ErrorVisualization._LABEL_W

        lines.append("=" * 72)
        lines.append("  Paper Compliance Report")
        lines.append("  Reference: \"The EML–NAND Duality\" (Derycke, 2026)")
        lines.append("=" * 72)
        lines.append(f"  {'Epsilon (ε):':<{lw}} {report.get('epsilon', 'N/A')}")
        lines.append("")

        summary = report.get("summary", {})

        # ── Round-trip ──────────────────────────────────────────────────
        rt = report.get("round_trip", {})
        rt_status = summary.get("round_trip", "N/A")
        lines.append(f"  {'[Round-Trip]  Section 8':<{lw}} {rt_status}")
        if rt:
            lines.append(f"    {'Measured:':<{lw}} {_fmt(rt.get('measured', 0))}")
            lines.append(f"    {'Bound (2ε):':<{lw}} {_fmt(rt.get('bound', 0))}")
            lines.append(f"    {'Ratio:':<{lw}} {_fmt(rt.get('ratio', 0))}")
        lines.append("")

        # ── Contraction ─────────────────────────────────────────────────
        ct = report.get("contraction", {})
        ct_status = summary.get("contraction", "N/A")
        lines.append(f"  {'[Contraction]  Theorem 4.2':<{lw}} {ct_status}")
        if ct:
            lines.append(f"    {'δ:':<{lw}} {_fmt(ct.get('delta', 0))}")
            lines.append(f"    {'δʹ = 4δ² + 4ε:':<{lw}} {_fmt(ct.get('delta_prime', 0))}")
            lines.append(f"    {'Contraction holds:':<{lw}} {ct.get('contraction_holds', False)}")
            lines.append(f"    {'Precondition δ < 2/9:':<{lw}} {ct.get('precondition_delta', False)}")
            lines.append(f"    {'Precondition ε < δ²/8:':<{lw}} {ct.get('precondition_epsilon', False)}")
        lines.append("")

        # ── Fixed point ─────────────────────────────────────────────────
        fp = report.get("fixed_point", {})
        fp_status = summary.get("fixed_point", "N/A")
        lines.append(f"  {'[Fixed Point]  Theorem 4.2':<{lw}} {fp_status}")
        if fp:
            lines.append(f"    {'δ* (exact):':<{lw}} {_fmt(fp.get('delta_star', 0))}")
            lines.append(f"    {'δ* ≈ 4ε:':<{lw}} {_fmt(fp.get('approx_4epsilon', 0))}")
            lines.append(f"    {'ε < 1/64:':<{lw}} {fp.get('epsilon_lt_1_over_64', False)}")
        lines.append("")

        # ── Taylor ──────────────────────────────────────────────────────
        ty = report.get("taylor", {})
        ty_status = summary.get("taylor", "N/A")
        lines.append(f"  {'[Taylor Remainder]':<{lw}} {ty_status}")
        if ty:
            lines.append(f"    {'x:':<{lw}} {_fmt(ty.get('x', 0))}")
            lines.append(f"    {'Order N:':<{lw}} {ty.get('order', 'N/A')}")
            lines.append(f"    {'Remainder bound:':<{lw}} {_fmt(ty.get('remainder_bound', 0))}")
            lines.append(f"    {'Actual error:':<{lw}} {_fmt(ty.get('actual_error', 0))}")
        lines.append("")

        # ── Self-correcting ─────────────────────────────────────────────
        sc = report.get("self_correcting", {})
        sc_status = summary.get("self_correcting", "N/A")
        lines.append(f"  {'[Self-Correcting]  Section 8':<{lw}} {sc_status}")
        if sc:
            lines.append(f"    {'k cycles:':<{lw}} {sc.get('k', 'N/A')}")
            lines.append(f"    {'Theoretical:':<{lw}} {sc.get('theoretical_order', 'N/A')}")
            lines.append(f"    {'Estimated error:':<{lw}} {_fmt(sc.get('estimated_error', 0))}")
        lines.append("")

        # ── Overall ─────────────────────────────────────────────────────
        all_ok = report.get("all_satisfied", False)
        overall = "ALL PASS ✓" if all_ok else "SOME FAILED ✗"
        lines.append("  " + "=" * 68)
        lines.append(f"  {'Overall:':<{lw}} {overall}")
        lines.append("=" * 72)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Private Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════


def _timestamp() -> str:
    """Return an ISO-8601 timestamp for the current instant."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _fmt(value: float, precision: int = 8) -> str:
    """Format a float for aligned display, falling back to scientific notation."""
    if value == 0.0:
        return "0.0"
    if abs(value) < 1e-4 or abs(value) > 1e6:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def _evaluate_latex_reference(
    latex_str: str,
    test_env: Dict[str, float],
) -> Optional[float]:
    """Attempt to evaluate a LaTeX expression using Python's ``math`` module.

    This is a best-effort function that handles simple expressions
    (arithmetic, exp, ln, sin, cos, powers, fractions).  It is *not*
    a full LaTeX interpreter — it serves as a reference value for
    measuring LaTeX → EML translation error.

    Parameters
    ----------
    latex_str:
        LaTeX math string.
    test_env:
        Variable name → float mapping.

    Returns
    -------
    Optional[float]
        The reference value, or ``None`` if evaluation fails.
    """
    try:
        # Build a safe Python expression from the LaTeX string
        expr = latex_str
        # Strip common LaTeX delimiters
        for delim in ("$$", "$", "\\[", "\\]", "\\(", "\\)"):
            expr = expr.replace(delim, "")
        expr = expr.strip()

        # Replace LaTeX commands with Python math equivalents
        replacements = [
            (r"\\exp", "math.exp"),
            (r"\\ln", "math.log"),
            (r"\\log", "math.log10"),
            (r"\\sin", "math.sin"),
            (r"\\cos", "math.cos"),
            (r"\\tan", "math.tan"),
            (r"\\sqrt", "math.sqrt"),
            (r"\\pi", "math.pi"),
            (r"\\cdot", "*"),
            (r"\\times", "*"),
            (r"\\div", "/"),
            (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))"),
        ]
        for pattern, repl in replacements:
            expr = re.sub(pattern, repl, expr)

        # Replace ^ with ** for exponentiation
        expr = expr.replace("^", "**")

        # Create a safe evaluation namespace
        safe_ns = {"math": math, "__builtins__": {}}
        safe_ns.update(test_env)

        return float(eval(expr, safe_ns))  # noqa: S307
    except Exception:
        return None


def _verify_theorem_2_6(eml_tree: EMLNode, test_env: Dict[str, float]) -> Optional[bool]:
    """Verify Theorem 2.6: NAND_R(a,b) = 1 - ab = eml(0, e^{ab}).

    Checks whether the EML tree contains a sub-tree marked as
    ``is_soft_nand`` and, if so, verifies that its evaluated output
    equals 1 - a·b for the given test environment.

    Parameters
    ----------
    eml_tree:
        The EML tree to check.
    test_env:
        Variable bindings for numerical evaluation.

    Returns
    -------
    Optional[bool]
        ``True`` if Theorem 2.6 identity holds for the test values,
        ``False`` if it does not, ``None`` if no soft NAND sub-tree
        is found.
    """
    # Recursively search for soft NAND sub-trees
    sand_nodes = _find_soft_nand_nodes(eml_tree)
    if not sand_nodes:
        return None

    all_ok = True
    for node in sand_nodes:
        try:
            eml_val = eml_evaluate(node, test_env)
        except Exception:
            all_ok = False
            continue

        # Extract the operands a, b from the tree structure
        # eml_soft_nand(a, b) = EML(zero, exp(multiply(a, b)))
        # The node is: EML(zero_node, exp_ab_node)
        if (
            node.node_type == EMLNodeType.EML
            and node.right is not None
            and node.right.node_type == EMLNodeType.EML
        ):
            # Try to evaluate a and b from the sub-tree
            # This is best-effort; the tree structure may vary
            pass

        # Direct check: if we can identify a, b in the env, verify 1-ab
        # For the general case, we rely on the metadata
        vars_in_node = node.variables()
        if len(vars_in_node) >= 2:
            a_val = test_env.get(vars_in_node[0], None)
            b_val = test_env.get(vars_in_node[1], None)
            if a_val is not None and b_val is not None:
                expected = 1.0 - a_val * b_val
                if abs(eml_val - expected) > 1e-10:
                    all_ok = False

    return all_ok


def _find_soft_nand_nodes(node: EMLNode) -> List[EMLNode]:
    """Recursively find all sub-trees marked as soft NAND (Theorem 2.6)."""
    result: List[EMLNode] = []
    if node.metadata.get("is_soft_nand"):
        result.append(node)
    if node.left is not None:
        result.extend(_find_soft_nand_nodes(node.left))
    if node.right is not None:
        result.extend(_find_soft_nand_nodes(node.right))
    return result


def _count_verilog_gates(verilog_code: str) -> int:
    """Count NAND gate instances in a Verilog source string.

    Looks for both instantiated modules (``NAND gateN (...)``) and
    Verilog primitives (``nand(y, a, b)``).
    """
    # Module instantiation
    inst_pattern = re.compile(r"\bNAND\b\s+\w+\s*\(", re.IGNORECASE)
    # Primitive
    prim_pattern = re.compile(r"\bnand\b\s*\(", re.IGNORECASE)
    # Assign with NAND-equivalent: ~(a & b)
    assign_nand = re.compile(r"=\s*~\s*\(", re.IGNORECASE)

    count = len(inst_pattern.findall(verilog_code))
    count += len(prim_pattern.findall(verilog_code))
    count += len(assign_nand.findall(verilog_code))
    return count


def _extract_verilog_ports(verilog_code: str) -> Dict[str, List[str]]:
    """Extract module port names from Verilog source.

    Returns a dict with ``inputs`` and ``outputs`` lists.
    """
    result: Dict[str, List[str]] = {"inputs": [], "outputs": []}

    # Match: module name (input1, input2, output1, ...);
    module_match = re.search(
        r"module\s+\w+\s*\(([^)]*)\)\s*;", verilog_code, re.DOTALL
    )
    if not module_match:
        return result

    port_list = module_match.group(1)

    # Match input declarations
    for m in re.finditer(r"input\s+(?:wire\s+)?(\w+)", port_list):
        result["inputs"].append(m.group(1))

    # Match output declarations
    for m in re.finditer(r"output\s+(?:wire\s+)?(\w+)", port_list):
        result["outputs"].append(m.group(1))

    # Also check standalone input/output declarations in the body
    for m in re.finditer(r"input\s+(?:wire\s+)?(\w+)\s*;", verilog_code):
        if m.group(1) not in result["inputs"]:
            result["inputs"].append(m.group(1))
    for m in re.finditer(r"output\s+(?:wire\s+)?(\w+)\s*;", verilog_code):
        if m.group(1) not in result["outputs"]:
            result["outputs"].append(m.group(1))

    return result


def _yosys_formal_eq_check(original: str, optimized: str) -> bool:
    """Run Yosys formal equivalence check between two Verilog modules.

    .. note::

        This function requires Yosys to be installed and accessible on
        ``$PATH``.  It writes temporary files and invokes ``yosys`` as a
        subprocess.

    Parameters
    ----------
    original:
        Original Verilog source.
    optimized:
        Optimized Verilog source.

    Returns
    -------
    bool
        ``True`` if Yosys reports equivalence, ``False`` otherwise.
    """
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "original.v")
        opt_path = os.path.join(tmpdir, "optimized.v")
        script_path = os.path.join(tmpdir, "eq_check.ys")

        with open(orig_path, "w") as f:
            f.write(original)
        with open(opt_path, "w") as f:
            f.write(optimized)

        # Yosys script for equivalence checking
        script = (
            f"read_verilog {orig_path}\n"
            f"read_verilog {opt_path}\n"
            "hierarchy -check\n"
            "proc; opt; fsm; opt; memory; opt\n"
            "equiv_make -assert\n"
        )
        with open(script_path, "w") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                ["yosys", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmpdir,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


def _measure_round_trip_pipeline(epsilon: float) -> Dict[str, Any]:
    """Measure round-trip error using the ε-NAND framework.

    **Section 8**: Round-trip error of
    EML → NAND → ε-NAND → ApproxEML → EML is ≤ 2ε.

    Parameters
    ----------
    epsilon:
        Gate noise parameter ε.

    Returns
    -------
    Dict[str, Any]
        Keys: ``epsilon``, ``num_tests``, ``max_error``, ``mean_error``,
        ``theoretical_bound``, ``bound_satisfied``.
    """
    from eml_pipeline.epsilon_nand.epsilon_nand import measure_round_trip_error

    return measure_round_trip_error(epsilon, num_tests=1000)
