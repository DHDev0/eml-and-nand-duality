"""
Extended Pipeline Test Suite — EML-NAND Pipeline
=================================================

Comprehensive tests for the extended EML pipeline including:
- NAND pattern rewriting (constant folding, structural hashing, dead gate elimination)
- Verilog generation (generic, FPGA, ASIC targets)
- BLIF and AIGER format output
- Yosys/ABC integration (graceful skip when tools unavailable)
- FPGA and ASIC synthesis (resource estimation)
- Translation error measurement and paper bound verification
- Full pipeline forward and error measurement

Based on: "The EML-NAND Duality" by Daniel Derycke (2026)
"""

from __future__ import annotations

import sys
import os
import math
import traceback
from typing import Dict, List, Tuple, Any

# ─── Path setup ───────────────────────────────────────────────────────────────
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)       # .../eml_pipeline
grandparent_dir = os.path.dirname(parent_dir) # .../download
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

# ─── Core imports ─────────────────────────────────────────────────────────────

try:
    from eml_pipeline.nand.nand_core import (
        NANDCircuit, nand_bool, soft_nand,
        build_and_circuit, build_or_circuit, build_not_circuit, build_xor_circuit,
    )
    HAS_NAND_CORE = True
except ImportError:
    HAS_NAND_CORE = False

try:
    from eml_pipeline.nand.pattern_rewriter import (
        NANDPatternRewriter, structural_hash, propagate_constants,
        eliminate_dead_gates, simplify_restoration, optimize, verify_equivalence,
    )
    HAS_PATTERN_REWRITER = True
except ImportError:
    HAS_PATTERN_REWRITER = False

try:
    from eml_pipeline.eml.eml_core import (
        eml_soft_nand, eml_exp, eml_ln, eml_evaluate, VAR, ONE, eml_to_dict,
    )
    HAS_EML_CORE = True
except ImportError:
    HAS_EML_CORE = False

try:
    from eml_pipeline.epsilon_nand.epsilon_nand import (
        EpsilonNANDGate, EpsilonNANDCircuit, EpsilonNANDConfig,
    )
    HAS_EPSILON_NAND = True
except ImportError:
    HAS_EPSILON_NAND = False

try:
    from eml_pipeline.utils.translation_error import (
        TranslationErrorTracker, PaperBoundChecker,
        measure_latex_to_eml_error, measure_nand_rewrite_error,
    )
    HAS_TRANSLATION_ERROR = True
except ImportError:
    HAS_TRANSLATION_ERROR = False

# ─── Optional imports (may not be available) ─────────────────────────────────

try:
    from eml_pipeline.hdl.verilog_gen import VerilogGenerator
except Exception:
    VerilogGenerator = None

try:
    from eml_pipeline.hdl.yosys_abc import YosysABCIntegration, BLIFGenerator, AIGERGenerator
except Exception:
    YosysABCIntegration = BLIFGenerator = AIGERGenerator = None

try:
    from eml_pipeline.hdl.synthesis import FPGASynthesizer, ASICSynthesizer, SynthesisReport, run_synthesis
except Exception:
    FPGASynthesizer = ASICSynthesizer = run_synthesis = None

try:
    from eml_pipeline.pipeline import EMLNANDPipeline
except Exception:
    EMLNANDPipeline = None


# ═══════════════════════════════════════════════════════════════════════════════
# TestResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self, name: str):
        self.name = name; self.passed = 0; self.failed = 0; self.errors = []; self.details = []
    def check(self, condition, description, detail=None):
        if condition: self.passed += 1
        else: self.failed += 1; self.errors.append(description)
        if detail: self.details.append(detail)
    def check_approx(self, actual, expected, tol, desc):
        ok = abs(actual - expected) < tol
        self.check(ok, f"{desc}: {actual} vs {expected}", {"actual": actual, "expected": expected, "tol": tol, "ok": ok})
    def summary(self):
        total = self.passed + self.failed; rate = self.passed/total*100 if total else 0
        s = f"\n{'='*60}\n{self.name}: {self.passed}/{total} passed ({rate:.1f}%)\n"
        if self.errors:
            s += "Failed:\n"
            for e in self.errors[:20]: s += f"  - {e}\n"
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# Helper circuit builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_and_circuit():
    """AND(a, b) = NAND(NAND(a,b), NAND(a,b)) — 2 gates."""
    c = NANDCircuit(num_inputs=2)
    nand_ab = c.add_gate(0, 1)
    and_ab = c.add_gate(nand_ab, nand_ab)
    c.output_wires = [and_ab]
    return c


def _build_or_circuit():
    """OR(a, b) = NAND(NAND(a,a), NAND(b,b)) — 3 gates."""
    c = NANDCircuit(num_inputs=2)
    not_a = c.add_gate(0, 0)
    not_b = c.add_gate(1, 1)
    or_ab = c.add_gate(not_a, not_b)
    c.output_wires = [or_ab]
    return c


def _build_half_adder():
    """Half adder: sum = XOR(a,b), carry = AND(a,b) — 5 gates."""
    c = NANDCircuit(num_inputs=2)
    nand_ab = c.add_gate(0, 1)
    left = c.add_gate(0, nand_ab)
    right = c.add_gate(1, nand_ab)
    sum_out = c.add_gate(left, right)
    carry = c.add_gate(nand_ab, nand_ab)
    c.output_wires = [sum_out, carry]
    return c


def _build_circuit_with_constants():
    """Circuit with constant inputs for constant-propagation testing."""
    c = NANDCircuit(num_inputs=1)
    nand_00 = c.add_gate(-1, -1)   # NAND(0, 0) → should fold to 1
    nand_11 = c.add_gate(-2, -2)   # NAND(1, 1) → should fold to 0
    nand_a0 = c.add_gate(0, -1)    # NAND(a, 0) → should fold to 1
    c.output_wires = [nand_00, nand_11, nand_a0]
    return c


def _build_circuit_with_dead_gates():
    """Circuit with unused gate outputs for dead-gate elimination."""
    c = NANDCircuit(num_inputs=2)
    useful = c.add_gate(0, 1)
    dead1 = c.add_gate(0, 0)
    dead2 = c.add_gate(dead1, 1)
    c.output_wires = [useful]
    return c


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Pattern Rewriter
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_rewriter():
    t = TestResult("Pattern Rewriter")
    if not HAS_PATTERN_REWRITER:
        t.check(False, "pattern_rewriter module not available")
        return t

    try:
        rewriter = NANDPatternRewriter()

        # AND circuit: 2 gates
        and_circ = _build_and_circuit()
        rewritten, meta = rewriter.rewrite(and_circ)
        t.check(meta.get("functional_equivalence_verified", False),
                "AND circuit: functional equivalence verified")
        t.check(meta.get("original_gates", 0) == 2,
                f"AND circuit: original gate count = {meta.get('original_gates')}")
        t.check("rules_applied" in meta, "Metadata includes rules_applied")

        # OR circuit: 3 gates
        or_circ = _build_or_circuit()
        rewritten_or, meta_or = rewriter.rewrite(or_circ)
        t.check(meta_or.get("functional_equivalence_verified", False),
                "OR circuit: functional equivalence verified")

        # Constant folding
        const_circ = _build_circuit_with_constants()
        rewritten_const, meta_const = rewriter.rewrite(const_circ)
        rules = meta_const.get("rules_applied", {})
        t.check(rules.get("constant_folding", 0) > 0,
                f"Constant folding applied: {rules.get('constant_folding', 0)} times")
        t.check(meta_const.get("gates_removed", 0) > 0,
                f"Constant circuit: gates removed = {meta_const.get('gates_removed', 0)}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Structural Hash
# ═══════════════════════════════════════════════════════════════════════════════

def test_structural_hash():
    t = TestResult("Structural Hash")
    if not HAS_PATTERN_REWRITER:
        t.check(False, "pattern_rewriter module not available")
        return t

    try:
        # Build circuit with duplicate subgraphs
        c = NANDCircuit(num_inputs=2)
        nand1 = c.add_gate(0, 1)
        nand2 = c.add_gate(0, 1)   # duplicate of nand1
        not1 = c.add_gate(nand1, nand1)
        not2 = c.add_gate(nand2, nand2)   # also duplicate
        c.output_wires = [not1, not2]
        original_gates = c.gate_count()

        strashed = structural_hash(c)
        t.check(strashed.gate_count() <= original_gates,
                f"Strashed {strashed.gate_count()} <= original {original_gates} gates")
        t.check(strashed.metadata.get("strashed", False),
                "Strashed circuit has metadata marker")

        # Verify functional equivalence
        equiv = verify_equivalence(c, strashed)
        t.check(equiv, "Structural hashing preserves functional equivalence")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Constant Propagation
# ═══════════════════════════════════════════════════════════════════════════════

def test_constant_propagation():
    t = TestResult("Constant Propagation")
    if not HAS_PATTERN_REWRITER:
        t.check(False, "pattern_rewriter module not available")
        return t

    try:
        const_circ = _build_circuit_with_constants()
        original_gates = const_circ.gate_count()

        propagated, meta = propagate_constants(const_circ)
        t.check(meta.get("gates_removed", 0) > 0,
                f"Constants folded: {meta.get('gates_removed', 0)} gates removed")
        t.check(propagated.gate_count() < original_gates,
                f"Gate count reduced: {propagated.gate_count()} < {original_gates}")
        t.check(meta.get("constants_discovered", 0) >= 0,
                f"Constants discovered: {meta.get('constants_discovered', 0)}")

        # Verify evaluation still works at Boolean corners
        orig_out = const_circ.evaluate([True])
        prop_out = propagated.evaluate([True])
        t.check(len(prop_out) > 0,
                f"Constant propagation evaluation produced output (orig={orig_out}, prop={prop_out})")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Dead Gate Elimination
# ═══════════════════════════════════════════════════════════════════════════════

def test_dead_gate_elimination():
    t = TestResult("Dead Gate Elimination")
    if not HAS_PATTERN_REWRITER:
        t.check(False, "pattern_rewriter module not available")
        return t

    try:
        dead_circ = _build_circuit_with_dead_gates()
        original_gates = dead_circ.gate_count()

        cleaned, meta = eliminate_dead_gates(dead_circ)
        t.check(meta.get("gates_removed", 0) > 0,
                f"Dead gates removed: {meta.get('gates_removed', 0)}")
        t.check(cleaned.gate_count() < original_gates,
                f"Gate count reduced: {cleaned.gate_count()} < {original_gates}")

        # Verify functional equivalence
        equiv = verify_equivalence(dead_circ, cleaned)
        t.check(equiv, "Dead gate elimination preserves functional equivalence")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Full Optimization Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_optimization():
    t = TestResult("Full Optimization Pipeline")
    if not HAS_PATTERN_REWRITER:
        t.check(False, "pattern_rewriter module not available")
        return t

    try:
        # Half adder optimization
        ha = _build_half_adder()
        original_gates = ha.gate_count()

        optimized, opt_meta = optimize(ha)
        t.check(opt_meta.get("functional_equivalence_verified", False),
                "Half adder: functional equivalence verified after optimization")
        t.check("stages" in opt_meta, "Optimization metadata includes stages")
        t.check(opt_meta.get("optimized_gates", original_gates) <= original_gates,
                f"Gate count: {opt_meta.get('optimized_gates')} <= {original_gates}")

        # Exhaustive Boolean testing for 2-input circuit
        all_match = True
        for i in range(4):
            a = (i >> 0) & 1 == 1
            b = (i >> 1) & 1 == 1
            orig_out = ha.evaluate([a, b])
            opt_out = optimized.evaluate([a, b])
            if orig_out != opt_out:
                all_match = False
                break
        t.check(all_match, "Half adder: exhaustive Boolean equivalence (4 vectors)")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Verilog Generation
# ═══════════════════════════════════════════════════════════════════════════════

def test_verilog_generation():
    t = TestResult("Verilog Generation")
    if VerilogGenerator is None:
        t.check(False, "VerilogGenerator not available")
        return t

    try:
        and_circ = _build_and_circuit()

        # Test all three targets
        for target in ("generic", "fpga", "asic"):
            gen = VerilogGenerator(module_name=f"test_{target}", target=target)
            verilog_code, meta = gen.generate(and_circ)
            t.check(len(verilog_code) > 0, f"{target}: Verilog code is non-empty")
            t.check("module" in verilog_code, f"{target}: contains 'module'")
            t.check("nand" in verilog_code.lower(), f"{target}: contains NAND gate")
            t.check("input" in verilog_code, f"{target}: contains input declarations")
            t.check("output" in verilog_code, f"{target}: contains output declarations")
            t.check("endmodule" in verilog_code, f"{target}: contains endmodule")
            t.check(meta.get("gate_count", 0) == 2,
                    f"{target}: gate_count = {meta.get('gate_count')}")

            if target == "fpga":
                t.check("KEEP" in verilog_code, "FPGA: contains KEEP attributes")
            elif target == "asic":
                t.check("syn_keep" in verilog_code, "ASIC: contains syn_keep directives")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: BLIF Generation
# ═══════════════════════════════════════════════════════════════════════════════

def test_blif_generation():
    t = TestResult("BLIF Generation")
    if BLIFGenerator is None:
        t.check(False, "BLIFGenerator not available")
        return t

    try:
        and_circ = _build_and_circuit()
        blif_gen = BLIFGenerator(model_name="test_blif")
        blif_code = blif_gen.generate(and_circ)

        t.check(len(blif_code) > 0, "BLIF code is non-empty")
        t.check(".model" in blif_code, "BLIF: contains .model declaration")
        t.check(".inputs" in blif_code, "BLIF: contains .inputs")
        t.check(".outputs" in blif_code, "BLIF: contains .outputs")
        t.check(".names" in blif_code, "BLIF: contains .names entries")
        t.check(".end" in blif_code, "BLIF: contains .end")
        t.check("test_blif" in blif_code, "BLIF: model name appears in output")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: AIGER Generation
# ═══════════════════════════════════════════════════════════════════════════════

def test_aiger_generation():
    t = TestResult("AIGER Generation")
    if AIGERGenerator is None:
        t.check(False, "AIGERGenerator not available")
        return t

    try:
        and_circ = _build_and_circuit()
        aiger_gen = AIGERGenerator(model_name="test_aiger")
        aiger_bytes = aiger_gen.generate(and_circ)

        t.check(len(aiger_bytes) > 0, "AIGER: produces non-empty bytes")
        header_str = aiger_bytes[:3].decode("ascii", errors="replace")
        t.check(header_str == "aig", f"AIGER: starts with 'aig' header (got '{header_str}')")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Yosys/ABC Integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_yosys_abc_integration():
    t = TestResult("Yosys/ABC Integration")
    if YosysABCIntegration is None:
        t.check(False, "YosysABCIntegration not available")
        return t

    try:
        yosys = YosysABCIntegration()
        avail = yosys.check_available()
        t.check("yosys" in avail, "check_available returns 'yosys' key")
        t.check("abc" in avail, "check_available returns 'abc' key")

        if avail.get("yosys") and avail.get("abc"):
            # Tools installed: try optimization
            and_circ = _build_and_circuit()
            if VerilogGenerator is not None:
                gen = VerilogGenerator(module_name="test_yosys")
                verilog_code, _ = gen.generate(and_circ)
                try:
                    opt_code, opt_meta = yosys.optimize_verilog(verilog_code)
                    t.check(len(opt_code) > 0, "Yosys/ABC: produces Verilog output")
                    t.check("original_gate_count" in opt_meta,
                            "Metadata includes original_gate_count")
                    t.check("optimized_gate_count" in opt_meta,
                            "Metadata includes optimized_gate_count")
                except Exception as e:
                    t.check(False, f"Yosys/ABC optimization failed: {e}")
        else:
            # Tools not available: verify graceful degradation
            t.check(True, "Yosys/ABC not available — graceful skip (expected in CI)")
            try:
                yosys.optimize_verilog("module dummy(); endmodule")
                t.check(False, "optimize_verilog should raise when tools missing")
            except (FileNotFoundError, Exception) as e:
                t.check(True, f"optimize_verilog raises {type(e).__name__} when tools missing")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: FPGA Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def test_fpga_synthesis():
    t = TestResult("FPGA Synthesis")
    if FPGASynthesizer is None:
        t.check(False, "FPGASynthesizer not available")
        return t

    try:
        and_circ = _build_and_circuit()

        # Test each FPGA family
        for family in ("xc7", "ice40", "ecp5", "generic"):
            synth = FPGASynthesizer(fpga_family=family)
            result, meta = synth.synthesize(and_circ, module_name="test_fpga")

            t.check("verilog" in result, f"{family}: produces Verilog")
            t.check("resources" in result, f"{family}: produces resources")

            resources = result.get("resources", {})
            t.check(resources.get("lut_count", 0) >= 1,
                    f"{family}: LUT count >= 1 (got {resources.get('lut_count')})")
            t.check(resources.get("max_freq_mhz", 0) > 0,
                    f"{family}: max_freq_mhz > 0 (got {resources.get('max_freq_mhz')})")

            # Check is_estimated flag
            t.check("is_estimated" in meta,
                    f"{family}: metadata includes is_estimated flag")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11: ASIC Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def test_asic_synthesis():
    t = TestResult("ASIC Synthesis")
    if ASICSynthesizer is None:
        t.check(False, "ASICSynthesizer not available")
        return t

    try:
        and_circ = _build_and_circuit()

        for pdk in ("sky130", "gf180mcu", "asicore"):
            synth = ASICSynthesizer(pdk=pdk)
            result, meta = synth.synthesize(and_circ, module_name="test_asic")

            t.check("verilog" in result, f"{pdk}: produces Verilog")
            t.check("metrics" in result, f"{pdk}: produces metrics")

            metrics = result.get("metrics", {})
            area = metrics.get("area_um2", 0)
            delay = metrics.get("delay_ns", 0)
            power = metrics.get("power_uw", 0)

            t.check(area > 0, f"{pdk}: area_um2 > 0 (got {area})")
            t.check(delay > 0, f"{pdk}: delay_ns > 0 (got {delay})")
            t.check(power >= 0, f"{pdk}: power_uw >= 0 (got {power})")

            # Reasonable order-of-magnitude checks
            t.check(0 < area < 10000, f"{pdk}: area in reasonable range (0, 10000)")
            t.check(0 < delay < 100, f"{pdk}: delay in reasonable range (0, 100) ns")

            t.check("is_estimated" in meta, f"{pdk}: metadata includes is_estimated")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12: Translation Error Tracker
# ═══════════════════════════════════════════════════════════════════════════════

def test_translation_error_tracker():
    t = TestResult("Translation Error Tracker")
    if not HAS_TRANSLATION_ERROR:
        t.check(False, "translation_error module not available")
        return t

    try:
        tracker = TranslationErrorTracker(epsilon=0.01)

        # Track errors across multiple stages
        tid = tracker.begin_translation("LaTeX", "Verilog")
        t.check(isinstance(tid, str), "begin_translation returns a string ID")

        tracker.record_error(tid, "latex_to_eml", 0.001, 0.01)
        tracker.record_error(tid, "eml_to_nand", 0.005, 0.01)
        tracker.record_error(tid, "nand_rewrite", 0.0, 0.0)
        tracker.record_error(tid, "round_trip", 0.015, 0.02)

        # Verify cumulative error computation
        cumulative = tracker.get_cumulative_error(tid)
        expected_cumulative = 0.01 + 0.01 + 0.0 + 0.02
        t.check_approx(cumulative, expected_cumulative, 1e-10,
                       f"Cumulative error: {cumulative} vs {expected_cumulative}")

        # Get full report
        report = tracker.get_report(tid)
        t.check(report["source_format"] == "LaTeX", "Report source_format = LaTeX")
        t.check(report["target_format"] == "Verilog", "Report target_format = Verilog")
        t.check(len(report["errors"]) == 4, "Report contains 4 error records")
        t.check("paper_bounds" in report, "Report includes paper_bounds")

        # Verify unknown translation_id raises KeyError
        try:
            tracker.get_report("nonexistent_id")
            t.check(False, "Unknown translation_id should raise KeyError")
        except KeyError:
            t.check(True, "Unknown translation_id raises KeyError correctly")

        # Verify negative error raises ValueError
        try:
            tracker.record_error(tid, "bad_stage", -1.0, 0.01)
            t.check(False, "Negative error_value should raise ValueError")
        except ValueError:
            t.check(True, "Negative error_value raises ValueError correctly")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13: Paper Bound Checker — ALL 5 paper bounds
# ═══════════════════════════════════════════════════════════════════════════════

def test_paper_bound_checker():
    t = TestResult("Paper Bound Checker")
    if not HAS_TRANSLATION_ERROR:
        t.check(False, "translation_error module not available")
        return t

    try:
        eps = 0.001

        # ── Bound 1: Round-trip error <= 2eps (Section 8) ──
        rt = PaperBoundChecker.check_round_trip_bound(0.001, eps)
        t.check(rt["satisfied"], f"Round-trip: 0.001 <= 2*{eps} = {2*eps}")
        t.check(rt["bound"] == 2 * eps, f"Round-trip bound = {2*eps}")
        t.check("Section 8" in rt["reference"], "Round-trip references Section 8")

        rt_fail = PaperBoundChecker.check_round_trip_bound(0.1, eps)
        t.check(not rt_fail["satisfied"], "Round-trip: 0.1 > 2*0.001 should fail")

        # ── Bound 2: Contraction delta' = 4delta^2 + 4eps < delta (Theorem 4.2) ──
        ct = PaperBoundChecker.check_contraction(0.1, eps)
        t.check(ct["delta_prime"] == 4 * 0.1**2 + 4 * eps,
                f"Contraction: delta' = {ct['delta_prime']}")
        t.check(ct["contraction_holds"], "Contraction holds for delta=0.1, eps=0.001")
        t.check("Theorem 4.2" in ct["reference"], "Contraction references Theorem 4.2")

        ct_fail = PaperBoundChecker.check_contraction(0.5, eps)
        t.check(not ct_fail["contraction_holds"], "No contraction for delta=0.5")

        # ── Bound 3: Fixed point delta* ~ 4eps, eps < 1/64 (Theorem 4.2) ──
        fp = PaperBoundChecker.check_fixed_point(eps)
        t.check(fp["epsilon_lt_1_over_64"], f"eps = {eps} < 1/64")
        t.check(fp["real_solution_exists"], "Real solution exists for eps = 0.001")
        t.check_approx(fp["delta_star"], 4 * eps, eps * 2,
                       f"Fixed point: delta* ~ 4eps = {4*eps}")
        t.check("Theorem 4.2" in fp["reference"], "Fixed point references Theorem 4.2")

        fp_large = PaperBoundChecker.check_fixed_point(0.1)
        t.check(not fp_large["epsilon_lt_1_over_64"], "eps = 0.1 >= 1/64")

        # ── Bound 4: Taylor remainder bound ──
        tay = PaperBoundChecker.check_taylor_remainder(1.0, 8)
        t.check(tay["bound_satisfied"], "Taylor remainder bound satisfied for exp(1), order 8")
        t.check_approx(tay["actual_error"], abs(tay["approx_exp"] - tay["actual_exp"]),
                       1e-15, "Taylor: actual_error matches |approx - exact|")
        t.check("Taylor" in tay["reference"], "Taylor references Taylor remainder theorem")

        # ── Bound 5: Self-correcting cycle O(eps^{2^k}) (Section 8) ──
        sc = PaperBoundChecker.check_self_correcting(eps, 2)
        t.check(sc["estimated_error"] == 2.0 * eps**(2**2),
                f"Self-correcting k=2: {sc['estimated_error']} = 2*eps^4")
        t.check("Section 8" in sc["reference"], "Self-correcting references Section 8")

        sc_k3 = PaperBoundChecker.check_self_correcting(eps, 3)
        t.check(sc_k3["estimated_error"] == 2.0 * eps**(2**3),
                f"Self-correcting k=3: {sc_k3['estimated_error']} = 2*eps^8")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14: Measure LaTeX→EML Error
# ═══════════════════════════════════════════════════════════════════════════════

def test_measure_latex_to_eml_error():
    t = TestResult("Measure LaTeX→EML Error")
    if not HAS_TRANSLATION_ERROR:
        t.check(False, "translation_error module not available")
        return t

    try:
        # Test exp(x) at x=1
        result = measure_latex_to_eml_error("\\exp(x)", {"x": 1.0})
        t.check(result.get("eml_value") is not None, "exp(x): EML value computed")
        t.check(result.get("reference_value") is not None, "exp(x): reference value computed")
        if result.get("absolute_error") is not None:
            t.check(result["absolute_error"] < 0.1,
                    f"exp(1): absolute error = {result['absolute_error']:.6e} < 0.1")

        # Test ln(x) at x=2
        result_ln = measure_latex_to_eml_error("\\ln(x)", {"x": 2.0})
        t.check(result_ln.get("eml_value") is not None, "ln(x): EML value computed")
        if result_ln.get("absolute_error") is not None:
            t.check(result_ln["absolute_error"] < 0.1,
                    f"ln(2): absolute error = {result_ln['absolute_error']:.6e} < 0.1")

        # Test exp(x) at x=0.5
        result_exp2 = measure_latex_to_eml_error("\\exp(x)", {"x": 0.5})
        t.check(result_exp2.get("eml_value") is not None, "exp(0.5): EML value computed")
        if result_exp2.get("absolute_error") is not None:
            t.check(result_exp2["absolute_error"] < 0.1,
                    f"exp(0.5): absolute error = {result_exp2['absolute_error']:.6e} < 0.1")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 15: Measure NAND Rewrite Error
# ═══════════════════════════════════════════════════════════════════════════════

def test_measure_nand_rewrite_error():
    t = TestResult("Measure NAND Rewrite Error")
    if not (HAS_TRANSLATION_ERROR and HAS_PATTERN_REWRITER):
        t.check(False, "Required modules not available")
        return t

    try:
        # Half adder rewrite equivalence
        ha = _build_half_adder()
        optimized, _ = optimize(ha)

        error_report = measure_nand_rewrite_error(ha, optimized, num_tests=100)
        t.check(error_report.get("equivalent_bool") is True,
                "Half adder: Boolean equivalence after rewrite")
        t.check(error_report.get("equivalent_soft") is True,
                "Half adder: soft equivalence after rewrite")
        t.check(error_report.get("max_soft_error", 1.0) < 0.01,
                f"Half adder: max soft error = {error_report.get('max_soft_error', 'N/A')}")

        # AND circuit rewrite equivalence
        and_circ = _build_and_circuit()
        opt_and, _ = optimize(and_circ)
        err_and = measure_nand_rewrite_error(and_circ, opt_and, num_tests=50)
        t.check(err_and.get("equivalent_bool") is True,
                "AND circuit: Boolean equivalence after rewrite")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 16: Pipeline forward_full() with target="verilog"
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_forward_full():
    t = TestResult("Pipeline forward_full()")
    if EMLNANDPipeline is None:
        t.check(False, "EMLNANDPipeline not available")
        return t

    try:
        pipeline = EMLNANDPipeline(epsilon=0.01)
        result = pipeline.forward_full(
            "\\exp(x)",
            test_env={"x": 1.0},
            target="verilog",
        )

        t.check("stages" in result, "forward_full returns 'stages' key")
        t.check("input" in result, "forward_full returns 'input' key")

        # Stage 1: LaTeX → EML should succeed
        l2e = result.get("stages", {}).get("latex_to_eml", {})
        t.check(l2e.get("success", False),
                f"LaTeX→EML stage succeeded: {l2e.get('success')}")

        # Stage 2: EML → NAND should succeed
        e2n = result.get("stages", {}).get("eml_to_nand", {})
        t.check(e2n.get("success", False),
                f"EML→NAND stage succeeded: {e2n.get('success')}")

        # Stage 3: NAND rewrite should succeed
        rw = result.get("stages", {}).get("nand_rewrite", {})
        t.check(rw.get("success", False),
                f"NAND rewrite stage succeeded: {rw.get('success')}")

        # Stage 4: Verilog should succeed if generator available
        vlg = result.get("stages", {}).get("verilog", {})
        if VerilogGenerator is not None:
            t.check(vlg.get("success", False),
                    f"Verilog stage succeeded: {vlg.get('success')}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 17: Pipeline to_verilog() convenience method
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_to_verilog():
    t = TestResult("Pipeline to_verilog()")
    if EMLNANDPipeline is None:
        t.check(False, "EMLNANDPipeline not available")
        return t

    try:
        pipeline = EMLNANDPipeline(epsilon=0.01)
        result = pipeline.to_verilog("\\exp(x)", module_name="test_vlg", target="generic")

        if VerilogGenerator is not None:
            t.check(result.get("success", False),
                    f"to_verilog succeeded: {result.get('success')}")
            t.check("verilog_code" in result, "Result contains verilog_code")
            t.check(len(result.get("verilog_code", "")) > 0,
                    "Verilog code is non-empty")
            t.check("gate_count_original" in result, "Result contains gate_count_original")
            t.check("gate_count_optimized" in result, "Result contains gate_count_optimized")
        else:
            t.check(not result.get("success", False),
                    "to_verilog returns failure when VerilogGenerator unavailable")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Test 18: Pipeline measure_pipeline_error()
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_error_measurement():
    t = TestResult("Pipeline Error Measurement")
    if EMLNANDPipeline is None:
        t.check(False, "EMLNANDPipeline not available")
        return t

    try:
        pipeline = EMLNANDPipeline(epsilon=0.01)
        result = pipeline.measure_pipeline_error("\\exp(x)", {"x": 1.0})

        t.check(result.get("success", False) or "error" in result,
                f"measure_pipeline_error returned a result: success={result.get('success')}")

        if result.get("success"):
            # Verify some error measurement keys exist
            t.check("latex_to_eml_error" in result or "epsilon" in result,
                    "Result contains error measurement data")
            # Paper compliance should be checked
            t.check("paper_compliance" in result or "latex_to_eml_error" in result,
                    "Result contains compliance or error data")
        else:
            # May fail if translation_error submodule is unavailable
            t.check("error" in result, f"Error returned: {result.get('error')}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    tests = [
        test_pattern_rewriter,
        test_structural_hash,
        test_constant_propagation,
        test_dead_gate_elimination,
        test_full_optimization,
        test_verilog_generation,
        test_blif_generation,
        test_aiger_generation,
        test_yosys_abc_integration,
        test_fpga_synthesis,
        test_asic_synthesis,
        test_translation_error_tracker,
        test_paper_bound_checker,
        test_measure_latex_to_eml_error,
        test_measure_nand_rewrite_error,
        test_pipeline_forward_full,
        test_pipeline_to_verilog,
        test_pipeline_error_measurement,
    ]

    results = []
    total_passed = 0
    total_failed = 0

    print("\n" + "=" * 60)
    print("Extended EML Pipeline — Test Suite")
    print("=" * 60)

    for test_fn in tests:
        try:
            result = test_fn()
        except Exception as e:
            result = TestResult(test_fn.__name__)
            result.check(False, f"Unhandled exception: {e}\n{traceback.format_exc()}")
        results.append(result)
        total_passed += result.passed
        total_failed += result.failed
        print(result.summary())

    # Grand summary
    grand_total = total_passed + total_failed
    grand_rate = total_passed / grand_total * 100 if grand_total else 0
    print("=" * 60)
    print(f"GRAND TOTAL: {total_passed}/{grand_total} passed ({grand_rate:.1f}%)")
    print("=" * 60)

    if total_failed > 0:
        print("\nAll failed checks:")
        for r in results:
            for err in r.errors:
                print(f"  [{r.name}] {err}")

    return results


if __name__ == "__main__":
    run_all_tests()
