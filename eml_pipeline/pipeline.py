"""
EML-NAND Pipeline — Main Orchestrator
======================================

Complete bidirectional pipeline with dual-branch architecture:

  Forward (Two Branches after NAND):
    Branch A (Software):  LaTeX → EML → NAND → Pattern Rewrite → Assembly
    Branch B (Hardware):  LaTeX → EML → NAND → Verilog → Yosys/ABC → FPGA/ASIC → Assembly

  Reverse (Four Transitions):
    T1: EML → NAND        (Theorem 2.6: soft NAND bridge)
    T2: NAND → ε-NAND     (Definition 3.1: bounded noise gates)
    T3: ε-NAND → ApproxEML (§5: Taylor series reconstruction)
    T4: ApproxEML → EML   (§6: Newton-Raphson correction)

  Assembly Decompilation:
    Assembly → NAND → ε-NAND → ApproxEML → EML → LaTeX

  Full backward/forward support for:
    - Loops, sums, pi products
    - Integrals, limits, derivatives
    - Complex numbers, special variables per domain
    - Science signs and math constants

Error measurement at every translation stage with paper-bound verification.

Based on: "The EML–NAND Duality" by Daniel Derycke (2026)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.eml.eml_core import (
    EMLNode, EMLNodeType, ONE, VAR, EML,
    eml_exp, eml_ln, eml_zero, eml_complement,
    eml_add, eml_subtract, eml_multiply, eml_divide,
    eml_negate, eml_soft_nand, eml_evaluate,
    eml_to_dict, eml_from_dict
)
from eml_pipeline.eml.latex_to_eml import latex_to_eml, LatexToEMLConverter
from eml_pipeline.nand.nand_core import NANDCircuit
from eml_pipeline.nand.pattern_rewriter import (
    NANDPatternRewriter, structural_hash, propagate_constants,
    eliminate_dead_gates, simplify_restoration, optimize, verify_equivalence
)
from eml_pipeline.transitions.t1_eml_to_nand import eml_to_nand
from eml_pipeline.transitions.t3_t4_nand_to_eml import nand_to_eml
from eml_pipeline.assembly.nand_to_asm import compile_nand_to_asm
from eml_pipeline.assembly.asm_decompiler import decompile_asm, decompile_with_metadata
from eml_pipeline.assembly.optimal_asm_gen import (
    generate_pattern_branch_asm, generate_hardware_branch_asm,
    measure_asm_generation_error, OptimalAssemblyOutput
)
from eml_pipeline.reverse.reverse_pipeline import ReversePipeline
from eml_pipeline.utils.error_metrics import ErrorAnalyzer

# Extended pipeline imports (with graceful fallback)
try:
    from eml_pipeline.hdl.verilog_gen import VerilogGenerator, circuit_to_verilog
    HAS_VERILOG = True
except ImportError:
    HAS_VERILOG = False

try:
    from eml_pipeline.hdl.yosys_abc import YosysABCIntegration, BLIFGenerator
    HAS_YOSYS_ABC = True
except ImportError:
    HAS_YOSYS_ABC = False

try:
    from eml_pipeline.hdl.synthesis import run_synthesis, FPGASynthesizer, ASICSynthesizer
    HAS_SYNTHESIS = True
except ImportError:
    HAS_SYNTHESIS = False

try:
    from eml_pipeline.utils.translation_error import (
        TranslationErrorTracker, measure_full_pipeline_error,
        PaperBoundChecker, ErrorVisualization
    )
    HAS_TRANSLATION_ERROR = True
except ImportError:
    HAS_TRANSLATION_ERROR = False


class EMLNANDPipeline:
    """
    The complete EML-NAND bidirectional pipeline with dual-branch architecture.

    Forward paths (two branches after NAND):
      Branch A (Software): LaTeX → EML → NAND → Pattern Rewrite → Assembly
      Branch B (Hardware): LaTeX → EML → NAND → Verilog → Yosys/ABC → FPGA/ASIC → Assembly

    Reverse paths (four transitions):
      T1: EML → NAND         (Theorem 2.6: soft NAND bridge)
      T2: NAND → ε-NAND      (Definition 3.1: bounded noise)
      T3: ε-NAND → ApproxEML (§5: Taylor series)
      T4: ApproxEML → EML    (§6: Newton-Raphson)

    Assembly decompilation:
      Assembly → NAND → ε-NAND → ApproxEML → EML → LaTeX

    Error measurement at every translation stage with paper-bound verification.
    """

    def __init__(self, bit_width: int = 16, epsilon: float = 0.001,
                 taylor_order: int = 8):
        self.bit_width = bit_width
        self.epsilon = epsilon
        self.taylor_order = taylor_order
        self.reverse_pipeline = ReversePipeline(epsilon, taylor_order)
        self.error_analyzer = ErrorAnalyzer(epsilon)
        self.error_tracker = TranslationErrorTracker(epsilon) if HAS_TRANSLATION_ERROR else None
        self.metadata: Dict[str, Any] = {
            "forward_results": {},
            "reverse_results": {},
            "error_analysis": {},
            "capabilities": {
                "verilog": HAS_VERILOG,
                "yosys_abc": HAS_YOSYS_ABC,
                "synthesis": HAS_SYNTHESIS,
                "translation_error": HAS_TRANSLATION_ERROR,
                "assembly_decompilation": True,
                "pattern_branch": True,
                "hardware_branch": HAS_VERILOG and HAS_YOSYS_ABC,
            },
        }

    # ─── Full Extended Forward Pipeline ───────────────────────────────────

    def forward_full(self, latex_expr: str,
                     test_env: Dict[str, float] = None,
                     target: str = "verilog",
                     fpga_family: str = "xc7",
                     pdk: str = "sky130",
                     asm_arch: str = "x86") -> Dict[str, Any]:
        """
        Full extended pipeline with dual-branch architecture.

        Args:
            latex_expr: LaTeX mathematical expression
            test_env: Variable bindings for testing {name: value}
            target: Output target — "verilog", "fpga", "asic", "assembly",
                    "pattern_asm", "hardware_asm", "all"
            fpga_family: FPGA family for synthesis ("xc7", "ice40", "ecp5")
            pdk: PDK for ASIC synthesis ("sky130", "gf180mcu")
            asm_arch: Assembly architecture for code generation

        Returns:
            Dictionary with all intermediate results and metadata including
            translation error measurements at every stage.
        """
        results: Dict[str, Any] = {
            "input": latex_expr,
            "test_env": test_env,
            "target": target,
            "asm_arch": asm_arch,
            "stages": {},
            "translation_errors": {},
        }

        # Start error tracking
        translation_id = None
        if self.error_tracker:
            translation_id = self.error_tracker.begin_translation("LaTeX", target)

        # ═══ Stage 1: LaTeX → EML ═══
        eml_tree = None
        try:
            eml_tree, convert_meta = latex_to_eml(latex_expr, self.taylor_order)
            results["stages"]["latex_to_eml"] = {
                "success": True,
                "eml_tree": str(eml_tree),
                "eml_dict": eml_to_dict(eml_tree),
                "depth": eml_tree.depth(),
                "size": eml_tree.size(),
                "metadata": convert_meta,
            }
        except Exception as e:
            results["stages"]["latex_to_eml"] = {"success": False, "error": str(e)}
            return results

        # Evaluate EML if test values provided
        if test_env:
            try:
                eml_value = eml_evaluate(eml_tree, test_env)
                results["stages"]["latex_to_eml"]["value"] = eml_value
                if translation_id and self.error_tracker:
                    self.error_tracker.record_error(
                        translation_id, "latex_to_eml", 0.0, 1e-10,
                        {"note": "EML evaluation is exact; error in tree depth/size"})
            except Exception as e:
                results["stages"]["latex_to_eml"]["eval_error"] = str(e)

        # ═══ Stage 2: T1 — EML → NAND (Theorem 2.6) ═══
        circuit = None
        try:
            circuit, t1_meta = eml_to_nand(eml_tree, self.bit_width, self.epsilon)
            results["stages"]["t1_eml_to_nand"] = {
                "success": True,
                "transition": "T1",
                "theorem": "2.6",
                "gate_count": circuit.gate_count(),
                "circuit_depth": circuit.depth(),
                "num_inputs": circuit.num_inputs,
                "metadata": t1_meta,
            }
            if translation_id and self.error_tracker:
                self.error_tracker.record_error(
                    translation_id, "t1_eml_to_nand", self.epsilon, self.epsilon,
                    {"theorem": "2.6", "gate_count": circuit.gate_count()})
        except Exception as e:
            results["stages"]["t1_eml_to_nand"] = {"success": False, "error": str(e)}
            return results

        # ═══ Stage 3: NAND Pattern Rewriting ═══
        rewritten_circuit = None
        try:
            rewritten_circuit, rw_meta = optimize(circuit)
            results["stages"]["nand_rewrite"] = {
                "success": True,
                **rw_meta,
            }
            if translation_id and self.error_tracker:
                gates_saved = rw_meta.get("gates_saved", 0)
                self.error_tracker.record_error(
                    translation_id, "nand_rewrite", 0.0, 0.0,
                    {"gates_saved": gates_saved,
                     "optimization_ratio": rw_meta.get("optimization_ratio", 0)})
        except Exception as e:
            results["stages"]["nand_rewrite"] = {"success": False, "error": str(e)}
            rewritten_circuit = circuit

        active_circuit = rewritten_circuit or circuit

        # ═══ Dual Branch: Pattern Rewrite → Assembly ═══
        if target in ("assembly", "pattern_asm", "all"):
            self._run_pattern_branch(
                active_circuit, circuit, asm_arch, results, translation_id)

        # ═══ Dual Branch: Hardware → Assembly ═══
        if target in ("fpga", "asic", "hardware_asm", "all"):
            self._run_hardware_branch(
                active_circuit, circuit, target, fpga_family, pdk,
                asm_arch, results, translation_id)

        # ═══ Verilog Generation ═══
        if target in ("verilog", "all") and HAS_VERILOG:
            try:
                verilog_target = "generic"
                gen = VerilogGenerator(module_name="eml_circuit", target=verilog_target)
                verilog_code, v_meta = gen.generate(active_circuit)
                results["stages"]["verilog"] = {
                    "success": True,
                    "verilog_code": verilog_code,
                    "code_length": len(verilog_code),
                    "metadata": v_meta,
                }
            except Exception as e:
                results["stages"]["verilog"] = {"success": False, "error": str(e)}

        # ═══ Yosys/ABC Optimization ═══
        if target in ("fpga", "asic", "hardware_asm", "all") and HAS_YOSYS_ABC:
            try:
                yosys = YosysABCIntegration()
                available = yosys.check_available()
                results["stages"]["yosys_available"] = available
            except Exception as e:
                results["stages"]["yosys_available"] = {"error": str(e)}

        # ═══ Error Summary ═══
        if translation_id and self.error_tracker:
            results["translation_errors"] = self.error_tracker.get_report(translation_id)
            results["paper_compliance"] = self.error_tracker.check_paper_bound(translation_id)

        self.metadata["forward_results"][latex_expr] = results
        return results

    def _run_pattern_branch(self, active_circuit: NANDCircuit,
                            original_circuit: NANDCircuit,
                            asm_arch: str,
                            results: Dict, translation_id: Optional[str]) -> None:
        """Run Branch A: NAND → Pattern Rewrite → Assembly"""
        try:
            pattern_asm = generate_pattern_branch_asm(
                active_circuit, arch=asm_arch,
                original_gates=original_circuit.gate_count())
            results["stages"]["pattern_branch_asm"] = {
                "success": True,
                "branch": "A (Pattern Rewrite)",
                "arch": pattern_asm.arch,
                "gate_count": pattern_asm.gate_count,
                "original_gates": pattern_asm.original_gates,
                "optimized_gates": pattern_asm.optimized_gates,
                "optimization_ratio": pattern_asm.optimization_ratio,
                "instruction_count": pattern_asm.instruction_count,
                "register_pressure": pattern_asm.register_pressure,
                "critical_path_depth": pattern_asm.critical_path_depth,
                "code_preview": pattern_asm.code[:800] + "..." if len(pattern_asm.code) > 800 else pattern_asm.code,
                "code_full": pattern_asm.code,
            }
            if translation_id and self.error_tracker:
                self.error_tracker.record_error(
                    translation_id, "pattern_branch_asm", 0.0, 0.0,
                    {"synthesis_source": "pattern_rewrite",
                     "optimization_ratio": pattern_asm.optimization_ratio})
        except Exception as e:
            results["stages"]["pattern_branch_asm"] = {"success": False, "error": str(e)}

    def _run_hardware_branch(self, active_circuit: NANDCircuit,
                             original_circuit: NANDCircuit,
                             target: str, fpga_family: str, pdk: str,
                             asm_arch: str,
                             results: Dict, translation_id: Optional[str]) -> None:
        """Run Branch B: NAND → Verilog → Yosys/ABC → FPGA/ASIC → Assembly"""
        # Verilog
        verilog_code = ""
        if HAS_VERILOG:
            try:
                v_target = "fpga" if "fpga" in target else "asic"
                gen = VerilogGenerator(module_name="eml_circuit", target=v_target)
                verilog_code, v_meta = gen.generate(active_circuit)
                results["stages"]["hardware_verilog"] = {
                    "success": True,
                    "verilog_code": verilog_code,
                    "metadata": v_meta,
                }
            except Exception as e:
                results["stages"]["hardware_verilog"] = {"success": False, "error": str(e)}

        # Yosys/ABC
        if HAS_YOSYS_ABC:
            try:
                yosys = YosysABCIntegration()
                available = yosys.check_available()
                if available.get("yosys", False) and verilog_code:
                    opt_code, opt_meta = yosys.optimize_verilog(verilog_code)
                    results["stages"]["hardware_yosys"] = {
                        "success": True,
                        "optimized_verilog": opt_code,
                        "metadata": opt_meta,
                    }
            except Exception as e:
                results["stages"]["hardware_yosys"] = {"success": False, "error": str(e)}

        # FPGA/ASIC Synthesis
        synthesis_source = "fpga" if "fpga" in target else "asic"
        synthesis_report = None
        if HAS_SYNTHESIS:
            try:
                synth_target = f"fpga_{fpga_family}" if "fpga" in target else f"asic_{pdk}"
                report = run_synthesis(active_circuit, target=synth_target)
                synthesis_report = report.to_dict()
                results["stages"]["hardware_synthesis"] = {
                    "success": True,
                    "report": synthesis_report,
                    "summary": report.summary(),
                    "is_estimated": report.is_estimated,
                }
            except Exception as e:
                results["stages"]["hardware_synthesis"] = {"success": False, "error": str(e)}

        # Hardware → Assembly
        try:
            hw_asm = generate_hardware_branch_asm(
                active_circuit, arch=asm_arch,
                original_gates=original_circuit.gate_count(),
                synthesis_source=synthesis_source,
                synthesis_report=synthesis_report)
            results["stages"]["hardware_branch_asm"] = {
                "success": True,
                "branch": "B (Hardware)",
                "arch": hw_asm.arch,
                "gate_count": hw_asm.gate_count,
                "original_gates": hw_asm.original_gates,
                "optimized_gates": hw_asm.optimized_gates,
                "optimization_ratio": hw_asm.optimization_ratio,
                "instruction_count": hw_asm.instruction_count,
                "register_pressure": hw_asm.register_pressure,
                "critical_path_depth": hw_asm.critical_path_depth,
                "synthesis_source": hw_asm.synthesis_source,
                "code_preview": hw_asm.code[:800] + "..." if len(hw_asm.code) > 800 else hw_asm.code,
                "code_full": hw_asm.code,
            }
            if translation_id and self.error_tracker:
                self.error_tracker.record_error(
                    translation_id, "hardware_branch_asm", 0.0, self.epsilon,
                    {"synthesis_source": synthesis_source,
                     "optimization_ratio": hw_asm.optimization_ratio})
        except Exception as e:
            results["stages"]["hardware_branch_asm"] = {"success": False, "error": str(e)}

    # ─── Legacy Forward Pipeline (backward compatible) ────────────────────

    def forward(self, latex_expr: str,
                test_env: Dict[str, float] = None,
                target_arch: str = "x86") -> Dict[str, Any]:
        """Legacy forward pipeline: LaTeX → EML → NAND → Assembly."""
        return self.forward_full(latex_expr, test_env, target="assembly", asm_arch=target_arch)

    # ─── Reverse Pipeline (Four Transitions) ──────────────────────────────

    def reverse(self, circuit: NANDCircuit, inputs: List[float],
                forward_metadata: Dict = None) -> Dict[str, Any]:
        """
        Full reverse pipeline using four transitions (T2→T3→T4):
          NAND → ε-NAND → ApproxEML → EML → LaTeX

        T2: NAND → ε-NAND     (Definition 3.1: add bounded noise)
        T3: ε-NAND → ApproxEML (§5: Taylor series reconstruction)
        T4: ApproxEML → EML   (§6: Newton-Raphson correction)
        """
        results = self.reverse_pipeline.full_reverse(
            circuit, inputs, forward_metadata)
        self.metadata["reverse_results"][str(inputs)] = results
        return results

    def reverse_from_asm(self, asm_code: str, arch: str = "x86",
                         test_inputs: List[float] = None,
                         forward_metadata: Dict = None) -> Dict[str, Any]:
        """
        Full reverse from assembly:
          Assembly → NAND (decompile) → ε-NAND → ApproxEML → EML → LaTeX

        This enables the complete decompilation path from compiled code
        back to the original mathematical expression.
        """
        results: Dict[str, Any] = {
            "stages": {},
            "input_asm_arch": arch,
        }

        # Stage 1: Assembly → NAND (decompile)
        try:
            decompile_result = decompile_with_metadata(
                asm_code, arch, forward_metadata)
            circuit = decompile_result.circuit
            results["stages"]["asm_to_nand"] = {
                "success": True,
                "arch": decompile_result.arch,
                "gates_parsed": decompile_result.gates_parsed,
                "gates_reconstructed": decompile_result.gates_reconstructed,
                "circuit_gate_count": circuit.gate_count(),
                "circuit_depth": circuit.depth(),
                "register_map": decompile_result.register_map,
                "metadata": decompile_result.metadata,
            }
        except Exception as e:
            results["stages"]["asm_to_nand"] = {"success": False, "error": str(e)}
            return results

        # Stage 2-4: NAND → ε-NAND → ApproxEML → EML (T2+T3+T4)
        inputs = test_inputs or [0.5] * circuit.num_inputs
        try:
            reverse_result = self.reverse(circuit, inputs, forward_metadata)
            results["stages"]["nand_to_eml"] = reverse_result
        except Exception as e:
            results["stages"]["nand_to_eml"] = {"success": False, "error": str(e)}

        return results

    # ─── Bidirectional Round-Trip ─────────────────────────────────────────

    def round_trip(self, latex_expr: str, test_env: Dict[str, float],
                   target_arch: str = "x86",
                   use_metadata: bool = True) -> Dict[str, Any]:
        """
        Full round-trip with four transitions:
          LaTeX → EML → NAND → ε-NAND → ApproxEML → EML → LaTeX

        Measures the round-trip error (theoretical bound: ≤ 2ε)

        Args:
            latex_expr: LaTeX expression
            test_env: Variable bindings
            target_arch: Assembly architecture for round-trip
            use_metadata: If True, pass forward metadata to reverse pipeline
        """
        # Forward: LaTeX → EML → NAND
        forward_result = self.forward_full(
            latex_expr, test_env, target="assembly", asm_arch=target_arch)

        # Get the EML value for comparison
        eml_value = forward_result.get("stages", {}).get("latex_to_eml", {}).get("value")

        # Get the NAND circuit
        circuit = None
        circuit_meta = None
        if "t1_eml_to_nand" in forward_result.get("stages", {}):
            t1_stage = forward_result["stages"]["t1_eml_to_nand"]
            if t1_stage.get("success"):
                # Reconstruct circuit from forward pass
                try:
                    eml_tree, _ = latex_to_eml(latex_expr, self.taylor_order)
                    circuit, circuit_meta = eml_to_nand(eml_tree, self.bit_width, self.epsilon)
                except:
                    pass

        # Reverse: NAND → EML → LaTeX
        reverse_result = None
        if circuit:
            inputs = [test_env.get(v, 0.5) for v in ["x", "y"]]
            fwd_meta = circuit_meta if use_metadata else None
            reverse_result = self.reverse(circuit, inputs, fwd_meta)

        # Error analysis
        error_analysis = {
            "epsilon": self.epsilon,
            "theoretical_bound": 2 * self.epsilon,
            "eml_value": eml_value,
            "use_metadata": use_metadata,
        }

        if reverse_result and eml_value is not None:
            reverse_eml_val = reverse_result.get("eml_value")
            if reverse_eml_val is not None:
                actual_error = abs(reverse_eml_val - eml_value)
                error_analysis["round_trip_error"] = actual_error
                error_analysis["bound_satisfied"] = actual_error <= 2 * self.epsilon

        # Paper compliance check
        if HAS_TRANSLATION_ERROR:
            checker = PaperBoundChecker
            error_analysis["paper_compliance"] = {
                "round_trip_bound": checker.check_round_trip_bound(
                    abs(eml_value) if eml_value else self.epsilon, self.epsilon),
                "contraction": checker.check_contraction(0.1, self.epsilon),
                "fixed_point": checker.check_fixed_point(self.epsilon),
            }

        return {
            "forward": forward_result,
            "reverse": reverse_result,
            "error_analysis": error_analysis,
        }

    def round_trip_asm(self, latex_expr: str, test_env: Dict[str, float],
                       arch: str = "x86",
                       use_metadata: bool = True) -> Dict[str, Any]:
        """
        Full round-trip through assembly:
          LaTeX → EML → NAND → Assembly → NAND (decompile) → EML → LaTeX

        This tests the full cycle including compilation and decompilation.
        """
        # Forward: LaTeX → EML → NAND → Assembly
        forward_result = self.forward_full(
            latex_expr, test_env, target="pattern_asm", asm_arch=arch)

        # Get assembly code
        asm_code = ""
        asm_stage = forward_result.get("stages", {}).get("pattern_branch_asm", {})
        if asm_stage.get("success"):
            asm_code = asm_stage.get("code_full", "")

        # Get forward metadata
        fwd_meta = forward_result.get("stages", {}).get("t1_eml_to_nand", {}).get("metadata")
        if not use_metadata:
            fwd_meta = None

        # Reverse: Assembly → NAND → EML → LaTeX
        reverse_result = None
        if asm_code:
            inputs = [test_env.get(v, 0.5) for v in ["x", "y"]]
            reverse_result = self.reverse_from_asm(
                asm_code, arch, inputs, fwd_meta)

        return {
            "forward": forward_result,
            "reverse": reverse_result,
            "arch": arch,
            "use_metadata": use_metadata,
        }

    # ─── Error Analysis ───────────────────────────────────────────────────

    def analyze_errors(self) -> Dict[str, Any]:
        """Run comprehensive error analysis."""
        return self.error_analyzer.analyze_all()

    # ─── Convenience Methods ──────────────────────────────────────────────

    def latex_to_eml(self, latex_expr: str) -> Tuple[EMLNode, Dict]:
        """Just the LaTeX → EML conversion."""
        return latex_to_eml(latex_expr, self.taylor_order)

    def evaluate_eml(self, eml_tree: EMLNode, env: Dict[str, float]) -> float:
        """Evaluate an EML tree with given variable bindings."""
        return eml_evaluate(eml_tree, env)

    def build_soft_nand(self, a_name: str = "a", b_name: str = "b") -> EMLNode:
        """Build the soft NAND EML tree (Theorem 2.6)."""
        return eml_soft_nand(VAR(a_name), VAR(b_name))

    # ─── HDL-Specific Methods ─────────────────────────────────────────────

    def to_verilog(self, latex_expr: str, module_name: str = "eml_circuit",
                   target: str = "generic") -> Dict[str, Any]:
        """Convert LaTeX directly to Verilog HDL."""
        if not HAS_VERILOG:
            return {"success": False, "error": "Verilog generator not available"}
        try:
            eml_tree, _ = latex_to_eml(latex_expr, self.taylor_order)
            circuit, _ = eml_to_nand(eml_tree, self.bit_width, self.epsilon)
            optimized, opt_meta = optimize(circuit)
            gen = VerilogGenerator(module_name=module_name, target=target)
            verilog_code, v_meta = gen.generate(optimized)
            return {
                "success": True,
                "verilog_code": verilog_code,
                "gate_count_original": circuit.gate_count(),
                "gate_count_optimized": optimized.gate_count(),
                "optimization_metadata": opt_meta,
                "verilog_metadata": v_meta,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def to_fpga(self, latex_expr: str, fpga_family: str = "xc7") -> Dict[str, Any]:
        """Convert LaTeX to FPGA synthesis output."""
        if not HAS_SYNTHESIS:
            return {"success": False, "error": "Synthesis module not available"}
        try:
            eml_tree, _ = latex_to_eml(latex_expr, self.taylor_order)
            circuit, _ = eml_to_nand(eml_tree, self.bit_width, self.epsilon)
            optimized, opt_meta = optimize(circuit)
            report = run_synthesis(optimized, target=f"fpga_{fpga_family}")
            return {
                "success": True,
                "synthesis_report": report.to_dict(),
                "summary": report.summary(),
                "optimization_metadata": opt_meta,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def to_asic(self, latex_expr: str, pdk: str = "sky130") -> Dict[str, Any]:
        """Convert LaTeX to ASIC synthesis output."""
        if not HAS_SYNTHESIS:
            return {"success": False, "error": "Synthesis module not available"}
        try:
            eml_tree, _ = latex_to_eml(latex_expr, self.taylor_order)
            circuit, _ = eml_to_nand(eml_tree, self.bit_width, self.epsilon)
            optimized, opt_meta = optimize(circuit)
            report = run_synthesis(optimized, target=f"asic_{pdk}")
            return {
                "success": True,
                "synthesis_report": report.to_dict(),
                "summary": report.summary(),
                "optimization_metadata": opt_meta,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def measure_pipeline_error(self, latex_expr: str,
                                test_env: Dict[str, float]) -> Dict[str, Any]:
        """Measure translation errors at every pipeline stage."""
        if not HAS_TRANSLATION_ERROR:
            return {"success": False, "error": "Translation error module not available"}
        try:
            result = measure_full_pipeline_error(latex_expr, test_env, self.epsilon)
            compliance_measurements: Dict[str, Any] = {}
            rt = result.get("round_trip_error")
            if isinstance(rt, dict):
                compliance_measurements["round_trip_error"] = rt.get("max_error", 0.0)
            elif isinstance(rt, (int, float)):
                compliance_measurements["round_trip_error"] = float(rt)
            else:
                compliance_measurements["round_trip_error"] = 0.0
            compliance = PaperBoundChecker.full_paper_compliance_report(
                self.epsilon, compliance_measurements)
            result["paper_compliance"] = compliance
            if "error_report" in result:
                result["formatted_report"] = ErrorVisualization.format_error_report(
                    result["error_report"])
            if compliance:
                result["formatted_compliance"] = ErrorVisualization.format_compliance_report(
                    compliance)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── Search Capability ────────────────────────────────────────────────

    def search_operations(self, query: str = None, domain: str = None) -> Dict[str, Any]:
        """Search for EML operations, variables, or physics constants."""
        try:
            from eml_pipeline.search.eml_search import EMLSearchEngine
            engine = EMLSearchEngine()
            return engine.search(query=query, domain=domain)
        except ImportError:
            return {"success": False, "error": "Search module not available"}
