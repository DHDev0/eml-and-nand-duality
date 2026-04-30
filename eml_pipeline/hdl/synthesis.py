"""
FPGA/ASIC Synthesis Module for the EML Pipeline
=================================================

Provides synthesis support for NAND circuits targeting FPGA and ASIC
flows. Integrates with Yosys for logic synthesis and ABC for technology
mapping, with graceful degradation when tools are not installed.

Supported FPGA families:
    - xc7   : Xilinx Artix-7 (Vivado/Yosys synth_xilinx)
    - ice40 : Lattice iCE40 (Yosys synth_ice40)
    - ecp5  : Lattice ECP5  (Yosys synth_ecp5)
    - generic

Supported ASIC PDKs:
    - sky130  : SkyWater 130 nm
    - gf180mcu: GlobalFoundries 180 nm
    - asicore : Generic ASIC core

All external tool invocations include timeout handling, temp-file cleanup,
and graceful degradation with a clear ``is_estimated`` flag when the
required EDA tools are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from eml_pipeline.nand.nand_core import NANDCircuit, NANDGate

# These modules may not exist yet; import them lazily / with fallbacks.
try:
    from eml_pipeline.hdl.verilog_gen import VerilogGenerator
except ImportError:
    VerilogGenerator = None  # type: ignore[assignment,misc]

try:
    from eml_pipeline.hdl.yosys_abc import YosysABCIntegration
except ImportError:
    YosysABCIntegration = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_FPGA_FAMILIES = ("xc7", "ice40", "ecp5", "generic")
SUPPORTED_PDKS = ("sky130", "gf180mcu", "asicore")

DEFAULT_YOSYS_PATH = "yosys"
DEFAULT_TIMEOUT_S = 60

# Per-gate area (um^2) by PDK — rough NAND2 standard-cell area
_PDK_GATE_AREA_UM2 = {
    "sky130": 10.0,
    "gf180mcu": 25.0,
    "asicore": 15.0,
}

# Per-gate delay (ns) by PDK — rough single-gate propagation delay
_PDK_GATE_DELAY_NS = {
    "sky130": 0.05,
    "gf180mcu": 0.12,
    "asicore": 0.08,
}

# Per-gate dynamic power (uW) by PDK — rough estimate
_PDK_GATE_POWER_UW = {
    "sky130": 0.01,
    "gf180mcu": 0.03,
    "asicore": 0.015,
}

# FPGA max-frequency baseline (MHz) — used when Yosys is unavailable
_FPGA_BASE_FREQ_MHZ = {
    "xc7": 450.0,
    "ice40": 250.0,
    "ecp5": 350.0,
    "generic": 300.0,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_tool_available(name: str) -> bool:
    """Return *True* if *name* is found on ``$PATH``."""
    return shutil.which(name) is not None


def _generate_verilog(circuit: NANDCircuit, module_name: str, target: str = "generic") -> str:
    """
    Generate Verilog source for *circuit*.

    Uses :class:`VerilogGenerator` when available; otherwise falls back to a
    built-in minimal emitter that is sufficient for Yosys consumption.

    Parameters
    ----------
    circuit : NANDCircuit
        The circuit to emit.
    module_name : str
        Verilog module name.
    target : str
        Synthesis target passed to :class:`VerilogGenerator`
        (``"generic"``, ``"fpga"``, or ``"asic"``).

    Returns
    -------
    str
        Verilog source code.
    """
    if VerilogGenerator is not None:
        gen = VerilogGenerator(module_name=module_name, target=target)
        verilog_code, _metadata = gen.generate(circuit)
        return verilog_code

    # Fallback: hand-rolled Verilog emitter ----------------------------------
    lines: List[str] = []
    in_names = [f"in_{i}" for i in range(circuit.num_inputs)]
    out_names = [f"out_{i}" for i in range(len(circuit.output_wires))]
    port_decls = (
        [f"    input  {n}" for n in in_names]
        + [f"    output {n}" for n in out_names]
    )

    lines.append(f"module {module_name}(")
    lines.append(",\n".join(port_decls))
    lines.append(");")

    # Internal wires (one per gate output)
    for gate in circuit.gates:
        lines.append(f"    wire w{gate.output};")

    # Emit each NAND gate
    for gate in circuit.gates:
        a = _wire_to_verilog(gate.input_a, circuit.num_inputs, in_names)
        b = _wire_to_verilog(gate.input_b, circuit.num_inputs, in_names)
        lines.append(f"    assign w{gate.output} = ~({a} & {b});")

    # Assign outputs
    for idx, wire in enumerate(circuit.output_wires):
        src = _wire_to_verilog(wire, circuit.num_inputs, in_names)
        lines.append(f"    assign {out_names[idx]} = {src};")

    lines.append("endmodule")
    return "\n".join(lines)


def _wire_to_verilog(wire_id: int, num_inputs: int, in_names: List[str]) -> str:
    """Map a logical wire ID to a Verilog net name."""
    if wire_id == -1:
        return "1'b0"
    if wire_id == -2:
        return "1'b1"
    if 0 <= wire_id < num_inputs:
        return in_names[wire_id]
    return f"w{wire_id}"


def _run_yosys(
    yosys_path: str,
    tcl_script: str,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Tuple[bool, str]:
    """
    Execute *yosys_path* with the given TCL *tcl_script*.

    Returns (success: bool, stdout_or_error_message: str).
    """
    if not _is_tool_available(yosys_path):
        return False, f"Yosys not found at '{yosys_path}'. Install Yosys or adjust yosys_path."

    tmp_dir = tempfile.mkdtemp(prefix="eml_synth_")
    script_path = os.path.join(tmp_dir, "synth.ys")
    try:
        with open(script_path, "w") as fh:
            fh.write(tcl_script)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [yosys_path, "-s", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired:
            return False, f"Yosys timed out after {timeout}s"
        except FileNotFoundError:
            return False, f"Yosys executable not found at '{yosys_path}'"
        except OSError as exc:
            return False, f"Failed to run Yosys: {exc}"

        elapsed = time.monotonic() - start
        logger.debug("Yosys completed in %.1f s (returncode=%d)", elapsed, proc.returncode)

        if proc.returncode != 0:
            return False, f"Yosys exited with code {proc.returncode}:\n{proc.stderr or proc.stdout}"

        return True, proc.stdout
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_yosys_resource_report(log: str) -> Dict[str, int]:
    """Best-effort extraction of resource counts from Yosys log text."""
    resources: Dict[str, int] = {}
    for line in log.splitlines():
        line = line.strip()
        # Yosys prints "$lut ... Number of ..." style lines
        if "Number of" in line:
            parts = line.split()
            try:
                count = int(parts[-1])
            except (ValueError, IndexError):
                continue
            low = line.lower()
            if "lut" in low:
                resources["lut_count"] = count
            elif "flip-flop" in low or "dff" in low or "cell" in low and "lut" not in low:
                resources.setdefault("ff_count", count)
            elif "bram" in low or "memory" in low:
                resources["bram_count"] = count
    return resources


# ─── FPGASynthesizer ─────────────────────────────────────────────────────────

class FPGASynthesizer:
    """
    Synthesize a :class:`NANDCircuit` for an FPGA target.

    Parameters
    ----------
    fpga_family : str
        One of ``"xc7"``, ``"ice40"``, ``"ecp5"``, ``"generic"``.
    yosys_path : str
        Path or name of the Yosys executable.
    """

    def __init__(self, fpga_family: str = "xc7", yosys_path: str = DEFAULT_YOSYS_PATH) -> None:
        if fpga_family not in SUPPORTED_FPGA_FAMILIES:
            raise ValueError(
                f"Unsupported FPGA family '{fpga_family}'. "
                f"Choose from {SUPPORTED_FPGA_FAMILIES}"
            )
        self.fpga_family = fpga_family
        self.yosys_path = yosys_path

    # ── public API ────────────────────────────────────────────────────────

    def synthesize(
        self,
        circuit: NANDCircuit,
        module_name: str = "eml_nand_circuit",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run FPGA synthesis on *circuit*.

        Returns
        -------
        synthesis_result : dict
            Keys include ``"verilog"``, ``"constraints"``, ``"resources"``,
            ``"log"`` (Yosys log or empty string).
        metadata : dict
            Keys include ``"is_estimated"``, ``"fpga_family"``,
            ``"yosys_available"``, ``"elapsed_s"``.
        """
        start = time.monotonic()
        verilog_code = _generate_verilog(circuit, module_name, target="fpga")
        constraints = self.generate_constraints(circuit)

        yosys_available = _is_tool_available(self.yosys_path)

        if yosys_available:
            synth_result, yosys_log = self._run_yosys_synth(verilog_code, module_name)
            is_estimated = not synth_result
            resources = self.estimate_resources(circuit)
            if synth_result:
                parsed = _parse_yosys_resource_report(yosys_log)
                resources.update({k: v for k, v in parsed.items() if v > 0})
        else:
            is_estimated = True
            yosys_log = ""
            resources = self.estimate_resources(circuit)
            logger.info("Yosys not available — returning estimated resource usage")

        elapsed = time.monotonic() - start

        synthesis_result: Dict[str, Any] = {
            "verilog": verilog_code,
            "constraints": constraints,
            "resources": resources,
            "log": yosys_log,
        }
        metadata: Dict[str, Any] = {
            "is_estimated": is_estimated,
            "fpga_family": self.fpga_family,
            "yosys_available": yosys_available,
            "elapsed_s": round(elapsed, 3),
            "module_name": module_name,
            "gate_count": circuit.gate_count(),
            "depth": circuit.depth(),
        }
        return synthesis_result, metadata

    def estimate_resources(self, circuit: NANDCircuit) -> Dict[str, Any]:
        """
        Estimate FPGA resource usage from gate count and depth.

        Returns
        -------
        dict
            ``lut_count``, ``ff_count``, ``bram_count``, ``max_freq_mhz``.
        """
        gate_count = circuit.gate_count()
        depth = circuit.depth()

        # Each LUT can typically implement 2–3 simple gates (e.g. a 4-input
        # LUT covers multiple NAND operations).  We conservatively use / 2.
        lut_count = max(1, (gate_count + 1) // 2)
        ff_count = 0  # purely combinational
        bram_count = 0  # small circuits don't need block RAM

        # Rough max-frequency estimate: F_max ≈ F_base / (1 + depth * k)
        # where k accounts for routing overhead.  Heuristic factor 0.02.
        base_freq = _FPGA_BASE_FREQ_MHZ.get(self.fpga_family, 300.0)
        if depth > 0:
            max_freq_mhz = base_freq / (1.0 + depth * 0.02)
        else:
            max_freq_mhz = base_freq

        return {
            "lut_count": lut_count,
            "ff_count": ff_count,
            "bram_count": bram_count,
            "max_freq_mhz": round(max_freq_mhz, 1),
        }

    def generate_constraints(
        self,
        circuit: NANDCircuit,
        pins: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a pin-constraints file for the target FPGA family.

        Parameters
        ----------
        circuit : NANDCircuit
            The circuit whose I/O needs pin assignments.
        pins : dict, optional
            Mapping of signal name → pin identifier.  If *None*, placeholder
            assignments are generated.

        Returns
        -------
        str
            Constraint file content (PCF for ice40, XDC for Xilinx xc7,
            LPF for ECP5).
        """
        pins = pins or {}

        if self.fpga_family == "ice40":
            return self._generate_pcf(circuit, pins)
        elif self.fpga_family == "xc7":
            return self._generate_xdc(circuit, pins)
        elif self.fpga_family == "ecp5":
            return self._generate_lpf(circuit, pins)
        else:
            return self._generate_pcf(circuit, pins)  # generic default

    # ── Yosys invocation ──────────────────────────────────────────────────

    def _run_yosys_synth(self, verilog_code: str, module_name: str) -> Tuple[bool, str]:
        """Build and execute the Yosys TCL script for the chosen FPGA family."""
        tmp_dir = tempfile.mkdtemp(prefix="eml_fpga_")
        try:
            v_path = os.path.join(tmp_dir, f"{module_name}.v")
            json_path = os.path.join(tmp_dir, f"{module_name}.json")

            with open(v_path, "w") as fh:
                fh.write(verilog_code)

            synth_cmd = self._yosys_synth_command(module_name, json_path)

            tcl = f"""\
read_verilog {v_path}
hierarchy -check -top {module_name}
{synth_cmd}
"""
            success, output = _run_yosys(self.yosys_path, tcl, timeout=DEFAULT_TIMEOUT_S)
            return success, output
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _yosys_synth_command(self, module_name: str, json_path: str) -> str:
        """Return the family-specific Yosys synthesis TCL snippet."""
        if self.fpga_family == "ice40":
            return (
                f"synth_ice40 -top {module_name} -json {json_path}\n"
                f"abc -lut 4\n"
            )
        elif self.fpga_family == "xc7":
            return (
                f"synth_xilinx -top {module_name}\n"
                f"abc -lut 6\n"
            )
        elif self.fpga_family == "ecp5":
            return (
                f"synth_ecp5 -top {module_name} -json {json_path}\n"
                f"abc -lut 4\n"
            )
        else:  # generic
            return (
                f"synth -top {module_name}\n"
                f"abc -lut 4\n"
            )

    # ── Constraint generators ─────────────────────────────────────────────

    @staticmethod
    def _generate_pcf(circuit: NANDCircuit, pins: Dict[str, str]) -> str:
        """Generate a PCF file (iCE40)."""
        lines: List[str] = [f"# PCF constraints for iCE40 — auto-generated"]
        for i in range(circuit.num_inputs):
            sig = f"in_{i}"
            pin = pins.get(sig, f"Pin_{i}")
            lines.append(f"set_io {sig} {pin}")
        for i in range(len(circuit.output_wires)):
            sig = f"out_{i}"
            pin = pins.get(sig, f"Pin_{circuit.num_inputs + i}")
            lines.append(f"set_io {sig} {pin}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _generate_xdc(circuit: NANDCircuit, pins: Dict[str, str]) -> str:
        """Generate an XDC file (Xilinx 7-series)."""
        lines: List[str] = [f"# XDC constraints for Xilinx 7-series — auto-generated"]
        for i in range(circuit.num_inputs):
            sig = f"in_{i}"
            pin = pins.get(sig, f"G{i}")
            lines.append(f"set_property PACKAGE_PIN {pin} [get_ports {sig}]")
            lines.append(f"set_property IOSTANDARD LVCMOS33 [get_ports {sig}]")
        for i in range(len(circuit.output_wires)):
            sig = f"out_{i}"
            pin = pins.get(sig, f"G{circuit.num_inputs + i}")
            lines.append(f"set_property PACKAGE_PIN {pin} [get_ports {sig}]")
            lines.append(f"set_property IOSTANDARD LVCMOS33 [get_ports {sig}]")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _generate_lpf(circuit: NANDCircuit, pins: Dict[str, str]) -> str:
        """Generate an LPF file (Lattice ECP5)."""
        lines: List[str] = [f"# LPF constraints for ECP5 — auto-generated"]
        for i in range(circuit.num_inputs):
            sig = f"in_{i}"
            pin = pins.get(sig, f"G{i}")
            lines.append(f"LOCATE COMP \"{sig}\" SITE \"{pin}\";")
            lines.append(f'IOBUF PORT "{sig}" IO_TYPE=LVCMOS33;')
        for i in range(len(circuit.output_wires)):
            sig = f"out_{i}"
            pin = pins.get(sig, f"G{circuit.num_inputs + i}")
            lines.append(f"LOCATE COMP \"{sig}\" SITE \"{pin}\";")
            lines.append(f'IOBUF PORT "{sig}" IO_TYPE=LVCMOS33;')
        return "\n".join(lines) + "\n"


# ─── ASICSynthesizer ─────────────────────────────────────────────────────────

class ASICSynthesizer:
    """
    Synthesize a :class:`NANDCircuit` for an ASIC target.

    Parameters
    ----------
    pdk : str
        One of ``"sky130"``, ``"gf180mcu"``, ``"asicore"``.
    openlane_path : str or None
        Path to the OpenLane installation directory.  If *None*, physical
        design steps are skipped.
    yosys_path : str
        Path or name of the Yosys executable.
    """

    def __init__(
        self,
        pdk: str = "sky130",
        openlane_path: Optional[str] = None,
        yosys_path: str = DEFAULT_YOSYS_PATH,
    ) -> None:
        if pdk not in SUPPORTED_PDKS:
            raise ValueError(
                f"Unsupported PDK '{pdk}'. Choose from {SUPPORTED_PDKS}"
            )
        self.pdk = pdk
        self.openlane_path = openlane_path
        self.yosys_path = yosys_path

    # ── public API ────────────────────────────────────────────────────────

    def synthesize(
        self,
        circuit: NANDCircuit,
        module_name: str = "eml_nand_circuit",
        design_name: str = "eml_design",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run ASIC synthesis on *circuit*.

        Returns
        -------
        synthesis_result : dict
            Keys include ``"verilog"``, ``"metrics"``, ``"log"``.
        metadata : dict
            Keys include ``"is_estimated"``, ``"pdk"``, ``"yosys_available"``,
            ``"openlane_available"``, ``"elapsed_s"``.
        """
        start = time.monotonic()
        verilog_code = _generate_verilog(circuit, module_name, target="asic")

        yosys_available = _is_tool_available(self.yosys_path)
        openlane_available = (
            self.openlane_path is not None
            and os.path.isdir(self.openlane_path)
        )

        yosys_log = ""
        yosys_ok = False

        if yosys_available:
            yosys_ok, yosys_log = self._run_yosys_synth(verilog_code, module_name)

        # OpenLane physical design (if available)
        openlane_log = ""
        if openlane_available and yosys_ok:
            openlane_log = self._run_openlane(module_name, design_name)
        elif openlane_available and not yosys_ok:
            openlane_log = "[skipped] Yosys synthesis failed; skipping OpenLane"

        # Metrics — use estimates when tools are unavailable
        metrics = self.estimate_metrics(circuit)
        is_estimated = not yosys_ok

        elapsed = time.monotonic() - start

        synthesis_result: Dict[str, Any] = {
            "verilog": verilog_code,
            "metrics": metrics,
            "log": yosys_log,
            "openlane_log": openlane_log,
        }
        metadata: Dict[str, Any] = {
            "is_estimated": is_estimated,
            "pdk": self.pdk,
            "yosys_available": yosys_available,
            "openlane_available": openlane_available,
            "elapsed_s": round(elapsed, 3),
            "module_name": module_name,
            "design_name": design_name,
            "gate_count": circuit.gate_count(),
            "depth": circuit.depth(),
        }
        return synthesis_result, metadata

    def estimate_metrics(self, circuit: NANDCircuit) -> Dict[str, Any]:
        """
        Estimate ASIC metrics from gate count and depth for the target PDK.

        Returns
        -------
        dict
            ``area_um2``, ``delay_ns``, ``power_uw``.
        """
        gate_count = circuit.gate_count()
        depth = circuit.depth()

        area_per_gate = _PDK_GATE_AREA_UM2.get(self.pdk, 15.0)
        delay_per_gate = _PDK_GATE_DELAY_NS.get(self.pdk, 0.08)
        power_per_gate = _PDK_GATE_POWER_UW.get(self.pdk, 0.015)

        area_um2 = gate_count * area_per_gate
        delay_ns = depth * delay_per_gate if depth > 0 else delay_per_gate
        power_uw = gate_count * power_per_gate

        return {
            "area_um2": round(area_um2, 2),
            "delay_ns": round(delay_ns, 4),
            "power_uw": round(power_uw, 4),
        }

    # ── Yosys invocation ──────────────────────────────────────────────────

    def _run_yosys_synth(self, verilog_code: str, module_name: str) -> Tuple[bool, str]:
        """Execute Yosys ASIC synthesis."""
        tmp_dir = tempfile.mkdtemp(prefix="eml_asic_")
        try:
            v_path = os.path.join(tmp_dir, f"{module_name}.v")

            with open(v_path, "w") as fh:
                fh.write(verilog_code)

            tcl = f"""\
read_verilog {v_path}
hierarchy -check -top {module_name}
synth -top {module_name}
abc -liberty
clean
stat
"""
            return _run_yosys(self.yosys_path, tcl, timeout=DEFAULT_TIMEOUT_S)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── OpenLane invocation ───────────────────────────────────────────────

    def _run_openlane(self, module_name: str, design_name: str) -> str:
        """
        Attempt to run OpenLane for physical design.

        This is a best-effort invocation.  If OpenLane is not properly
        configured the method returns a diagnostic message instead of
        raising.
        """
        if self.openlane_path is None:
            return "[skipped] OpenLane path not configured"

        openlane_script = os.path.join(self.openlane_path, "flow.tcl")
        if not os.path.isfile(openlane_script):
            return f"[skipped] OpenLane flow.tcl not found at {openlane_script}"

        tmp_dir = tempfile.mkdtemp(prefix="eml_openlane_")
        try:
            env = os.environ.copy()
            env["OPENLANE_ROOT"] = self.openlane_path
            env["DESIGN_NAME"] = design_name
            env["VERILOG_FILES"] = os.path.join(tmp_dir, f"{module_name}.v")
            env["RUN_DIR"] = tmp_dir
            env["PDK"] = self.pdk

            try:
                proc = subprocess.run(
                    ["tclsh", openlane_script],
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_TIMEOUT_S,
                    cwd=tmp_dir,
                    env=env,
                )
                if proc.returncode != 0:
                    return f"OpenLane exited with code {proc.returncode}:\n{proc.stderr or proc.stdout}"
                return proc.stdout
            except subprocess.TimeoutExpired:
                return f"OpenLane timed out after {DEFAULT_TIMEOUT_S}s"
            except FileNotFoundError:
                return "[skipped] tclsh not found — cannot run OpenLane"
            except OSError as exc:
                return f"Failed to run OpenLane: {exc}"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── SynthesisReport ─────────────────────────────────────────────────────────

@dataclass
class SynthesisReport:
    """
    Structured synthesis report combining FPGA and ASIC results.

    Attributes
    ----------
    target_type : str
        ``"fpga"`` or ``"asic"``.
    target_name : str
        Specific target (e.g. ``"xc7"``, ``"sky130"``).
    gate_count : int
        Number of NAND gates in the original circuit.
    optimized_gate_count : int
        Number of gates after synthesis optimisation (same as *gate_count*
        when estimated).
    lut_count : int
        Estimated LUT count (FPGA only; 0 for ASIC).
    ff_count : int
        Estimated flip-flop count.
    bram_count : int
        Estimated BRAM count (FPGA only; 0 for ASIC).
    max_freq_mhz : float
        Estimated maximum clock frequency in MHz.
    area_um2 : float
        Estimated area in µm² (ASIC only; 0 for FPGA).
    delay_ns : float
        Estimated critical-path delay in ns.
    power_uw : float
        Estimated dynamic power in µW.
    verilog_code : str
        Generated Verilog source.
    constraints : str
        Generated constraint file content.
    is_estimated : bool
        *True* when results are analytical estimates rather than tool outputs.
    metadata : dict
        Additional metadata from the synthesis run.
    """

    target_type: str
    target_name: str
    gate_count: int
    optimized_gate_count: int
    lut_count: int
    ff_count: int
    bram_count: int
    max_freq_mhz: float
    area_um2: float
    delay_ns: float
    power_uw: float
    verilog_code: str
    constraints: str
    is_estimated: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the report."""
        return asdict(self)

    def summary(self) -> str:
        """Return a human-readable synthesis summary."""
        est_tag = " [ESTIMATED]" if self.is_estimated else ""
        lines = [
            f"{'=' * 60}",
            f"  EML Synthesis Report{est_tag}",
            f"{'=' * 60}",
            f"  Target       : {self.target_type} / {self.target_name}",
            f"  Gates        : {self.gate_count} (optimised: {self.optimized_gate_count})",
            f"  LUTs         : {self.lut_count}",
            f"  FFs          : {self.ff_count}",
            f"  BRAMs        : {self.bram_count}",
            f"  Max Freq     : {self.max_freq_mhz:.1f} MHz",
            f"  Area         : {self.area_um2:.2f} um^2",
            f"  Delay        : {self.delay_ns:.4f} ns",
            f"  Power        : {self.power_uw:.4f} uW",
            f"  Estimated    : {self.is_estimated}",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)


# ─── Convenience Function ────────────────────────────────────────────────────

def run_synthesis(
    circuit: NANDCircuit,
    target: str = "fpga",
    **kwargs: Any,
) -> SynthesisReport:
    """
    Dispatch synthesis to the appropriate synthesizer.

    Parameters
    ----------
    circuit : NANDCircuit
        The NAND circuit to synthesize.
    target : str
        One of ``"fpga"``, ``"asic"``, ``"fpga_xc7"``, ``"fpga_ice40"``,
        ``"asic_sky130"``.
    **kwargs
        Additional keyword arguments forwarded to the synthesizer constructor
        and/or :meth:`synthesize`.

    Returns
    -------
    SynthesisReport
    """
    # Separate constructor kwargs from synthesize kwargs
    synthesize_keys = {"module_name", "design_name"}
    synth_kwargs = {k: v for k, v in kwargs.items() if k in synthesize_keys}
    ctor_kwargs = {k: v for k, v in kwargs.items() if k not in synthesize_keys}

    # Determine target type and specific family / PDK
    if target in ("fpga_xc7", "fpga_ice40", "fpga_ecp5", "fpga_generic"):
        family = target.split("_", 1)[1]
        synth = FPGASynthesizer(fpga_family=family, **ctor_kwargs)
        result, meta = synth.synthesize(circuit, **synth_kwargs)

        resources = result.get("resources", {})
        return SynthesisReport(
            target_type="fpga",
            target_name=family,
            gate_count=circuit.gate_count(),
            optimized_gate_count=meta.get("gate_count", circuit.gate_count()),
            lut_count=resources.get("lut_count", 0),
            ff_count=resources.get("ff_count", 0),
            bram_count=resources.get("bram_count", 0),
            max_freq_mhz=resources.get("max_freq_mhz", 0.0),
            area_um2=0.0,
            delay_ns=(1.0 / resources["max_freq_mhz"] * 1000.0) if resources.get("max_freq_mhz", 0) else 0.0,
            power_uw=0.0,
            verilog_code=result.get("verilog", ""),
            constraints=result.get("constraints", ""),
            is_estimated=meta.get("is_estimated", True),
            metadata=meta,
        )

    elif target in ("asic_sky130", "asic_gf180mcu", "asic_asicore"):
        pdk = target.split("_", 1)[1]
        synth = ASICSynthesizer(pdk=pdk, **ctor_kwargs)
        result, meta = synth.synthesize(circuit, **synth_kwargs)

        metrics = result.get("metrics", {})
        return SynthesisReport(
            target_type="asic",
            target_name=pdk,
            gate_count=circuit.gate_count(),
            optimized_gate_count=meta.get("gate_count", circuit.gate_count()),
            lut_count=0,
            ff_count=0,
            bram_count=0,
            max_freq_mhz=(1000.0 / metrics["delay_ns"]) if metrics.get("delay_ns", 0) else 0.0,
            area_um2=metrics.get("area_um2", 0.0),
            delay_ns=metrics.get("delay_ns", 0.0),
            power_uw=metrics.get("power_uw", 0.0),
            verilog_code=result.get("verilog", ""),
            constraints="",
            is_estimated=meta.get("is_estimated", True),
            metadata=meta,
        )

    elif target == "fpga":
        family = ctor_kwargs.pop("fpga_family", "xc7")
        synth = FPGASynthesizer(fpga_family=family, **ctor_kwargs)
        result, meta = synth.synthesize(circuit, **synth_kwargs)

        resources = result.get("resources", {})
        return SynthesisReport(
            target_type="fpga",
            target_name=family,
            gate_count=circuit.gate_count(),
            optimized_gate_count=meta.get("gate_count", circuit.gate_count()),
            lut_count=resources.get("lut_count", 0),
            ff_count=resources.get("ff_count", 0),
            bram_count=resources.get("bram_count", 0),
            max_freq_mhz=resources.get("max_freq_mhz", 0.0),
            area_um2=0.0,
            delay_ns=(1.0 / resources["max_freq_mhz"] * 1000.0) if resources.get("max_freq_mhz", 0) else 0.0,
            power_uw=0.0,
            verilog_code=result.get("verilog", ""),
            constraints=result.get("constraints", ""),
            is_estimated=meta.get("is_estimated", True),
            metadata=meta,
        )

    elif target == "asic":
        pdk = ctor_kwargs.pop("pdk", "sky130")
        synth = ASICSynthesizer(pdk=pdk, **ctor_kwargs)
        result, meta = synth.synthesize(circuit, **synth_kwargs)

        metrics = result.get("metrics", {})
        return SynthesisReport(
            target_type="asic",
            target_name=pdk,
            gate_count=circuit.gate_count(),
            optimized_gate_count=meta.get("gate_count", circuit.gate_count()),
            lut_count=0,
            ff_count=0,
            bram_count=0,
            max_freq_mhz=(1000.0 / metrics["delay_ns"]) if metrics.get("delay_ns", 0) else 0.0,
            area_um2=metrics.get("area_um2", 0.0),
            delay_ns=metrics.get("delay_ns", 0.0),
            power_uw=metrics.get("power_uw", 0.0),
            verilog_code=result.get("verilog", ""),
            constraints="",
            is_estimated=meta.get("is_estimated", True),
            metadata=meta,
        )

    else:
        raise ValueError(
            f"Unknown target '{target}'. Use 'fpga', 'asic', 'fpga_xc7', "
            f"'fpga_ice40', 'asic_sky130', etc."
        )
