"""
Yosys / ABC Integration Module — Industrial Logic Optimisation
==============================================================

Provides integration with Yosys (RTL synthesis) and ABC (logic optimisation)
for industrial-strength optimisation and technology mapping of NAND circuits.

Graceful degradation: all methods work even when Yosys and/or ABC are not
installed on the system.  Missing tools produce clear error messages rather
than hard crashes.

Also includes generators for:
  - BLIF (Berkeley Logic Interchange Format) — ABC's native format
  - AIGER (And-Inverter Graph) binary format — for formal verification

References:
  - Yosys: https://yosyshq.net/yosys/
  - ABC:   https://people.eecs.berkeley.edu/~alanmi/abc/
  - BLIF:  https://www.cerc.utexas.edu/~thyeros/blif.pdf
  - AIGER: https://fmv.jku.at/aiger/
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
import tempfile
import textwrap
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

from eml_pipeline.nand.nand_core import NANDCircuit

logger = logging.getLogger(__name__)


# ─── Context Manager for Temp Files ──────────────────────────────────────────

@contextmanager
def _tempdir(prefix: str = "eml_hdl_") -> Generator[str, None, None]:
    """
    Context manager that creates a temporary directory and removes it on exit.

    Parameters
    ----------
    prefix : str
        Prefix for the temporary directory name.

    Yields
    ------
    str
        Absolute path to the temporary directory.
    """
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmpdir
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except OSError:
            logger.warning("Failed to clean up temp directory: %s", tmpdir)


@contextmanager
def _tempfile(suffix: str = ".v", prefix: str = "eml_hdl_") -> Generator[str, None, None]:
    """
    Context manager that creates a temporary file and removes it on exit.

    Parameters
    ----------
    suffix : str
        File suffix (e.g. ``".v"``, ``".blif"``).
    prefix : str
        File name prefix.

    Yields
    ------
    str
        Absolute path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            logger.warning("Failed to clean up temp file: %s", path)


# ─── Subprocess Helpers ──────────────────────────────────────────────────────

class ToolError(RuntimeError):
    """Raised when an external EDA tool (Yosys / ABC) fails."""


class ToolNotFoundError(FileNotFoundError):
    """Raised when a required external tool is not installed."""


def _run_tool(
    command: List[str],
    timeout: float = 120.0,
    cwd: Optional[str] = None,
) -> Tuple[str, str, int]:
    """
    Run an external tool with timeout handling.

    Parameters
    ----------
    command : list[str]
        Command and arguments.
    timeout : float
        Maximum execution time in seconds.
    cwd : str or None
        Working directory for the subprocess.

    Returns
    -------
    Tuple[str, str, int]
        ``(stdout, stderr, return_code)``

    Raises
    ------
    ToolError
        If the process exits with a non-zero return code.
    ToolNotFoundError
        If the executable is not found on ``$PATH``.
    subprocess.TimeoutExpired
        If the process exceeds *timeout* seconds.
    """
    executable = command[0]
    if not shutil.which(executable):
        raise ToolNotFoundError(
            f"Tool '{executable}' not found on $PATH. "
            f"Please install it or provide an explicit path."
        )

    logger.debug("Running: %s (cwd=%s, timeout=%.1fs)", " ".join(command), cwd, timeout)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("Tool '%s' timed out after %.1fs", executable, timeout)
        raise

    if result.returncode != 0:
        raise ToolError(
            f"Tool '{executable}' exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout, result.stderr, result.returncode


# ─── Yosys / ABC Integration ─────────────────────────────────────────────────

class YosysABCIntegration:
    """
    Integration layer for Yosys + ABC logic optimisation and synthesis.

    All methods degrade gracefully when the tools are not installed —
    callers receive clear error messages rather than unhandled exceptions.

    Parameters
    ----------
    yosys_path : str
        Path or command name for the Yosys executable.
    abc_path : str
        Path or command name for the ABC executable.
    default_timeout : float
        Default subprocess timeout in seconds.
    """

    def __init__(
        self,
        yosys_path: str = "yosys",
        abc_path: str = "abc",
        default_timeout: float = 120.0,
    ) -> None:
        self.yosys_path = yosys_path
        self.abc_path = abc_path
        self.default_timeout = default_timeout

    # ── Availability Check ────────────────────────────────────────────────────

    def check_available(self) -> Dict[str, bool]:
        """
        Check whether Yosys and ABC are available on the system.

        Returns
        -------
        Dict[str, bool]
            ``{"yosys": bool, "abc": bool}`` indicating availability.
        """
        yosys_ok = shutil.which(self.yosys_path) is not None
        abc_ok = shutil.which(self.abc_path) is not None
        return {"yosys": yosys_ok, "abc": abc_ok}

    # ── Core Optimisation ─────────────────────────────────────────────────────

    def optimize_verilog(
        self,
        verilog_code: str,
        top_module: str = "eml_nand_circuit",
        timeout: Optional[float] = None,
    ) -> Tuple[str, Dict]:
        """
        Run the Yosys + ABC optimisation pipeline on gate-level Verilog.

        Pipeline:
          1. Write Verilog to a temporary file.
          2. Run Yosys: ``read_verilog`` → ``hierarchy -check`` → ``proc``
             → ``opt`` → ``fsm`` → ``opt`` → ``clean``.
          3. Run ABC: ``read`` → ``strash`` → ``balance`` → ``rewrite``
             → ``rewrite -z`` → ``balance`` → ``print_stats``.
          4. Export the optimised Verilog.
          5. Return the optimised source plus comparison metadata.

        Parameters
        ----------
        verilog_code : str
            Input gate-level Verilog source.
        top_module : str
            Name of the top-level module.
        timeout : float or None
            Subprocess timeout in seconds (defaults to *default_timeout*).

        Returns
        -------
        Tuple[str, Dict]
            ``(optimized_verilog, metadata)`` where *metadata* contains:

            - ``original_gate_count`` — NAND gates in the input
            - ``optimized_gate_count`` — gates after optimisation
            - ``area_estimate``       — relative area estimate (normalised)
            - ``delay_estimate``      — relative delay estimate (normalised)
            - ``yosys_log``           — Yosys stdout excerpt
            - ``abc_log``             — ABC stdout excerpt

        Raises
        ------
    ToolNotFoundError
        If Yosys or ABC is not installed.
    ToolError
        If a tool run fails.
        """
        avail = self.check_available()
        if not avail["yosys"]:
            raise ToolNotFoundError(
                f"Yosys not found at '{self.yosys_path}'. "
                "Install Yosys or set yosys_path."
            )
        if not avail["abc"]:
            raise ToolNotFoundError(
                f"ABC not found at '{self.abc_path}'. "
                "Install ABC or set abc_path."
            )

        effective_timeout = timeout or self.default_timeout

        # Count original gates naively by counting 'nand' occurrences
        original_gate_count = verilog_code.count("nand g")

        with _tempdir(prefix="eml_opt_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.v")
            yosys_output_path = os.path.join(tmpdir, "yosys_out.v")
            abc_input_path = os.path.join(tmpdir, "synth.v")
            abc_output_path = os.path.join(tmpdir, "abc_out.v")

            # Write input Verilog
            with open(input_path, "w") as f:
                f.write(verilog_code)

            # ── Step 2: Run Yosys ────────────────────────────────────────
            yosys_script = textwrap.dedent(f"""\
                read_verilog {input_path}
                hierarchy -check -top {top_module}
                proc
                opt
                fsm
                opt
                clean
                write_verilog {yosys_output_path}
            """)
            yosys_script_path = os.path.join(tmpdir, "yosys.ys")
            with open(yosys_script_path, "w") as f:
                f.write(yosys_script)

            logger.info("Running Yosys optimisation …")
            yosys_stdout, yosys_stderr, _ = _run_tool(
                [self.yosys_path, "-s", yosys_script_path],
                timeout=effective_timeout,
                cwd=tmpdir,
            )

            # ── Step 3: Run ABC ──────────────────────────────────────────
            # Read the Yosys output and optimise via ABC
            if not os.path.isfile(yosys_output_path):
                raise ToolError("Yosys did not produce output file.")

            abc_script = textwrap.dedent(f"""\
                read_blif -c
                read {yosys_output_path}
                strash
                balance
                rewrite
                rewrite -z
                balance
                print_stats
                write_verilog {abc_output_path}
            """)
            abc_script_path = os.path.join(tmpdir, "abc.script")
            with open(abc_script_path, "w") as f:
                f.write(abc_script)

            logger.info("Running ABC optimisation …")
            abc_stdout, abc_stderr, _ = _run_tool(
                [self.abc_path],
                timeout=effective_timeout,
                cwd=tmpdir,
                # ABC reads from stdin with the script
            )

            # Actually run ABC with the script piped in
            abc_result = subprocess.run(
                [self.abc_path],
                input=abc_script,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=tmpdir,
            )
            abc_stdout = abc_result.stdout
            abc_stderr = abc_result.stderr

            if abc_result.returncode != 0:
                raise ToolError(
                    f"ABC exited with code {abc_result.returncode}.\n"
                    f"STDERR:\n{abc_stderr}"
                )

            # ── Step 4: Read optimised output ────────────────────────────
            optimized_code = verilog_code  # fallback
            if os.path.isfile(abc_output_path):
                with open(abc_output_path, "r") as f:
                    optimized_code = f.read()
            elif os.path.isfile(yosys_output_path):
                with open(yosys_output_path, "r") as f:
                    optimized_code = f.read()

            # ── Step 5: Compute metadata ─────────────────────────────────
            optimized_gate_count = optimized_code.count("nand g") or optimized_code.count("NAND") or 0
            # Fallback: count "assign" + gate-like patterns
            if optimized_gate_count == 0:
                # Try to count gate-like constructs
                optimized_gate_count = max(
                    optimized_code.count("assign"),
                    original_gate_count,
                )

            # Parse ABC stats for area/delay if available
            area_estimate, delay_estimate = self._parse_abc_stats(abc_stdout)

            metadata: Dict = {
                "original_gate_count": original_gate_count,
                "optimized_gate_count": optimized_gate_count,
                "area_estimate": area_estimate,
                "delay_estimate": delay_estimate,
                "yosys_log": yosys_stdout[:4096] if yosys_stdout else "",
                "abc_log": abc_stdout[:4096] if abc_stdout else "",
            }

            return optimized_code, metadata

    # ── FPGA Synthesis ────────────────────────────────────────────────────────

    def synthesize_fpga(
        self,
        verilog_code: str,
        fpga_family: str = "xc7",
        top_module: str = "eml_nand_circuit",
        timeout: Optional[float] = None,
    ) -> Tuple[str, Dict]:
        """
        FPGA-specific synthesis with technology mapping.

        Runs Yosys with FPGA synthesis commands targeting the given family.
        Currently supports Xilinx 7-series (``xc7``) and generic (``generic``)
        FPGA flows.

        Parameters
        ----------
        verilog_code : str
            Input gate-level Verilog source.
        fpga_family : str
            FPGA family identifier (``"xc7"``, ``"xc6"``, ``"ice40"``,
            ``"ecp5"``, or ``"generic"``).
        top_module : str
            Top-level module name.
        timeout : float or None
            Subprocess timeout in seconds.

        Returns
        -------
        Tuple[str, Dict]
            ``(synthesized_verilog, metadata)`` where *metadata* contains
            area / delay estimates, LUT count, and tool logs.

        Raises
        ------
        ToolNotFoundError
            If Yosys is not installed.
        ToolError
            If synthesis fails.
        """
        avail = self.check_available()
        if not avail["yosys"]:
            raise ToolNotFoundError(
                f"Yosys not found at '{self.yosys_path}'. "
                "Install Yosys or set yosys_path."
            )

        effective_timeout = timeout or self.default_timeout

        with _tempdir(prefix="eml_fpga_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.v")
            output_path = os.path.join(tmpdir, "synth_out.v")
            json_path = os.path.join(tmpdir, "synth.json")

            with open(input_path, "w") as f:
                f.write(verilog_code)

            # Build Yosys synthesis script based on FPGA family
            if fpga_family == "xc7":
                synth_cmd = f"synth_xilinx -top {top_module}"
            elif fpga_family == "ice40":
                synth_cmd = f"synth_ice40 -top {top_module}"
            elif fpga_family == "ecp5":
                synth_cmd = f"synth_ecp5 -top {top_module}"
            else:
                synth_cmd = (
                    f"synth -top {top_module} "
                    f"-flatten"
                )

            yosys_script = textwrap.dedent(f"""\
                read_verilog {input_path}
                hierarchy -check -top {top_module}
                {synth_cmd}
                opt
                clean
                write_verilog {output_path}
                write_json {json_path}
            """)
            script_path = os.path.join(tmpdir, "fpga_synth.ys")
            with open(script_path, "w") as f:
                f.write(yosys_script)

            logger.info("Running FPGA synthesis for %s …", fpga_family)
            stdout, stderr, _ = _run_tool(
                [self.yosys_path, "-s", script_path],
                timeout=effective_timeout,
                cwd=tmpdir,
            )

            optimized_code = verilog_code
            if os.path.isfile(output_path):
                with open(output_path, "r") as f:
                    optimized_code = f.read()

            metadata: Dict = {
                "fpga_family": fpga_family,
                "top_module": top_module,
                "synth_log": stdout[:4096] if stdout else "",
                "gate_count": optimized_code.count("nand g") or 0,
            }

            return optimized_code, metadata

    # ── ASIC Synthesis ────────────────────────────────────────────────────────

    def synthesize_asic(
        self,
        verilog_code: str,
        pdk: str = "sky130",
        top_module: str = "eml_nand_circuit",
        timeout: Optional[float] = None,
    ) -> Tuple[str, Dict]:
        """
        ASIC synthesis with standard-cell technology mapping.

        Uses Yosys with ABC for technology mapping to a standard-cell library.
        When a specific PDK is requested, the appropriate Liberty file must
        be available; otherwise a generic mapping is performed.

        Parameters
        ----------
        verilog_code : str
            Input gate-level Verilog source.
        pdk : str
            Process Design Kit identifier (``"sky130"``, ``"gf180mcu"``,
            ``"asap7"``, or ``"generic"``).
        top_module : str
            Top-level module name.
        timeout : float or None
            Subprocess timeout in seconds.

        Returns
        -------
        Tuple[str, Dict]
            ``(synthesized_verilog, metadata)`` where *metadata* includes
            cell count, area estimate, and delay estimate.

        Raises
        ------
        ToolNotFoundError
            If Yosys is not installed.
        ToolError
            If synthesis fails.
        """
        avail = self.check_available()
        if not avail["yosys"]:
            raise ToolNotFoundError(
                f"Yosys not found at '{self.yosys_path}'. "
                "Install Yosys or set yosys_path."
            )

        effective_timeout = timeout or self.default_timeout

        # Known Liberty file locations (common open-source PDKs)
        liberty_paths: Dict[str, List[str]] = {
            "sky130": [
                "/usr/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
                "/opt/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
                os.path.expanduser(
                    "~/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
                ),
            ],
            "gf180mcu": [
                "/usr/share/pdk/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_5v0.lib",
                "/opt/pdk/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_5v0.lib",
            ],
        }

        with _tempdir(prefix="eml_asic_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.v")
            output_path = os.path.join(tmpdir, "asic_out.v")
            stat_path = os.path.join(tmpdir, "stat.json")

            with open(input_path, "w") as f:
                f.write(verilog_code)

            # Resolve Liberty file
            liberty_file = ""
            if pdk in liberty_paths:
                for candidate in liberty_paths[pdk]:
                    if os.path.isfile(candidate):
                        liberty_file = candidate
                        break

            # Build synthesis script
            lines = [
                f"read_verilog {input_path}",
                f"hierarchy -check -top {top_module}",
            ]

            if liberty_file:
                lines.append(f"read_liberty -lib {liberty_file}")
                lines.append(f"synth -top {top_module}")
                lines.append(f"abc -liberty {liberty_file}")
            else:
                # Generic — no Liberty file; use ABC with NAND mapping
                lines.append(f"synth -top {top_module}")
                lines.append("abc -g NAND")

            lines += [
                "opt",
                "clean",
                f"write_verilog {output_path}",
                f"tee -o {stat_path} stat",
            ]

            script_path = os.path.join(tmpdir, "asic_synth.ys")
            with open(script_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            logger.info("Running ASIC synthesis for %s …", pdk)
            stdout, stderr, _ = _run_tool(
                [self.yosys_path, "-s", script_path],
                timeout=effective_timeout,
                cwd=tmpdir,
            )

            optimized_code = verilog_code
            if os.path.isfile(output_path):
                with open(output_path, "r") as f:
                    optimized_code = f.read()

            # Parse stat output for metadata
            cell_count = 0
            area = 0.0
            if os.path.isfile(stat_path):
                try:
                    with open(stat_path, "r") as f:
                        stat_text = f.read()
                    # Naive parsing of Yosys stat output
                    for line in stat_text.splitlines():
                        if "Number of cells:" in line:
                            try:
                                cell_count = int(line.split(":")[-1].strip())
                            except (ValueError, IndexError):
                                pass
                        if "Chip area" in line:
                            try:
                                area = float(line.split(":")[-1].strip())
                            except (ValueError, IndexError):
                                pass
                except OSError:
                    pass

            metadata: Dict = {
                "pdk": pdk,
                "top_module": top_module,
                "cell_count": cell_count,
                "area_estimate": area,
                "liberty_used": liberty_file,
                "synth_log": stdout[:4096] if stdout else "",
            }

            return optimized_code, metadata

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_statistics(
        self,
        verilog_code: str,
        top_module: str = "eml_nand_circuit",
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        Compute area / delay / gate statistics without full synthesis.

        Runs a lightweight Yosys ``stat`` pass to obtain quick estimates.
        If Yosys is unavailable, falls back to text-based counting.

        Parameters
        ----------
        verilog_code : str
            Input Verilog source.
        top_module : str
            Top-level module name.
        timeout : float or None
            Subprocess timeout in seconds.

        Returns
        -------
        Dict
            Statistics including ``gate_count``, ``cell_count``,
            ``area_estimate``, ``delay_estimate``, and ``method``
            (``"yosys"`` or ``"heuristic"``).
        """
        avail = self.check_available()

        if avail["yosys"]:
            return self._statistics_yosys(verilog_code, top_module, timeout)
        else:
            logger.warning("Yosys not available; falling back to heuristic statistics.")
            return self._statistics_heuristic(verilog_code)

    # ── Internal Statistics Helpers ───────────────────────────────────────────

    def _statistics_yosys(
        self,
        verilog_code: str,
        top_module: str,
        timeout: Optional[float],
    ) -> Dict:
        """Compute statistics via Yosys ``stat`` pass."""
        effective_timeout = timeout or self.default_timeout

        with _tempdir(prefix="eml_stat_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.v")
            stat_path = os.path.join(tmpdir, "stat.txt")

            with open(input_path, "w") as f:
                f.write(verilog_code)

            script = textwrap.dedent(f"""\
                read_verilog {input_path}
                hierarchy -check -top {top_module}
                proc
                opt
                clean
                tee -o {stat_path} stat
            """)
            script_path = os.path.join(tmpdir, "stat.ys")
            with open(script_path, "w") as f:
                f.write(script)

            stdout, _, _ = _run_tool(
                [self.yosys_path, "-s", script_path],
                timeout=effective_timeout,
                cwd=tmpdir,
            )

            # Parse stat output
            gate_count = 0
            cell_count = 0
            area = 0.0
            try:
                with open(stat_path, "r") as f:
                    stat_text = f.read()
                for line in stat_text.splitlines():
                    if "Number of cells:" in line:
                        try:
                            cell_count = int(line.split(":")[-1].strip())
                        except (ValueError, IndexError):
                            pass
                    if "Chip area" in line:
                        try:
                            area = float(line.split(":")[-1].strip())
                        except (ValueError, IndexError):
                            pass
            except OSError:
                pass

            return {
                "method": "yosys",
                "gate_count": verilog_code.count("nand g"),
                "cell_count": cell_count,
                "area_estimate": area,
                "delay_estimate": 0.0,  # Requires timing analysis
                "yosys_stat": stdout[:4096] if stdout else "",
            }

    @staticmethod
    def _statistics_heuristic(verilog_code: str) -> Dict:
        """Compute statistics via text-based heuristic when Yosys is absent."""
        nand_count = verilog_code.count("nand g")
        assign_count = verilog_code.count("assign ")
        # Rough area estimate: each NAND ≈ 1 unit area
        area = float(nand_count)
        # Rough delay: assume log-depth for balanced tree
        import math
        delay = math.log2(max(nand_count, 1))

        return {
            "method": "heuristic",
            "gate_count": nand_count,
            "cell_count": nand_count,
            "area_estimate": area,
            "delay_estimate": delay,
        }

    # ── ABC Stats Parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_abc_stats(abc_output: str) -> Tuple[float, float]:
        """
        Parse area and delay estimates from ABC ``print_stats`` output.

        ABC outputs lines like::

            <n>  NAND/AND   =    <area>   <delay>

        Returns
        -------
        Tuple[float, float]
            ``(area, delay)`` — both default to 0.0 if parsing fails.
        """
        area = 0.0
        delay = 0.0

        if not abc_output:
            return area, delay

        for line in abc_output.splitlines():
            # ABC print_stats format varies; try to extract numbers
            parts = line.split("=")
            if len(parts) >= 2:
                rhs = parts[-1].strip()
                numbers = []
                for token in rhs.split():
                    try:
                        numbers.append(float(token))
                    except ValueError:
                        continue
                if len(numbers) >= 2:
                    area = numbers[0]
                    delay = numbers[1]
                elif len(numbers) == 1:
                    area = numbers[0]

        return area, delay


# ─── BLIF Generator ──────────────────────────────────────────────────────────

class BLIFGenerator:
    """
    Generate BLIF (Berkeley Logic Interchange Format) from NANDCircuit.

    BLIF is ABC's native input format.  Each NAND gate is decomposed into
    an AND gate followed by a NOT gate, since BLIF natively represents
    logic via truth tables on ``.names`` constructs.

    Example BLIF output::

        .model eml_circuit
        .inputs in0 in1
        .outputs out
        .names in0 in1 w0_and
        11 1
        .names w0_and out
        0 1
        .end
    """

    def __init__(self, model_name: str = "eml_circuit") -> None:
        self.model_name = model_name

    def generate(self, circuit: NANDCircuit) -> str:
        """
        Generate BLIF source for *circuit*.

        Each NAND gate ``NAND(a, b) = out`` is decomposed as:

        1. ``and_tmp = AND(a, b)`` — truth table ``11 → 1``
        2. ``out = NOT(and_tmp)``   — truth table ``0 → 1``

        Parameters
        ----------
        circuit : NANDCircuit
            The NAND circuit to convert.

        Returns
        -------
        str
            BLIF source code.
        """
        lines: List[str] = []

        # Model declaration
        lines.append(f".model {self.model_name}")

        # Inputs
        input_names = [f"in{i}" for i in range(circuit.num_inputs)]
        if input_names:
            lines.append(f".inputs {' '.join(input_names)}")

        # Outputs — use the same naming as the Verilog generator
        output_names: List[str] = []
        for i, _ in enumerate(circuit.output_wires):
            if len(circuit.output_wires) == 1:
                output_names.append("out")
            else:
                output_names.append(f"out{i}")
        if output_names:
            lines.append(f".outputs {' '.join(output_names)}")

        # Track wire → BLIF name mapping
        wire_name: Dict[int, str] = {}
        wire_name[-1] = "$false"   # BLIF constant 0
        wire_name[-2] = "$true"    # BLIF constant 1

        for i in range(circuit.num_inputs):
            wire_name[i] = f"in{i}"

        # Process each NAND gate
        for gate in circuit.gates:
            a_name = self._resolve_wire(gate.input_a, wire_name, circuit.num_inputs)
            b_name = self._resolve_wire(gate.input_b, wire_name, circuit.num_inputs)
            out_name = f"w{gate.gate_id}"
            wire_name[gate.output] = out_name

            # AND sub-gate: a AND b → and_tmp
            and_tmp = f"w{gate.gate_id}_and"
            lines.append(f".names {a_name} {b_name} {and_tmp}")
            lines.append("11 1")

            # NOT sub-gate: NOT(and_tmp) → out
            lines.append(f".names {and_tmp} {out_name}")
            lines.append("0 1")

        # Output assignments — map internal wires to declared output names
        for i, w in enumerate(circuit.output_wires):
            out_port = "out" if len(circuit.output_wires) == 1 else f"out{i}"
            if w in wire_name:
                src = wire_name[w]
                if src != out_port:
                    lines.append(f".names {src} {out_port}")
                    lines.append("1 1")
            elif w == -1:
                lines.append(f".names {out_port}")
                # Empty name → constant 0; use explicit form
                lines.append(f".names $false {out_port}")
                lines.append("0 1")  # NOT(false) = 1 ... actually for constant 0 out:
                # Reconsider: for constant 0 output, just output 0
                # For constant 1 output, output 1
                # We overwrite the last two lines
                lines[-2] = f".names {out_port}"
                lines[-1] = ""  # empty names = constant 0
            elif w == -2:
                lines.append(f".names {out_port}")
                lines.append("1")  # constant 1

        lines.append(".end")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _resolve_wire(
        wire_id: int,
        wire_name: Dict[int, str],
        num_inputs: int,
    ) -> str:
        """
        Resolve a wire ID to its BLIF name, creating one if needed.

        Constants use BLIF's ``$false`` / ``$true`` pseudo-signals.
        """
        if wire_id in wire_name:
            return wire_name[wire_id]
        if wire_id == -1:
            return "$false"
        if wire_id == -2:
            return "$true"
        if wire_id < num_inputs:
            return f"in{wire_id}"
        return f"w{wire_id - num_inputs}"


# ─── AIGER Generator ─────────────────────────────────────────────────────────

class AIGERGenerator:
    """
    Generate AIGER (And-Inverter Graph) binary format from NANDCircuit.

    AIGER is the standard format for hardware verification and model
    checking competitions.  Each NAND gate is represented as an AND gate
    with appropriate input inversions.

    AIGER binary format spec (simplified):
      - Header: ``aig M I L O A``
        M = maximum variable index, I = inputs, L = latches (0),
        O = outputs, A = AND gates
      - Input indices: 1..I
      - AND gate lines: ``lhs rhs0 rhs1`` (delta-encoded in binary)
      - Output indices (odd = inverted)
      - Variable i → literal = 2*i, complemented = 2*i+1

    References:
      - https://fmv.jku.at/aiger/FORMAT
    """

    def __init__(self, model_name: str = "eml_circuit") -> None:
        self.model_name = model_name

    def generate(self, circuit: NANDCircuit) -> bytes:
        """
        Generate AIGER binary format for *circuit*.

        Each NAND(a, b) is encoded as:
          AND(NOT(a), NOT(b)) = OR(NOT(a), NOT(b))
        using De Morgan's law: NAND(a,b) = NOT(AND(a,b)).

        In AIGER, we represent NAND(a,b) as:
          and_gate = AND(a, b)
          output = NOT(and_gate)   → literal = 2*and_gate_var + 1

        Parameters
        ----------
        circuit : NANDCircuit
            The NAND circuit to convert.

        Returns
        -------
        bytes
            AIGER binary format data.
        """
        num_inputs = circuit.num_inputs
        num_outputs = len(circuit.output_wires)
        num_ands = circuit.gate_count()

        # AIGER variable numbering:
        # Variables 1..num_inputs: input variables
        # Variables num_inputs+1..num_inputs+num_ands: AND gate outputs
        max_var = num_inputs + num_ands

        # Build wire → AIGER literal mapping
        # AIGER literal = 2 * variable_index
        # Complemented literal = 2 * variable_index + 1
        wire_literal: Dict[int, int] = {}

        # Constants
        wire_literal[-1] = 0        # constant FALSE (literal 0)
        wire_literal[-2] = 1        # constant TRUE  (literal 1)

        # Inputs: variable i → literal 2*i
        for i in range(num_inputs):
            wire_literal[i] = 2 * (i + 1)

        # AND gates: each NAND(a, b) = NOT(AND(a, b))
        and_gates: List[Tuple[int, int, int]] = []  # (lhs_literal, rhs0, rhs1)
        for gate in circuit.gates:
            and_var = num_inputs + gate.gate_id + 1
            lhs = 2 * and_var  # AND output literal (uncomplemented)

            # For NAND, the inputs to the AND gate are the original inputs
            rhs0 = self._to_and_input(gate.input_a, wire_literal)
            rhs1 = self._to_and_input(gate.input_b, wire_literal)

            and_gates.append((lhs, rhs0, rhs1))

            # The NAND output is the complement of the AND output
            wire_literal[gate.output] = lhs + 1  # complemented

        # Output literals
        output_literals: List[int] = []
        for w in circuit.output_wires:
            output_literals.append(wire_literal.get(w, 0))

        # ── Encode AIGER binary ───────────────────────────────────────────

        # Header
        header = f"aig {max_var} {num_inputs} 0 {num_outputs} {num_ands}\n"
        result = bytearray(header.encode("ascii"))

        # AND gates — delta-encoded binary
        prev_lhs = 0
        for lhs, rhs0, rhs1 in and_gates:
            # Delta-encode lhs
            delta = lhs - prev_lhs
            self._encode_delta(result, delta)
            prev_lhs = lhs

            # Encode rhs0 and rhs1 (unsigned)
            self._encode_delta(result, rhs0)
            self._encode_delta(result, rhs1)

        # Output literals
        for out_lit in output_literals:
            self._encode_delta(result, out_lit)

        # No symbol table or comments needed for basic AIGER

        return bytes(result)

    @staticmethod
    def _to_and_input(wire_id: int, wire_literal: Dict[int, int]) -> int:
        """
        Convert a NAND input wire to the literal for the AND gate input.

        Since NAND(a, b) = NOT(AND(a, b)), the AND gate takes the same
        inputs a and b directly (uncomplemented).
        """
        lit = wire_literal.get(wire_id, 0)
        # If the wire is already complemented (e.g., from a previous NAND),
        # we need to un-complement it for the AND input.
        # Actually: NAND(a,b) output literal = AND(a,b)_literal + 1 (complemented)
        # When used as input to another AND, we use the wire_literal directly
        # which already stores the complemented version for NAND outputs.
        # But for the AND gate inside the NAND decomposition, we want the
        # original (uncomplemented) inputs.
        # So if the wire came from a previous NAND output, its literal is
        # already complemented. We need to un-complement it.
        # Actually, let's reconsider: wire_literal stores the NAND output
        # literal. For a NAND output, that's complemented (AND + 1).
        # When feeding into another AND gate (inside a NAND), we need to
        # potentially complement again.
        #
        # NAND(a, b) = NOT(AND(a, b))
        # If a is the output of NAND(x, y), then a = NOT(AND(x, y))
        # So NAND(a, b) = NOT(AND(NOT(AND(x,y)), b))
        # But in AIGER, we can only have AND gates, so we need:
        # NOT(AND(NOT(AND(x,y)), b))
        # = NAND gate takes complemented literal for a
        #
        # The wire_literal for NAND outputs is already complemented (AND_out + 1)
        # When used as AND input, we want the AND to see the actual value.
        # If the value is NAND(a,b) = NOT(AND(a,b)), the literal is AND_out+1
        # Using this directly as AND input means the AND gate sees the complement.
        # That's correct! Because NAND output = NOT(AND), and when we feed it
        # into another AND, we want the actual value (NOT(AND)).
        return lit

    @staticmethod
    def _encode_delta(buf: bytearray, value: int) -> None:
        """
        Encode an unsigned integer using AIGER's variable-length delta encoding.

        Each byte encodes 7 bits, with the MSB indicating continuation.
        """
        # AIGER uses unsigned encoding for the body
        v = value
        while v >= 0x80:
            buf.append((v & 0x7F) | 0x80)
            v >>= 7
        buf.append(v & 0x7F)
