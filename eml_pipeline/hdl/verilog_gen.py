"""
Verilog Generator Module — Gate-Level NAND Circuit Export
==========================================================

Generates structural gate-level Verilog from NANDCircuit representations.
Supports multiple synthesis targets (generic, FPGA, ASIC) with appropriate
synthesis directives and timing annotations.

Wire conventions (from eml_pipeline.nand.nand_core):
  -1  = constant 0
  -2  = constant 1
  0..num_inputs-1      = input wires   (named in0, in1, ...)
  num_inputs..         = internal wires (named w0, w1, ...)

References:
  - Epsilon-Mu Lambda (EML) NAND duality pipeline
  - Sheffer stroke completeness (Theorem 3.1)
  - Signal restoration (Theorem 4.2)
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

from eml_pipeline.nand.nand_core import NANDCircuit, NANDGate


# ─── Wire Naming Helpers ─────────────────────────────────────────────────────

def _wire_name(wire_id: int, num_inputs: int) -> str:
    """
    Map a wire ID to a human-readable Verilog signal name.

    Parameters
    ----------
    wire_id : int
        Wire identifier following NANDCircuit conventions.
    num_inputs : int
        Number of primary inputs in the circuit.

    Returns
    -------
    str
        Verilog signal name:
          -1 → 1'b0 (constant zero)
          -2 → 1'b1 (constant one)
          0..num_inputs-1 → in0, in1, ... (primary inputs)
          num_inputs..   → w0, w1, ...    (internal wires)
    """
    if wire_id == -1:
        return "1'b0"
    if wire_id == -2:
        return "1'b1"
    if wire_id < num_inputs:
        return f"in{wire_id}"
    return f"w{wire_id - num_inputs}"


def _is_constant(wire_id: int) -> bool:
    """Return True if the wire ID represents a constant (−1 or −2)."""
    return wire_id < 0


def _is_input(wire_id: int, num_inputs: int) -> bool:
    """Return True if the wire ID represents a primary input."""
    return 0 <= wire_id < num_inputs


def _is_internal(wire_id: int, num_inputs: int) -> bool:
    """Return True if the wire ID represents an internal gate output."""
    return wire_id >= num_inputs


# ─── Timing Estimation ───────────────────────────────────────────────────────

def _estimate_timing(circuit: NANDCircuit) -> Dict[str, float]:
    """
    Estimate propagation delay and timing metrics for the circuit.

    Uses a simple unit-delay model where each NAND gate contributes
    one unit of delay. Real delays depend on the technology node.

    Parameters
    ----------
    circuit : NANDCircuit
        The circuit to analyze.

    Returns
    -------
    Dict[str, float]
        Timing estimates including critical_path, max_delay_ns, and
        per-output delays.
    """
    wire_depth: Dict[int, int] = {-1: 0, -2: 0}
    for i in range(circuit.num_inputs):
        wire_depth[i] = 0

    for gate in circuit.gates:
        d_a = wire_depth.get(gate.input_a, 0)
        d_b = wire_depth.get(gate.input_b, 0)
        wire_depth[gate.output] = max(d_a, d_b) + 1

    # Approximate NAND gate delay in 45nm CMOS ~ 0.02ns
    gate_delay_ns = 0.02
    critical_path = max(
        (wire_depth.get(w, 0) for w in circuit.output_wires),
        default=0,
    )

    output_delays = {
        _wire_name(w, circuit.num_inputs): wire_depth.get(w, 0) * gate_delay_ns
        for w in circuit.output_wires
    }

    return {
        "critical_path_depth": critical_path,
        "max_delay_ns": critical_path * gate_delay_ns,
        "gate_delay_ns": gate_delay_ns,
        "output_delays": output_delays,
    }


# ─── Verilog Generator ───────────────────────────────────────────────────────

class VerilogGenerator:
    """
    Generate gate-level Verilog from NANDCircuit objects.

    Supports three synthesis targets that control what synthesis directives
    are emitted alongside the structural NAND netlist:

    * ``"generic"`` — plain Verilog, no target-specific pragmas.
    * ``"fpga"``    — emits ``(* KEEP *)`` attributes on key wires so they
                       survive FPGA synthesis optimisation.
    * ``"asic"``    — emits ``/* synthesis syn_keep=1 */`` directives for
                       ASIC tool compatibility.

    Parameters
    ----------
    module_name : str
        Name of the Verilog module to generate.
    target : str
        Synthesis target — one of ``"generic"``, ``"fpga"``, ``"asic"``.

    Raises
    ------
    ValueError
        If *target* is not one of the supported values.
    """

    VALID_TARGETS = ("generic", "fpga", "asic")

    def __init__(self, module_name: str = "eml_nand_circuit", target: str = "generic") -> None:
        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"Invalid target '{target}'; must be one of {self.VALID_TARGETS}"
            )
        self.module_name = module_name
        self.target = target

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, circuit: NANDCircuit) -> Tuple[str, Dict]:
        """
        Generate structural gate-level Verilog for *circuit*.

        Parameters
        ----------
        circuit : NANDCircuit
            The NAND circuit to translate.

        Returns
        -------
        Tuple[str, Dict]
            A 2-tuple of ``(verilog_code, metadata)``.  The *metadata*
            dictionary contains:

            - ``module_name``  — generated module name
            - ``gate_count``   — number of NAND gates
            - ``depth``        — circuit depth
            - ``num_inputs``   — number of primary inputs
            - ``num_outputs``  — number of primary outputs
            - ``num_wires``    — number of internal wires
            - ``target``       — synthesis target
            - ``timing``       — timing annotation dict (see :func:`_estimate_timing`)
            - ``generated_at`` — ISO-8601 timestamp
        """
        lines: List[str] = []

        self._emit_header(lines, circuit)
        self._emit_module_declaration(lines, circuit)
        self._emit_wire_declarations(lines, circuit)
        self._emit_nand_primitive(lines)
        self._emit_gate_instances(lines, circuit)
        self._emit_output_assignments(lines, circuit)
        self._emit_footer(lines)

        verilog_code = "\n".join(lines) + "\n"

        timing = _estimate_timing(circuit)
        metadata = {
            "module_name": self.module_name,
            "gate_count": circuit.gate_count(),
            "depth": circuit.depth(),
            "num_inputs": circuit.num_inputs,
            "num_outputs": len(circuit.output_wires),
            "num_wires": sum(
                1 for g in circuit.gates if _is_internal(g.output, circuit.num_inputs)
            ),
            "target": self.target,
            "timing": timing,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        return verilog_code, metadata

    def generate_testbench(
        self,
        circuit: NANDCircuit,
        test_vectors: Optional[List[List[bool]]] = None,
    ) -> str:
        """
        Generate a Verilog testbench that drives the NAND circuit module.

        If *test_vectors* is ``None``, a full exhaustive truth-table test
        is generated (2^num_inputs vectors).  Each test vector is a list
        of booleans whose length must equal ``circuit.num_inputs``.

        Parameters
        ----------
        circuit : NANDCircuit
            The circuit for which to generate a testbench.
        test_vectors : list[list[bool]] or None
            Optional explicit test vectors.  When ``None``, all 2^n input
            combinations are generated automatically.

        Returns
        -------
        str
            Complete Verilog testbench source code.
        """
        if test_vectors is None:
            test_vectors = self._generate_exhaustive_vectors(circuit.num_inputs)

        tb_name = f"tb_{self.module_name}"
        lines: List[str] = []

        lines.append(f"// Testbench for {self.module_name}")
        lines.append(f"// Generated by EML Pipeline VerilogGenerator")
        lines.append(f"// Auto-generated at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        lines.append("")
        lines.append("`timescale 1ns / 1ps")
        lines.append("")
        lines.append(f"module {tb_name};")
        lines.append("")

        # Signal declarations
        lines.append(f"    reg [{circuit.num_inputs - 1}:0] inputs;")
        if len(circuit.output_wires) > 1:
            lines.append(f"    wire [{len(circuit.output_wires) - 1}:0] outputs;")
        else:
            lines.append("    wire outputs;")
        lines.append(f"    integer num_errors = 0;")
        lines.append(f"    integer num_tests = 0;")
        lines.append("")

        # Instantiate DUT
        port_map = []
        for i in range(circuit.num_inputs):
            port_map.append(f"        .in{i}(inputs[{i}])")
        if len(circuit.output_wires) > 1:
            for i, _ in enumerate(circuit.output_wires):
                port_map.append(f"        .out{i}(outputs[{i}])")
        else:
            port_map.append("        .out(outputs)")

        lines.append(f"    {self.module_name} dut(")
        lines.append(",\n".join(port_map))
        lines.append("    );")
        lines.append("")

        # Test stimulus
        lines.append("    initial begin")
        lines.append("        $display(\"=== EML Pipeline Testbench ===\");")
        lines.append(f"        $display(\"Module: {self.module_name}\");")
        lines.append(f"        $display(\"Inputs: {circuit.num_inputs}, Outputs: {len(circuit.output_wires)}\");")
        lines.append(f"        $display(\"Test vectors: {len(test_vectors)}\");")
        lines.append("        $display(\"\");")
        lines.append("")

        for idx, vector in enumerate(test_vectors):
            # Set inputs
            input_val = 0
            for bit_idx, bit_val in enumerate(vector):
                if bit_val:
                    input_val |= (1 << bit_idx)
            lines.append(f"        // Test vector {idx}")
            lines.append(f"        inputs = {circuit.num_inputs}'d{input_val};")
            lines.append(f"        #10;")

            # Check expected outputs
            expected = circuit.evaluate(vector)
            for out_idx, exp_val in enumerate(expected):
                exp_int = 1 if exp_val else 0
                if len(circuit.output_wires) > 1:
                    lines.append(
                        f"        if (outputs[{out_idx}] !== {exp_int}'b{exp_int}) begin"
                    )
                    lines.append(
                        f"            $display(\"FAIL: vec {idx} out{out_idx} "
                        f"expected={exp_int} got=%0b\", outputs[{out_idx}]);"
                    )
                    lines.append("            num_errors = num_errors + 1;")
                    lines.append("        end")
                else:
                    lines.append(
                        f"        if (outputs !== {exp_int}'b{exp_int}) begin"
                    )
                    lines.append(
                        f"            $display(\"FAIL: vec {idx} out "
                        f"expected={exp_int} got=%0b\", outputs);"
                    )
                    lines.append("            num_errors = num_errors + 1;")
                    lines.append("        end")
            lines.append(f"        num_tests = num_tests + {len(circuit.output_wires)};")

        lines.append("")
        lines.append("        $display(\"\");")
        lines.append("        $display(\"=== Test Summary ===\");")
        lines.append("        $display(\"Total checks: %0d\", num_tests);")
        lines.append("        $display(\"Errors: %0d\", num_errors);")
        lines.append("        if (num_errors == 0)")
        lines.append("            $display(\"PASS: All tests passed.\");")
        lines.append("        else")
        lines.append("            $display(\"FAIL: Some tests failed.\");")
        lines.append("        $finish;")
        lines.append("    end")
        lines.append("")
        lines.append("endmodule")

        return "\n".join(lines) + "\n"

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _generate_exhaustive_vectors(num_inputs: int) -> List[List[bool]]:
        """Generate all 2^num_inputs input combinations."""
        vectors: List[List[bool]] = []
        for val in range(1 << num_inputs):
            vector = [(val >> i) & 1 == 1 for i in range(num_inputs)]
            vectors.append(vector)
        return vectors

    def _emit_header(self, lines: List[str], circuit: NANDCircuit) -> None:
        """Emit the file-level header comment block."""
        lines.append("// " + "=" * 72)
        lines.append(f"// Module:     {self.module_name}")
        lines.append(f"// Source:     EML Pipeline (Epsilon-Mu Lambda NAND Duality)")
        lines.append(f"// Gate count: {circuit.gate_count()}")
        lines.append(f"// Depth:      {circuit.depth()}")
        lines.append(f"// Inputs:     {circuit.num_inputs}")
        lines.append(f"// Outputs:    {len(circuit.output_wires)}")
        lines.append(f"// Target:     {self.target}")
        lines.append("// Reference:  Sheffer stroke completeness (Theorem 3.1),")
        lines.append("//             Signal restoration (Theorem 4.2)")
        lines.append(f"// Generated:  {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        lines.append("// " + "=" * 72)
        lines.append("")

    def _emit_module_declaration(self, lines: List[str], circuit: NANDCircuit) -> None:
        """Emit the module port declaration."""
        port_list: List[str] = []
        for i in range(circuit.num_inputs):
            port_list.append(f"in{i}")
        for i, _ in enumerate(circuit.output_wires):
            if len(circuit.output_wires) == 1:
                port_list.append("out")
            else:
                port_list.append(f"out{i}")

        lines.append(f"module {self.module_name}(")
        lines.append(", ".join(f"    {p}" for p in port_list))
        lines.append(");")
        lines.append("")

    def _emit_wire_declarations(self, lines: List[str], circuit: NANDCircuit) -> None:
        """Emit input, output, and wire declarations with target-specific attributes."""

        # Input declarations
        for i in range(circuit.num_inputs):
            if self.target == "fpga":
                lines.append(f"    (* KEEP *) input in{i};")
            elif self.target == "asic":
                lines.append(f"    input in{i}; /* synthesis syn_keep=1 */")
            else:
                lines.append(f"    input in{i};")

        # Output declarations
        for i, _ in enumerate(circuit.output_wires):
            name = "out" if len(circuit.output_wires) == 1 else f"out{i}"
            if self.target == "fpga":
                lines.append(f"    (* KEEP *) output {name};")
            elif self.target == "asic":
                lines.append(f"    output {name}; /* synthesis syn_keep=1 */")
            else:
                lines.append(f"    output {name};")

        lines.append("")

        # Internal wire declarations — only for wires that are not
        # directly driven to a constant or used as a primary output
        internal_wires: set = set()
        for gate in circuit.gates:
            if _is_internal(gate.output, circuit.num_inputs):
                internal_wires.add(gate.output)

        # Also declare wires for output assignments when outputs are internal
        for w in circuit.output_wires:
            if _is_internal(w, circuit.num_inputs):
                internal_wires.add(w)

        for wire_id in sorted(internal_wires):
            name = _wire_name(wire_id, circuit.num_inputs)
            if self.target == "fpga":
                lines.append(f"    (* KEEP *) wire {name};")
            elif self.target == "asic":
                lines.append(f"    wire {name}; /* synthesis syn_keep=1 */")
            else:
                lines.append(f"    wire {name};")

        # If we have internal wires, add a blank line for readability
        if internal_wires:
            lines.append("")

    def _emit_nand_primitive(self, lines: List[str]) -> None:
        """Emit the NAND gate primitive declaration."""
        lines.append("    // NAND gate primitive")
        lines.append("    primitive nand(out, a, b);")
        lines.append("        output out;")
        lines.append("        input a;")
        lines.append("        input b;")
        lines.append("        table")
        lines.append("            // a  b : out")
        lines.append("            0  0 : 1 ;")
        lines.append("            0  1 : 1 ;")
        lines.append("            1  0 : 1 ;")
        lines.append("            1  1 : 0 ;")
        lines.append("        endtable")
        lines.append("    endprimitive")
        lines.append("")

    def _emit_gate_instances(self, lines: List[str], circuit: NANDCircuit) -> None:
        """Emit NAND gate instantiation statements."""
        lines.append("    // NAND gate instances")

        for gate in circuit.gates:
            a_name = _wire_name(gate.input_a, circuit.num_inputs)
            b_name = _wire_name(gate.input_b, circuit.num_inputs)
            out_name = _wire_name(gate.output, circuit.num_inputs)

            # If either input is a constant, we may need to inject a wire
            # for the Verilog primitive (primitives require net connections).
            # Constants 1'b0 and 1'b1 can be passed directly to gate
            # primitives in most simulators, but for maximum compatibility
            # we use assign statements for constant inputs when needed.
            lines.append(
                f"    nand g{gate.gate_id}_{out_name}({out_name}, {a_name}, {b_name});"
            )

        if circuit.gates:
            lines.append("")

    def _emit_output_assignments(self, lines: List[str], circuit: NANDCircuit) -> None:
        """Emit continuous assignments mapping internal wires to output ports."""
        has_assignment = False
        for i, w in enumerate(circuit.output_wires):
            out_name = "out" if len(circuit.output_wires) == 1 else f"out{i}"

            # If the output wire is an internal wire, assign it
            if _is_internal(w, circuit.num_inputs):
                wire_name = _wire_name(w, circuit.num_inputs)
                lines.append(f"    assign {out_name} = {wire_name};")
                has_assignment = True
            elif _is_input(w, circuit.num_inputs):
                # Passthrough from input — legal but unusual
                wire_name = _wire_name(w, circuit.num_inputs)
                lines.append(f"    assign {out_name} = {wire_name};")
                has_assignment = True
            elif w == -1:
                lines.append(f"    assign {out_name} = 1'b0;")
                has_assignment = True
            elif w == -2:
                lines.append(f"    assign {out_name} = 1'b1;")
                has_assignment = True

        if has_assignment:
            lines.append("")

    def _emit_footer(self, lines: List[str]) -> None:
        """Emit the module closing."""
        lines.append("endmodule")


# ─── Convenience Function ────────────────────────────────────────────────────

def circuit_to_verilog(
    circuit: NANDCircuit,
    module_name: str = "eml_nand_circuit",
) -> str:
    """
    Convenience function: convert a NANDCircuit to a Verilog string.

    This is a shorthand for :meth:`VerilogGenerator.generate` that returns
    only the Verilog code (not the metadata).

    Parameters
    ----------
    circuit : NANDCircuit
        The circuit to convert.
    module_name : str
        Name for the generated Verilog module.

    Returns
    -------
    str
        Structural gate-level Verilog source code.
    """
    gen = VerilogGenerator(module_name=module_name)
    verilog_code, _ = gen.generate(circuit)
    return verilog_code
