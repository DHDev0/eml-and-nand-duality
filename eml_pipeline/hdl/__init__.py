"""
HDL Package — Hardware Description Language Generation & Optimisation
=====================================================================

Provides Verilog generation, Yosys/ABC integration, and format converters
(BLIF, AIGER) for NAND circuits produced by the EML pipeline.

Key classes
-----------
VerilogGenerator
    Structural gate-level Verilog generation with target-specific directives.
YosysABCIntegration
    Yosys + ABC synthesis and optimisation with graceful degradation.
BLIFGenerator
    BLIF (Berkeley Logic Interchange Format) export for ABC.
AIGERGenerator
    AIGER binary format export for formal verification.
"""

from eml_pipeline.hdl.verilog_gen import (
    VerilogGenerator,
    circuit_to_verilog,
)
from eml_pipeline.hdl.yosys_abc import (
    AIGERGenerator,
    BLIFGenerator,
    ToolError,
    ToolNotFoundError,
    YosysABCIntegration,
)

__all__ = [
    "VerilogGenerator",
    "circuit_to_verilog",
    "YosysABCIntegration",
    "BLIFGenerator",
    "AIGERGenerator",
    "ToolError",
    "ToolNotFoundError",
]
