"""
Assembly Module — NAND ↔ Assembly Compilation
===============================================

Two compilation directions:
1. Forward:  NAND → Assembly (compile_nand_to_asm)
2. Reverse:  Assembly → NAND (decompile_asm)

Two optimization branches after NAND:
  Branch A (Pattern Rewrite): NAND → Pattern Rewrite → Assembly
  Branch B (Hardware):        NAND → Verilog → Yosys/ABC → FPGA/ASIC → Assembly

Supported architectures: x86-64, ARM64, RISC-V, MIPS, WebAssembly (WASM)
"""

from eml_pipeline.assembly.nand_to_asm import (
    NANDToAssembly,
    NANDToX86,
    NANDToARM,
    NANDToRISCV,
    NANDToMIPS,
    NANDToWASM,
    AssemblyOutput,
    compile_nand_to_asm,
    COMPILERS,
)

from eml_pipeline.assembly.asm_decompiler import (
    ASMParserBase,
    X86ASMParser,
    ARMASMParser,
    RISCVASMParser,
    MIPSASMParser,
    WASMParser,
    ParsedNANDOp,
    DecompileResult,
    NANDCircuitReconstructor,
    decompile_asm,
    decompile_with_metadata,
)

from eml_pipeline.assembly.optimal_asm_gen import (
    OptimalAssemblyOutput,
    OptimalAssemblyGenerator,
    RegisterAllocator,
    generate_pattern_branch_asm,
    generate_hardware_branch_asm,
    measure_asm_generation_error,
)
