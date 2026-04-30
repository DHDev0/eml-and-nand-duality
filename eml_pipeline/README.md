# EML–NAND Duality Pipeline

A complete bidirectional compilation/decompilation pipeline implementing the mathematical framework from **"The EML–NAND Duality: A Mathematical Bridge Between Continuous and Discrete Universality"** by Daniel Derycke (2026), arXiv:2603.21852.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Core Mathematical Framework](#core-mathematical-framework)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: LaTeX → EML](#stage-1-latex--eml)
  - [Stage 2: EML → NAND (T1)](#stage-2-eml--nand-t1)
  - [Stage 3: NAND Pattern Rewriting](#stage-3-nand-pattern-rewriting)
  - [Stage 4: Verilog HDL Generation](#stage-4-verilog-hdl-generation)
  - [Stage 5: Yosys/ABC Optimization](#stage-5-yosysabc-optimization)
  - [Stage 6: FPGA/ASIC Synthesis](#stage-6-fpgaasic-synthesis)
  - [Reverse Pipeline (T3+T4)](#reverse-pipeline-t3t4)
- [Error Measurement](#error-measurement)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Paper References](#paper-references)

---

## Overview

This pipeline bridges two universal computational primitives — the **EML operator** from continuous mathematics and the **NAND gate** from discrete logic — through a closed, self-correcting cycle of four transitions (T1–T4). The result is a mathematically rigorous path from mathematical expressions written in LaTeX all the way to silicon (FPGA/ASIC), and back.

**Forward direction (Dual-Branch Architecture):**

```
                    ┌─── Branch A (Software) ────→ Assembly
                    │   Pattern Rewrite → Assembly
LaTeX → EML → NAND ─┤
                    │   Verilog → Yosys/ABC → FPGA/ASIC → Assembly
                    └─── Branch B (Hardware) ─────→ Assembly
```

**Reverse direction (Four Transitions + Assembly Decompilation):**

```
Assembly → NAND → ε-NAND → ApproxEML → EML → LaTeX
           T-asm   T2        T3          T4
```

The two universal primitives:

- **EML operator**: `eml(x, y) = eˣ − ln(y)` — a single binary operator that generates *all* elementary functions via the grammar `S → 1 | eml(S, S)`
- **NAND gate**: `NAND(a, b) = ¬(a ∧ b)` — a single binary gate that generates *all* Boolean functions

The four transitions form a **closed self-correcting cycle**:

```
              T1 (Theorem 2.6)
    EML  ──────────────────────→  NAND
     ↑                              ↓
     │                              │ T2 (Definition 3.1)
     │                              ↓
     │                           ε-NAND
     ↑                              ↓
     │                              │ T3 (Section 5, Taylor + artanh)
     │                              ↓
  EML  ←────────────────────── ApproxEML
              T4 (Newton-Raphson)

    Round-trip error ≤ 2ε  (Section 8)
```

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       FORWARD PIPELINE (Dual-Branch)                      │
│                                                                           │
│  LaTeX ──→ EML ──→ NAND ──→ Rewrite ──┬──→ Assembly (Branch A: Software)│
│              │        │         │        │   Pattern-optimized asm         │
│           Lemma 2.3  Thm 2.6  Thm 4.2  │                                │
│           Lemma 2.4           Prop 3.9  ├──→ Verilog (Branch B: HW)      │
│                                          │   → Yosys/ABC → FPGA/ASIC     │
│                                          │   → Hardware-optimized asm     │
│                                          └────────────────────────────────│
├──────────────────────────────────────────────────────────────────────────┤
│                       REVERSE PIPELINE (T1-T4)                            │
│                                                                           │
│  Assembly ──→ NAND ──→ ε-NAND ──→ ApproxEML ──→ EML ──→ LaTeX           │
│     T-asm      │        │             │            │        │             │
│   Decompile  Def 3.1   §5 Taylor    T4 NR     Pattern                  │
│   x86/ARM/   noise     + artanh    O(ε²ᵏ)   recognition               │
│   RISC-V/                                                                 │
│   MIPS/WASM                                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Mathematical Framework

### The EML Operator and Grammar

The **EML operator** is defined as:

> `eml(x, y) = eˣ − ln(y)`

The **EML grammar** is:

> `S → 1 | eml(S, S)`

This two-symbol grammar generates *every* elementary function as a finite binary tree. No additional constants, operators, or special functions are required.

### Key Constructions (Lemma 2.3)

Every elementary function is a finite binary tree over `eml` and `1`:

| Function | EML Tree | Verification |
|----------|----------|-------------|
| `exp(x)` | `eml(x, 1)` | `eˣ − ln(1) = eˣ` |
| `e` | `eml(1, 1)` | `e¹ − ln(1) = e` |
| `ln(x)` | `eml(1, eml(eml(1, x), 1))` | Three-level construction (see below) |
| `0` | `eml(1, eml(e, 1))` | `e − ln(eᵉ) = e − e = 0` |
| `1 − y` | `eml(0, eʸ)` | `e⁰ − ln(eʸ) = 1 − y` |

**ln(x) verification:**

```
eml(1, x)          = e − ln(x)
eml(e − ln(x), 1)  = e^(e − ln(x)) = eᵉ / x
eml(1, eᵉ / x)     = e − ln(eᵉ / x) = e − (e − ln(x)) = ln(x)  ✓
```

### Arithmetic from EML (Lemma 2.4)

| Operation | EML Construction | Domain |
|-----------|-----------------|--------|
| `x − y` | `eml(ln(x), eʸ)` | `x > 0` |
| `x + y` | `ln(exp(x) · exp(y))` | All `x, y` |
| `x × y` | `exp(ln(x) + ln(y))` | `x, y > 0` |
| `x / y` | `exp(ln(x) − ln(y))` | `x > 0, y > 0` |

### The Soft NAND Bridge (Theorem 2.6)

The key bridge connecting continuous and discrete computation:

```
NAND_ℝ(a, b) = 1 − ab = eml(0, e^{ab})
```

Where the multiplication `ab = exp(ln(a) + ln(b))` is itself an EML tree, giving total tree depth ≈ 12 levels. On the open interior `(0,1]²` this is exact; at Boolean corners it extends via continuous limit (Remark 2.6a).

### Four Transitions

| Transition | Direction | Nature | Error Bound |
|-----------|-----------|--------|-------------|
| **T1**: EML → NAND | Algebraic identity | Exact on interior, limit at boundary | 0 |
| **T2**: NAND → ε-NAND | Domain relaxation + noise | Approximate | ≤ ε |
| **T3**: ε-NAND → ApproxEML | Circuit construction | Constructive (Taylor + artanh) | ≤ ε |
| **T4**: ApproxEML → EML | Uniform limit + correction | Newton-Raphson | O(ε^{2ᵏ}) |

### Signal Restoration (Theorem 4.2)

The iterated double-NAND provides intrinsic error correction:

```
R(x) = NAND(NAND(x, x), NAND(x, x))
T(x) = 2x² − x⁴     (ideal dynamics)
```

**Contraction**: `δ' = 4δ² + 4ε < δ` (requires `δ < 2/9`, `ε < δ²/8`)

**Fixed point**: `δ* ≈ 4ε` (requires `ε ≤ 1/64`)

**Self-correcting**: After `k` round-trip cycles, error is `O(ε^{2ᵏ})` (doubly-exponential convergence).

---

## Pipeline Stages

### Stage 1: LaTeX → EML

**Module**: `eml_pipeline.eml.latex_to_eml`, `eml_pipeline.parsers.latex_parser`, `eml_pipeline.search.eml_search`

Parses LaTeX mathematical expressions and decomposes every operation into pure `eml(x, y)` binary trees.

**Components:**

- **LaTeX parser** (`parsers/latex_parser.py`) — comprehensive parser supporting 50+ LaTeX math constructs:
  - **Arithmetic**: `+`, `-`, `\times`, `\cdot`, `/`, `\frac{}{}`
  - **Powers/Roots**: `a^b`, `a^{bc}`, `\sqrt{x}`, `\sqrt[n]{x}`
  - **Exp/Log**: `e^{x}`, `\exp(x)`, `\ln(x)`, `\log(x)`, `\log_b(x)`
  - **Trig**: `\sin`, `\cos`, `\tan`, `\cot`, `\sec`, `\csc`, `\arcsin`, `\arccos`, `\arctan`
  - **Hyperbolic**: `\sinh`, `\cosh`, `\tanh`
  - **Calculus**: `\int`, `\oint`, `\sum`, `\prod`, `\lim`
  - **Special**: `\pi`, `\infty`, `\epsilon`, `\hbar`
  - **Greek**: Full lowercase + uppercase Greek alphabet
  - **Decorations**: `\hat{x}`, `\bar{x}`, `\vec{x}`, `\tilde{x}`, `\dot{x}`
  - **Structures**: Matrices, cases, binomials, factorials, absolute values

- **Converter** (`eml/latex_to_eml.py`) — decomposes every operation into pure `eml(x,y)` binary trees using the constructions from Lemma 2.3 and Lemma 2.4. Trigonometric functions use Taylor series decomposed entirely into EML arithmetic.

- **Search** (`search/eml_search.py`) — find operations by name, category, or LaTeX pattern. Domain-specific variable tracking:

  | Variable | Quantum Mechanics | Thermodynamics | Electromagnetism | General Relativity |
  |----------|------------------|----------------|------------------|--------------------|
  | ψ | Wave function | Stream function | Magnetic flux | — |
  | φ | Scalar field/phase | — | Electric potential | — |
  | η | — | Efficiency | — | — |

### Stage 2: EML → NAND (T1, Theorem 2.6)

**Module**: `eml_pipeline.transitions.t1_eml_to_nand`, `eml_pipeline.nand.nand_core`

Converts EML expression trees into NAND gate circuits via the soft NAND bridge.

**Key identity:**

```
NAND_ℝ(a, b) = 1 − ab = eml(0, e^{ab})
```

**Conversion process:**

1. Traverse the EML tree bottom-up
2. At each soft NAND node, emit NAND gate circuitry
3. Fixed-point encoding maps continuous values `[0, 1]` to integer wire values
4. Bit-width is configurable (default: 16-bit)

**Circuit model** (`nand/nand_core.py`):

- `NANDCircuit` — stores gates, inputs, outputs, and metadata
- `NANDGate` — a single `NAND(a, b)` gate with input/output wire IDs
- **Boolean evaluation**: `circuit.evaluate(inputs: List[bool])`
- **Soft evaluation**: `circuit.evaluate_soft(inputs: List[float], epsilon=0.0)` — evaluates using `1 − ab` with optional ε-NAND noise
- **Signal restoration**: `R(x) = NAND(NAND(x,x), NAND(x,x))` with contraction computation

### Stage 3: NAND Pattern Rewriting

**Module**: `eml_pipeline.nand.pattern_rewriter`

Simplifies NAND gate circuits using Boolean algebra identities and structural optimization. This is the "fast simplification" step before Verilog emission.

**Optimization pipeline** (run in order by `optimize()`):

| Step | Function | Description |
|------|----------|-------------|
| 1 | `propagate_constants()` | Fold constant-input gates (e.g., `NAND(a, 0) → 1`) |
| 2 | `structural_hash()` | Merge duplicate subgraphs into canonical DAG form (strashing) |
| 3 | `NANDPatternRewriter.rewrite()` | Apply Boolean algebra identities iteratively: constant folding, absorption, De Morgan, identity simplification |
| 4 | `eliminate_dead_gates()` | Remove gates whose outputs are never referenced |
| 5 | `simplify_restoration()` | Collapse `R(x) = x` patterns for Boolean wires (Theorem 4.2) |
| 6 | `structural_hash()` | Final canonical form pass |

**Equivalence verification** (`verify_equivalence()`):

- Circuits with ≤ 6 inputs: exhaustive Boolean testing (2ⁿ test cases)
- Larger circuits: random Boolean + soft value testing
- Soft equivalence: random inputs in `[0, 1]` with tolerance 0.01

### Stage 4: Verilog HDL Generation

**Module**: `eml_pipeline.hdl.verilog_gen`

Generates structural gate-level Verilog from `NANDCircuit` representations.

**Wire conventions** (from `nand_core`):

| Wire ID | Verilog Name | Meaning |
|---------|-------------|---------|
| −1 | `1'b0` | Constant zero |
| −2 | `1'b1` | Constant one |
| `0..num_inputs−1` | `in0, in1, ...` | Primary inputs |
| `num_inputs..` | `w0, w1, ...` | Internal wires |

**Three synthesis targets:**

| Target | Attributes | Use Case |
|--------|-----------|----------|
| `generic` | Plain Verilog | Simulation, verification |
| `fpga` | `(* KEEP *)` attributes on wires | FPGA synthesis (prevents optimization removing key signals) |
| `asic` | `/* synthesis syn_keep=1 */` directives | ASIC tool compatibility |

**Output includes:**

- NAND gate primitive declaration
- Gate instantiation statements
- Output wire assignments
- Timing annotation metadata (unit-delay model, ~0.02 ns/gate in 45 nm CMOS)
- Automatic testbench generation (`VerilogGenerator.generate_testbench()`) with exhaustive truth-table coverage

### Stage 5: Yosys/ABC Optimization

**Module**: `eml_pipeline.hdl.yosys_abc`

Integrates industrial logic optimization tools with graceful degradation when tools are not installed.

**Yosys** (RTL synthesis):

- Reads the generated Verilog netlist
- Runs `hierarchy -check` and synthesis passes
- Outputs optimized Verilog and statistics

**ABC** (technology mapping):

- AIG rewriting and DAG-aware optimization
- LUT mapping for FPGA targets (`abc -lut 4` or `abc -lut 6`)
- Standard cell mapping for ASIC (`abc -liberty`)

**Output formats:**

- BLIF (Berkeley Logic Interchange Format) — ABC's native format
- AIGER (And-Inverter Graph) — for formal verification

**Graceful degradation:** When Yosys/ABC are not installed, the pipeline falls back to pure-Python optimization (Stage 3 pattern rewriting) and reports `yosys_available: False` in metadata. All downstream stages continue with analytical estimates.

### Stage 6: FPGA/ASIC Synthesis

**Module**: `eml_pipeline.hdl.synthesis`

Provides synthesis support targeting FPGA and ASIC flows with resource estimation.

**Supported FPGA families:**

| Family | Device | Synthesis Command | Constraint Format |
|--------|--------|------------------|-------------------|
| `xc7` | Xilinx Artix-7 | `synth_xilinx` + `abc -lut 6` | XDC |
| `ice40` | Lattice iCE40 | `synth_ice40` + `abc -lut 4` | PCF |
| `ecp5` | Lattice ECP5 | `synth_ecp5` + `abc -lut 4` | LPF |
| `generic` | — | `synth` + `abc -lut 4` | PCF |

**FPGA resource estimation** (when Yosys is unavailable):

- **LUT count**: `(gate_count + 1) // 2` (conservative: ~2 gates per 4-input LUT)
- **FF count**: 0 (purely combinational)
- **BRAM count**: 0
- **Max frequency**: `F_base / (1 + depth × 0.02)` where `F_base` is family-specific (e.g., 450 MHz for xc7)

**Supported ASIC PDKs:**

| PDK | Node | Gate Area (µm²) | Gate Delay (ns) | Gate Power (µW) |
|-----|------|-----------------|-----------------|-----------------|
| `sky130` | SkyWater 130 nm | 10.0 | 0.05 | 0.01 |
| `gf180mcu` | GlobalFoundries 180 nm | 25.0 | 0.12 | 0.03 |
| `asicore` | Generic | 15.0 | 0.08 | 0.015 |

**ASIC metric estimation**: `area = gates × area_per_gate`, `delay = depth × delay_per_gate`, `power = gates × power_per_gate`

**Constraint generation**: Automatic pin assignment files (PCF/XDC/LPF) with placeholder pin mappings.

**OpenLane integration**: When OpenLane is installed and configured, the ASIC synthesizer can invoke the full physical design flow (synthesis → placement → routing). Without OpenLane, analytical estimates are returned with `is_estimated: True`.

**Synthesis report** (`SynthesisReport` dataclass):

- Structured report with `target_type`, `target_name`, gate counts, LUT/FF/BRAM counts (FPGA), area/delay/power (ASIC), Verilog code, constraints, and `is_estimated` flag
- Human-readable summary via `report.summary()`

### Reverse Pipeline (T3+T4)

**Module**: `eml_pipeline.transitions.t3_t4_nand_to_eml`, `eml_pipeline.reverse.reverse_pipeline`, `eml_pipeline.epsilon_nand.epsilon_nand`

Recovers continuous EML expressions from discrete NAND circuits through the ε-NAND framework.

**T2: NAND → ε-NAND** (Definition 3.1):

- Introduces bounded noise: `|G_ε(a, b) − (1 − ab)| ≤ ε`
- Error propagation through depth-`d` circuits: `δ_d ≤ 2^d · δ₀ + (2^d − 1) · ε` (Corollary 3.8)

**T3: ε-NAND → ApproxEML** (Section 5):

- Taylor series computation of `exp` and `ln` via ε-NAND arithmetic
- Uses `artanh` for stable ln computation: `ln(x) = 2 · artanh((x−1)/(x+1))`
- Taylor remainder bounds: `|R_N| ≤ M^{N+1}/(N+1)! · e^M`

**T4: ApproxEML → EML** (Newton-Raphson correction):

- Applies Newton-Raphson iteration to recover exact EML values from approximations
- Self-correcting: each iteration squares the error → `O(ε^{2ᵏ})` after `k` cycles
- Combined with signal restoration (Theorem 4.2) for robust error correction

**Reverse pipeline flow:**

```
NAND circuit → ε-NAND evaluation → Taylor exp/ln computation
    → ApproxEML tree → Newton-Raphson correction → EML tree → LaTeX
```

---

## Error Measurement

### Translation Error Tracker

**Module**: `eml_pipeline.utils.translation_error`

Records error at every pipeline stage and verifies compliance against the paper's theoretical bounds.

**Key components:**

| Component | Description |
|-----------|-------------|
| `TranslationErrorTracker` | Records and aggregates errors across all stages with UUID-based tracking |
| `PaperBoundChecker` | Static utilities for verifying theoretical bounds |
| `ErrorVisualization` | Formatting for human-readable error and compliance reports |
| `measure_full_pipeline_error()` | End-to-end error measurement across all stages |

**Paper bounds verified:**

| Bound | Formula | Reference |
|-------|---------|-----------|
| Round-trip error | `≤ 2ε` | Section 8 |
| Signal restoration contraction | `δ' = 4δ² + 4ε < δ` | Theorem 4.2 |
| Fixed point | `δ* ≈ 4ε` (requires `ε < 1/64`) | Theorem 4.2 |
| ε-NAND gate error | `\|G_ε(a,b) − (1−ab)\| ≤ ε` | Definition 3.1 |
| Depth-d error propagation | `δ_d ≤ 2^d · δ₀ + (2^d − 1) · ε` | Corollary 3.8 |
| Taylor exp remainder | `\|R_N\| ≤ M^{N+1}/(N+1)! · e^M` | Standard bound |
| Taylor ln remainder | `\|R'_N(y)\| ≤ 2C^{2N+3}/((2N+3)(1−C²))` | Standard bound |
| Self-correcting cycle | `O(ε^{2ᵏ})` after `k` steps | Section 8 |
| ε-NOT error | `\|η₁\| ≤ ε` | Proposition 3.3 |
| ε-AND error | `\|η₁ − η₂\| ≤ 2ε` | Proposition 3.5 |
| ε-OR error | `\|E_OR\| ≤ 3ε + ε²` | Proposition 3.7 |

**Complete error bounds summary:**

| Metric | Formula | Bound |
|--------|---------|-------|
| ε-NOT error | `|η₁|` | ≤ ε |
| ε-AND error | `|η₁ − η₂|` | ≤ 2ε |
| ε-OR error | `|E_OR|` | ≤ 3ε + ε² |
| Depth-d error | `δ_d` | ≤ 2^d·δ₀ + (2^d−1)·ε |
| Restoration contraction | `δ'` | = 4δ² + 4ε < δ |
| Fixed point | `δ*` | ≈ 4ε |
| Round-trip error | `E_RT` | ≤ 2ε |
| NR correction | `E_{k+1}` | = O(ε^{2ᵏ}) |
| Taylor exp | `R_N(x)` | ≤ M^{N+1}/(N+1)! · e^M |
| Taylor ln | `R'_N(y)` | ≤ 2C^{2N+3}/((2N+3)(1−C²)) |

---

## Project Structure

```
eml_pipeline/
├── __init__.py
├── pipeline.py                   # Main orchestrator (EMLNANDPipeline, dual-branch)
├── eml/
│   ├── eml_core.py               # EML operator, primitives (Lemma 2.3/2.4)
│   └── latex_to_eml.py           # LaTeX → EML decomposition
├── nand/
│   ├── nand_core.py              # NAND circuits, Boolean/soft evaluation, signal restoration
│   └── pattern_rewriter.py       # Gate simplification (hashing, constant prop, dead elim)
├── transitions/
│   ├── t1_eml_to_nand.py         # T1: EML → NAND (Theorem 2.6)
│   └── t3_t4_nand_to_eml.py     # T3+T4: NAND → ApproxEML → EML
├── epsilon_nand/
│   └── epsilon_nand.py           # ε-NAND framework, signal restoration, bounded noise
├── hdl/
│   ├── verilog_gen.py            # Verilog HDL generation (generic/FPGA/ASIC targets)
│   ├── yosys_abc.py              # Yosys/ABC integration with graceful fallback
│   └── synthesis.py              # FPGA/ASIC synthesis (FPGASynthesizer, ASICSynthesizer)
├── assembly/
│   ├── nand_to_asm.py            # NAND → Assembly (x86, ARM, RISC-V, MIPS, WASM)
│   ├── asm_decompiler.py         # Assembly → NAND decompilation (all 5 architectures)
│   └── optimal_asm_gen.py        # Optimal assembly from FPGA/ASIC synthesis
├── reverse/
│   └── reverse_pipeline.py       # Reverse pipeline (T2+T3+T4, assembly decompilation)
├── parsers/
│   └── latex_parser.py           # LaTeX math parser (50+ constructs)
├── search/
│   └── eml_search.py             # Operation search, variable lookup, domain categories
├── utils/
│   ├── error_metrics.py          # Error analysis (signal restoration, round-trip, contraction)
│   └── translation_error.py      # Translation error tracking, paper bounds verification
├── tests/
│   ├── test_bidirectional_pipeline.py  # Comprehensive bidirectional tests (66 tests)
│   ├── test_extended_pipeline.py       # Extended tests (HDL, synthesis, error tracking)
│   └── __init__.py
└── test_pipeline.py              # Original test suite (15 suites, 190+ checks)
```

---

## Installation

### Core Pipeline

The core pipeline is pure Python with no external dependencies:

```bash
# Install in editable mode
cd /home/z/my-project/download
pip install -e .
```

### Optional: Yosys + ABC (Industrial Optimization)

For industrial-strength logic optimization and technology mapping:

```bash
# Install Yosys from source
git clone https://github.com/YosysHQ/yosys
cd yosys
make -j$(nproc)
sudo make install

# Install ABC from source
git clone https://github.com/berkeley-abc/abc
cd abc
make -j$(nproc)
sudo ln -s $(pwd)/abc /usr/local/bin/abc
```

### Optional: OpenLane (ASIC Physical Design)

For full ASIC physical design (synthesis → placement → routing → GDSII):

```bash
# Install OpenLane
git clone https://github.com/The-OpenROAD-Project/OpenLane
cd OpenLane
make
# Configure PDKs (sky130, gf180mcu)
make pdk
```

> **Note:** The pipeline works fully without Yosys, ABC, or OpenLane. When these tools are absent, the pipeline uses analytical estimates and reports `is_estimated: True` in synthesis metadata.

---

## Quick Start

### 1. Basic LaTeX → EML Conversion

```python
from eml_pipeline.eml.eml_core import eml_exp, eml_ln, eml_soft_nand, eml_evaluate, VAR

# Construct EML trees directly
exp_x = eml_exp(VAR("x"))                # exp(x) = eml(x, 1)
ln_x = eml_ln(VAR("x"))                  # ln(x) = eml(1, eml(eml(1, x), 1))
nand_ab = eml_soft_nand(VAR("a"), VAR("b"))  # 1 − ab = eml(0, e^{ab})

# Evaluate numerically
value = eml_evaluate(exp_x, {"x": 1.5})  # Returns e^1.5 ≈ 4.4817
```

### 2. EML → NAND with Theorem 2.6

```python
from eml_pipeline.eml.latex_to_eml import latex_to_eml
from eml_pipeline.transitions.t1_eml_to_nand import eml_to_nand

# Parse LaTeX and convert to EML
eml_tree, meta = latex_to_eml(r"\exp(x) + \ln(y)")

# Convert EML to NAND circuit (Theorem 2.6)
circuit, t1_meta = eml_to_nand(eml_tree, bit_width=16, epsilon=0.001)

print(f"Gates: {circuit.gate_count()}, Depth: {circuit.depth()}")
```

### 3. NAND Circuit Optimization

```python
from eml_pipeline.nand.pattern_rewriter import optimize

# Run the full optimization pipeline
optimized, opt_meta = optimize(circuit)

print(f"Original gates: {opt_meta['original_gates']}")
print(f"Optimized gates: {opt_meta['optimized_gates']}")
print(f"Gates saved: {opt_meta['gates_saved']} ({opt_meta['optimization_ratio']:.1%})")
print(f"Equivalence verified: {opt_meta['functional_equivalence_verified']}")
```

### 4. Full Pipeline to Verilog

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(bit_width=16, epsilon=0.001, taylor_order=8)

# Convert LaTeX directly to Verilog
result = pipe.to_verilog(r"\exp(x)", module_name="exp_circuit", target="fpga")

if result["success"]:
    print(result["verilog_code"])
    print(f"Gate count: {result['gate_count_original']} → {result['gate_count_optimized']}")
```

### 5. FPGA Resource Estimation

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(bit_width=16, epsilon=0.001)

# Estimate FPGA resources for Xilinx Artix-7
result = pipe.to_fpga(r"\sin(x) + \cos(x)", fpga_family="xc7")

if result["success"]:
    report = result["synthesis_report"]
    print(f"LUTs: {report['lut_count']}")
    print(f"Max freq: {report['max_freq_mhz']} MHz")
    print(f"Estimated: {report['is_estimated']}")
    print(result["summary"])
```

### 6. ASIC Metric Estimation

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(bit_width=16, epsilon=0.001)

# Estimate ASIC metrics for SkyWater 130nm
result = pipe.to_asic(r"\exp(x)", pdk="sky130")

if result["success"]:
    report = result["synthesis_report"]
    print(f"Area: {report['area_um2']} µm²")
    print(f"Delay: {report['delay_ns']} ns")
    print(f"Power: {report['power_uw']} µW")
    print(result["summary"])
```

### 7. Error Measurement with Paper Compliance

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(bit_width=16, epsilon=0.001)

# Measure errors at every stage and verify against paper bounds
result = pipe.measure_pipeline_error(r"\exp(x)", {"x": 0.5})

if result["success"]:
    # Stage-by-stage errors
    print("LaTeX → EML error:", result["latex_to_eml_error"]["absolute_error"])
    print("EML → NAND error:", result["eml_to_nand_error"]["absolute_error"])

    # Paper compliance
    compliance = result["paper_compliance"]
    print("Round-trip ≤ 2ε:", compliance["round_trip"]["satisfied"])
    print("Contraction holds:", compliance["contraction"]["contraction_holds"])
    print("Fixed point δ*:", compliance["fixed_point"]["delta_star"])
```

### 8. Round-Trip Test

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(bit_width=16, epsilon=0.001, taylor_order=8)

# Full round-trip: LaTeX → EML → NAND → ε-NAND → ApproxEML → EML
result = pipe.round_trip(r"\exp(x)", {"x": 0.5})

# Check paper bound: round-trip error ≤ 2ε = 0.002
rt_analysis = result["error_analysis"]
print(f"Epsilon: {rt_analysis['epsilon']}")
print(f"Theoretical bound: {rt_analysis['theoretical_bound']}")
print(f"EML value: {rt_analysis['eml_value']}")
```

### 9. Dual-Branch Assembly Generation

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(bit_width=16, epsilon=0.001, taylor_order=8)

# Branch A: Pattern Rewrite → Assembly (fast Boolean simplification)
result = pipe.forward_full(r"\exp(x)", {"x": 1.0},
                           target="pattern_asm", asm_arch="x86")
branch_a = result["stages"]["pattern_branch_asm"]
print(f"Branch A gates: {branch_a['gate_count']}")
print(f"Optimization ratio: {branch_a['optimization_ratio']:.2%}")

# Branch B: Hardware → Assembly (Yosys/ABC optimized)
result = pipe.forward_full(r"\exp(x)", {"x": 1.0},
                           target="hardware_asm", asm_arch="arm")
branch_b = result["stages"]["hardware_branch_asm"]
print(f"Branch B synthesis: {branch_b['synthesis_source']}")
```

### 10. Assembly Decompilation (Reverse from Assembly)

```python
from eml_pipeline.assembly.asm_decompiler import decompile_asm, decompile_with_metadata

# Decompile assembly code back to NAND circuit
asm_code = """
    movq %rax, in0
    andq %rax, in1
    notq %rax
    movq %rax, out
"""
result = decompile_asm(asm_code, "x86_64")
print(f"Reconstructed {result.gates_reconstructed} NAND gates")

# With forward metadata for enhanced reconstruction
result = decompile_with_metadata(asm_code, "x86_64", forward_metadata)
print(f"Gate count match: {result.metadata.get('gate_count_match')}")

# Full reverse: Assembly → NAND → EML → LaTeX
pipe_result = pipe.reverse_from_asm(asm_code, "x86_64", test_inputs=[0.5, 0.5])
```

### 11. Round-Trip Through Assembly

```python
from eml_pipeline.pipeline import EMLNANDPipeline

pipe = EMLNANDPipeline(epsilon=0.001, taylor_order=8)

# Full round-trip: LaTeX → Assembly → LaTeX
result = pipe.round_trip_asm(r"\exp(x)", {"x": 1.0}, arch="x86", use_metadata=True)
print(f"Forward: {result['forward']['stages'].keys()}")
print(f"Reverse: {result['reverse']['stages'].keys()}")
```

---

## Running Tests

```bash
cd /home/z/my-project/download

# Comprehensive bidirectional tests (66 tests covering all transitions)
python -m pytest eml_pipeline/tests/test_bidirectional_pipeline.py -v

# Original test suite (15 test suites, 190+ checks)
python -m eml_pipeline.test_pipeline

# Extended test suite (HDL generation, synthesis, error tracking)
python -m eml_pipeline.tests.test_extended_pipeline
```

**Test coverage:**

| Test Suite | Description | Reference |
|-----------|-------------|-----------|
| T1: EML → NAND | Soft NAND bridge, EML primitives | Theorem 2.6 |
| T2: NAND → ε-NAND | Noisy gates, error bounds | Definition 3.1 |
| T3: ε-NAND → ApproxEML | Taylor series exp/ln | Section 5.4 |
| T4: ApproxEML → EML | Newton-Raphson correction | Section 6 |
| Forward Pipeline | All LaTeX expressions, dual-branch | — |
| Reverse Pipeline | With/without metadata | T2+T3+T4 |
| Round-Trip | Forward+reverse, with/without metadata | Section 8 |
| Signal Restoration | Contraction, fixed point | Theorem 4.2 |
| Assembly Decompilation | All 5 architectures | x86/ARM/RISC-V/MIPS/WASM |
| Optimal Assembly | Pattern+Hardware branches | Branch A+B |
| Pattern Rewriter | Constant prop, hashing, dead elim | Theorem 4.2 |
| LaTeX Support | Sums, products, integrals, limits | Lemma 2.3/2.4 |
| EML Core | exp, ln, zero, complement, arithmetic | Lemma 2.3/2.4 |
| Error Measurement | Paper-bound compliance | Section 8, Theorem 4.2 |
| EML Core Primitives | `exp`, `ln`, zero, complement, `e` | Lemma 2.3 |
| EML Arithmetic | Add, subtract, multiply, divide, power | Lemma 2.4 |
| Soft NAND | `1 − ab = eml(0, e^{ab})` | Theorem 2.6 |
| ε-NAND Framework | Noisy gates, error bounds | Definition 3.1 |
| Taylor Series | `exp`, `ln` via `artanh` | Section 5.4 |
| Newton-Raphson Correction | Quadratic convergence | T4 |
| LaTeX Parser | 50+ constructs | — |
| LaTeX → EML Conversion | Round-trip with evaluation | Lemma 2.3/2.4 |
| NAND → Assembly | x86, ARM, RISC-V, MIPS, WASM | — |
| EML ↔ NAND Round Trip | Bidirectional | Theorem 2.6 |
| Reverse Pipeline | EML → LaTeX | T3+T4 |
| Error Measurement | All metrics | Section 8, Theorem 4.2 |
| Search Module | Operation catalog | — |
| Full Pipeline | End-to-end | — |

---

## Paper References

**Primary reference:**

Daniel Derycke, "The EML–NAND Duality: A Mathematical Bridge Between Continuous and Discrete Universality", April 2026. arXiv:2603.21852.

**Theorems and lemmas implemented:**

| Reference | Statement | Implementation |
|-----------|-----------|---------------|
| Definition 2.1 | EML operator `eml(x,y) = eˣ − ln(y)` | `eml_core.py` |
| Lemma 2.3 | EML primitives: `exp`, `ln`, `0`, `1−y` | `eml_core.py` |
| Lemma 2.4 | EML arithmetic: `+`, `−`, `×`, `÷` | `eml_core.py` |
| Remark 2.4a | Domain restrictions for `ln(x)` (x > 0) | `eml_core.py` |
| Theorem 2.6 | Soft NAND bridge: `1 − ab = eml(0, e^{ab})` | `t1_eml_to_nand.py`, `eml_core.py` |
| Remark 2.6a | Continuous limit at Boolean corners | `t1_eml_to_nand.py` |
| Theorem 3.1 | Sheffer stroke completeness | `nand_core.py` |
| Definition 3.1 | ε-NAND gate: `\|G_ε(a,b) − (1−ab)\| ≤ ε` | `epsilon_nand.py` |
| Proposition 3.3 | ε-NOT error ≤ ε | `epsilon_nand.py` |
| Proposition 3.5 | ε-AND error ≤ 2ε | `epsilon_nand.py` |
| Proposition 3.7 | ε-OR error ≤ 3ε + ε² | `epsilon_nand.py` |
| Proposition 3.9 | Error propagation through depth | `epsilon_nand.py` |
| Corollary 3.8 | Depth-d error: `δ_d ≤ 2^d·δ₀ + (2^d−1)·ε` | `epsilon_nand.py` |
| Theorem 4.2 | Signal restoration: `δ' = 4δ² + 4ε < δ` | `nand_core.py`, `pattern_rewriter.py` |
| Section 5.4 | Taylor series for `exp` and `ln` via `artanh` | `t3_t4_nand_to_eml.py` |
| Section 8 | Round-trip error ≤ 2ε | `translation_error.py`, `error_metrics.py` |
| Section 8 | Self-correcting cycle `O(ε^{2ᵏ})` | `translation_error.py`, `t3_t4_nand_to_eml.py` |

**Key bibliographic references from the paper:**

| Ref | Citation | Relevance |
|-----|----------|-----------|
| [1] | Odrzywołek | EML operator discovery |
| [2] | Sheffer stroke / NAND universality | Functional completeness of NAND |
| [5] | Fuzzy logic / product t-norm | Soft NAND and continuous logic |
| [11] | von Neumann | Reliable computation from unreliable components |
| [12] | Pippenger | Noise thresholds for formula computation |
| [15, 16] | CORDIC algorithms | Taylor series computation in hardware |
