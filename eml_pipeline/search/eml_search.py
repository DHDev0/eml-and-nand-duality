"""
EML Search Module — Find Operations and Patterns
==================================================

Search for known mathematical operations, decompose them into EML trees,
and maintain a searchable catalog of all supported LaTeX operations.

This module provides:
1. Search for EML decompositions of mathematical operations
2. Catalog of all supported LaTeX → EML mappings
3. Variable lookup across domains (physics, math, engineering)
4. Pattern matching for EML tree optimization
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.eml.eml_core import (
    EMLNode, EMLNodeType, ONE, VAR, EML,
    eml_exp, eml_ln, eml_zero, eml_complement,
    eml_add, eml_subtract, eml_multiply, eml_divide,
    eml_negate, eml_power, eml_sqrt, eml_reciprocal,
    eml_sin_taylor, eml_cos_taylor, eml_tan_taylor,
    eml_soft_nand, eml_evaluate, eml_to_dict
)


# ─── Operation Catalog ────────────────────────────────────────────────────────

OPERATION_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Basic Arithmetic ──────────────────────────────────
    "addition": {
        "latex_patterns": ["a + b", "a+b"],
        "eml_construction": "x + y = x - (-y) = eml(ln(x), e^{-y})",
        "paper_reference": "Lemma 2.4",
        "category": "arithmetic",
        "depth": "O(depth_subtract + depth_negate)",
    },
    "subtraction": {
        "latex_patterns": ["a - b", "a-b", "a \\minus b"],
        "eml_construction": "x - y = eml(ln(x), e^y)",
        "paper_reference": "Lemma 2.4",
        "category": "arithmetic",
        "depth": "O(depth_ln + depth_exp + 1)",
    },
    "multiplication": {
        "latex_patterns": ["a \\times b", "a \\cdot b", "ab", "a * b"],
        "eml_construction": "x * y = exp(ln(x) + ln(y))",
        "paper_reference": "Lemma 2.4",
        "category": "arithmetic",
        "depth": "O(depth_exp + depth_ln + depth_add + 1)",
    },
    "division": {
        "latex_patterns": ["\\frac{a}{b}", "a / b", "a \\div b"],
        "eml_construction": "x / y = exp(ln(x) - ln(y))",
        "paper_reference": "Lemma 2.4",
        "category": "arithmetic",
        "depth": "O(depth_exp + depth_ln + depth_sub + 1)",
    },
    "negation": {
        "latex_patterns": ["-a"],
        "eml_construction": "-y = (1-y) - 1 = eml(ln(1-y), e)",
        "paper_reference": "Lemma 2.3 + 2.4",
        "category": "arithmetic",
        "depth": "O(depth_complement + depth_sub + 1)",
    },
    "power": {
        "latex_patterns": ["a^{b}", "a^b", "a^{n}"],
        "eml_construction": "x^n = exp(n * ln(x)) [n integer] or exp(b * ln(x))",
        "paper_reference": "Lemma 2.4",
        "category": "arithmetic",
        "depth": "O(n * depth_multiply) [integer n]",
    },
    "sqrt": {
        "latex_patterns": ["\\sqrt{a}", "a^{1/2}"],
        "eml_construction": "sqrt(x) = exp(ln(x)/2)",
        "paper_reference": "Lemma 2.4",
        "category": "arithmetic",
        "depth": "O(depth_exp + depth_ln + depth_div + 1)",
    },
    
    # ── Exponential / Logarithmic ──────────────────────────
    "exp": {
        "latex_patterns": ["e^{x}", "\\exp(x)", "e^x"],
        "eml_construction": "exp(x) = eml(x, 1)",
        "paper_reference": "Lemma 2.3",
        "category": "exp_log",
        "depth": 1,
    },
    "ln": {
        "latex_patterns": ["\\ln(x)", "\\log(x)", "\\log_{e}(x)"],
        "eml_construction": "ln(x) = eml(1, eml(eml(1, x), 1))",
        "paper_reference": "Lemma 2.3",
        "category": "exp_log",
        "depth": 3,
    },
    "log_base": {
        "latex_patterns": ["\\log_{b}(x)"],
        "eml_construction": "log_b(x) = ln(x) / ln(b)",
        "paper_reference": "Lemma 2.4",
        "category": "exp_log",
        "depth": "O(2*depth_ln + depth_div)",
    },
    
    # ── Trigonometric ──────────────────────────────────────
    "sin": {
        "latex_patterns": ["\\sin(x)", "\\sin x"],
        "eml_construction": "sin(x) = Taylor series in EML",
        "paper_reference": "Extended construction",
        "category": "trig",
        "depth": "O(order * depth_arithmetic)",
    },
    "cos": {
        "latex_patterns": ["\\cos(x)", "\\cos x"],
        "eml_construction": "cos(x) = Taylor series in EML",
        "paper_reference": "Extended construction",
        "category": "trig",
        "depth": "O(order * depth_arithmetic)",
    },
    "tan": {
        "latex_patterns": ["\\tan(x)", "\\tan x"],
        "eml_construction": "tan(x) = sin(x) / cos(x)",
        "paper_reference": "Extended construction",
        "category": "trig",
        "depth": "O(depth_sin + depth_cos + depth_div)",
    },
    "arcsin": {
        "latex_patterns": ["\\arcsin(x)"],
        "eml_construction": "arcsin(x) = Taylor series in EML",
        "paper_reference": "Extended construction",
        "category": "trig",
        "depth": "O(order * depth_arithmetic)",
    },
    "arccos": {
        "latex_patterns": ["\\arccos(x)"],
        "eml_construction": "arccos(x) = pi/2 - arcsin(x)",
        "paper_reference": "Extended construction",
        "category": "trig",
        "depth": "O(depth_arcsin + depth_sub)",
    },
    "arctan": {
        "latex_patterns": ["\\arctan(x)"],
        "eml_construction": "arctan(x) = Taylor series in EML",
        "paper_reference": "Extended construction",
        "category": "trig",
        "depth": "O(order * depth_arithmetic)",
    },
    
    # ── Hyperbolic ────────────────────────────────────────
    "sinh": {
        "latex_patterns": ["\\sinh(x)"],
        "eml_construction": "sinh(x) = (e^x - e^{-x}) / 2",
        "paper_reference": "Extended construction",
        "category": "hyperbolic",
    },
    "cosh": {
        "latex_patterns": ["\\cosh(x)"],
        "eml_construction": "cosh(x) = (e^x + e^{-x}) / 2",
        "paper_reference": "Extended construction",
        "category": "hyperbolic",
    },
    "tanh": {
        "latex_patterns": ["\\tanh(x)"],
        "eml_construction": "tanh(x) = sinh(x) / cosh(x)",
        "paper_reference": "Extended construction",
        "category": "hyperbolic",
    },
    
    # ── NAND / Boolean ────────────────────────────────────
    "soft_nand": {
        "latex_patterns": ["NAND(a,b)", "1-ab"],
        "eml_construction": "1-ab = eml(0, e^{ab}) [Theorem 2.6]",
        "paper_reference": "Theorem 2.6",
        "category": "nand_bridge",
        "depth": "~12 levels",
    },
    "boltzmann_nand": {
        "latex_patterns": ["NAND_T(a,b)"],
        "eml_construction": "e^{-ab/T} [Remark 2.6a]",
        "paper_reference": "Remark 2.6a",
        "category": "nand_bridge",
        "depth": "~15-18 levels",
    },
    
    # ── Calculus ──────────────────────────────────────────
    "integral": {
        "latex_patterns": ["\\int f(x) dx", "\\int_{a}^{b} f(x) dx"],
        "eml_construction": "Riemann sum → finite addition",
        "paper_reference": "Symbolic (evaluated numerically)",
        "category": "calculus",
    },
    "sum": {
        "latex_patterns": ["\\sum_{i=1}^{n} f(i)"],
        "eml_construction": "Finite addition of EML terms",
        "paper_reference": "Extended construction",
        "category": "calculus",
    },
    "product": {
        "latex_patterns": ["\\prod_{i=1}^{n} f(i)"],
        "eml_construction": "Finite multiplication of EML terms",
        "paper_reference": "Extended construction",
        "category": "calculus",
    },
    "limit": {
        "latex_patterns": ["\\lim_{x \\to a} f(x)"],
        "eml_construction": "Substitution with ε-approximation",
        "paper_reference": "Symbolic (evaluated numerically)",
        "category": "calculus",
    },
    
    # ── Constants ─────────────────────────────────────────
    "zero": {
        "latex_patterns": ["0"],
        "eml_construction": "0 = eml(1, eml(e, 1))",
        "paper_reference": "Lemma 2.3",
        "category": "constants",
        "depth": 3,
    },
    "e_const": {
        "latex_patterns": ["e"],
        "eml_construction": "e = eml(1, 1)",
        "paper_reference": "Lemma 2.3",
        "category": "constants",
        "depth": 1,
    },
    "pi": {
        "latex_patterns": ["\\pi"],
        "eml_construction": "pi = 4*arctan(1) via Taylor series",
        "paper_reference": "Extended construction",
        "category": "constants",
    },
}


def search_operation(query: str) -> List[Dict[str, Any]]:
    """Search the operation catalog for matching operations."""
    query_lower = query.lower().strip()
    results = []
    
    for name, info in OPERATION_CATALOG.items():
        score = 0
        # Check name match
        if query_lower in name:
            score += 10
        # Check latex patterns
        for pattern in info.get("latex_patterns", []):
            if query_lower in pattern.lower():
                score += 5
        # Check category
        if query_lower in info.get("category", ""):
            score += 3
        # Check paper reference
        if query_lower in info.get("paper_reference", "").lower():
            score += 2
        
        if score > 0:
            results.append({"name": name, "score": score, **info})
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_all_categories() -> Dict[str, List[str]]:
    """Get all operations grouped by category."""
    categories: Dict[str, List[str]] = {}
    for name, info in OPERATION_CATALOG.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
    return categories


def search_variable(name: str, domain: str = None) -> List[Dict[str, Any]]:
    """Search for a variable definition across domains."""
    from eml_pipeline.eml.latex_to_eml import PHYSICS_VARIABLES, PHYSICS_CONSTANTS, MATH_CONSTANTS
    
    results = []
    
    # Search math constants
    for k, v in MATH_CONSTANTS.items():
        if name in k or k in name:
            results.append({"name": k, "value": v, "domain": "math"})
    
    # Search physics constants
    for k, (v, desc) in PHYSICS_CONSTANTS.items():
        if name in k or k in name:
            results.append({"name": k, "value": v, "domain": "physics", "description": desc})
    
    # Search physics variables
    for k, info in PHYSICS_VARIABLES.items():
        if name in k or name in info.get("description", ""):
            results.append({"name": k, **info})
    
    # Filter by domain if specified
    if domain:
        results = [r for r in results if r.get("domain", "").startswith(domain)]
    
    return results
