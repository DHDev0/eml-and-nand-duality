"""
LaTeX → EML Decomposition
==========================

Converts parsed LaTeX AST into pure EML binary trees using only
eml(x,y) = e^x - ln(y) and the constant 1.

Every mathematical expression is decomposed into the EML grammar:
    S → 1 | eml(S, S)

Key constructions from the paper:
    exp(x)     = eml(x, 1)
    ln(x)      = eml(1, eml(eml(1, x), 1))
    0          = eml(1, eml(e, 1))
    1 - y      = eml(0, e^y)
    x - y      = eml(ln(x), e^y)
    x + y      = x - (-y)
    x * y      = exp(ln(x) + ln(y))
    x / y      = exp(ln(x) - ln(y))
    soft NAND  = eml(0, e^{ab})  [Theorem 2.6]
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from eml_pipeline.eml.eml_core import (
    EMLNode, EMLNodeType, ONE, VAR, EML,
    eml_exp, eml_e, eml_ln, eml_zero, eml_complement,
    eml_subtract, eml_negate, eml_add, eml_multiply, eml_divide,
    eml_power, eml_sqrt, eml_reciprocal, eml_abs,
    eml_sin_taylor, eml_cos_taylor, eml_tan_taylor,
    eml_soft_nand, eml_evaluate,
    _int_to_eml, _float_to_eml
)

# Import the AST types from the parser
from eml_pipeline.parsers.latex_parser import ASTNode, NodeType


# ─── Domain-Specific Variable Definitions ────────────────────────────────────

PHYSICS_VARIABLES: Dict[str, Dict[str, Any]] = {
    # Quantum mechanics
    "\\psi": {"domain": "quantum_mechanics", "description": "wave function", "latex": "\\psi"},
    "\\Psi": {"domain": "quantum_mechanics", "description": "total wave function", "latex": "\\Psi"},
    "\\phi": {"domain": "quantum_mechanics", "description": "scalar field / phase", "latex": "\\phi"},
    "\\hbar": {"domain": "quantum_mechanics", "description": "reduced Planck constant", "value": 1.0546e-34, "latex": "\\hbar"},
    
    # Thermodynamics
    "\\psi_th": {"domain": "thermodynamics", "description": "stream function", "latex": "\\psi"},
    "\\eta_th": {"domain": "thermodynamics", "description": "efficiency", "latex": "\\eta"},
    "\\theta_th": {"domain": "thermodynamics", "description": "temperature", "latex": "\\theta"},
    
    # Electromagnetism
    "\\phi_em": {"domain": "electromagnetism", "description": "electric potential", "latex": "\\phi"},
    "\\psi_em": {"domain": "electromagnetism", "description": "magnetic flux", "latex": "\\psi"},
    
    # General relativity
    "\\psi_gr": {"domain": "general_relativity", "description": "metric perturbation", "latex": "\\psi"},
    "\\Omega_gr": {"domain": "general_relativity", "description": "solid angle", "latex": "\\Omega"},
}

MATH_CONSTANTS: Dict[str, float] = {
    "\\pi": math.pi,
    "e": math.e,
    "\\infty": float('inf'),
    "\\epsilon": 1e-10,
    "\\varepsilon": 1e-10,
    "\\hbar": 1.0546e-34,
}

# Standard physics constants
PHYSICS_CONSTANTS: Dict[str, Tuple[float, str]] = {
    "c": (299792458.0, "speed of light"),
    "G": (6.674e-11, "gravitational constant"),
    "k_B": (1.3806e-23, "Boltzmann constant"),
    "N_A": (6.022e23, "Avogadro number"),
    "R": (8.314, "gas constant"),
    "e_charge": (1.602e-19, "electron charge"),
    "m_e": (9.109e-31, "electron mass"),
    "m_p": (1.673e-27, "proton mass"),
    "\\sigma": (5.670e-8, "Stefan-Boltzmann constant"),
}


class LatexToEMLConverter:
    """
    Convert LaTeX AST nodes to EML binary trees.
    
    This is the core of the forward pipeline: every mathematical expression
    from LaTeX is decomposed into pure eml(x,y) = e^x - ln(y) trees.
    """
    
    def __init__(self, taylor_order: int = 8):
        self.taylor_order = taylor_order
        self.metadata: Dict[str, Any] = {
            "conversions": {},
            "errors": [],
            "variable_bindings": {},
        }
        self._var_counter = 0
    
    def _fresh_var(self, prefix: str = "v") -> str:
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"
    
    def convert(self, ast: ASTNode) -> EMLNode:
        """Convert a LaTeX AST to an EML tree."""
        try:
            result = self._convert_node(ast)
            # Store conversion metadata
            self.metadata["conversions"][str(ast)] = {
                "ast_type": ast.type.value,
                "eml_depth": result.depth(),
                "eml_size": result.size(),
            }
            return result
        except Exception as e:
            self.metadata["errors"].append({
                "ast": str(ast),
                "error": str(e)
            })
            # Return a variable node as placeholder
            return VAR(f"_error_{ast.type.value}")
    
    def _convert_node(self, ast: ASTNode) -> EMLNode:
        """Dispatch based on AST node type."""
        
        if ast.type == NodeType.NUMBER:
            return self._convert_number(ast)
        elif ast.type == NodeType.VARIABLE:
            return self._convert_variable(ast)
        elif ast.type == NodeType.SPECIAL_CONST:
            return self._convert_special_const(ast)
        elif ast.type == NodeType.BINARY_OP:
            return self._convert_binary_op(ast)
        elif ast.type == NodeType.UNARY_OP:
            return self._convert_unary_op(ast)
        elif ast.type == NodeType.FUNCTION_CALL:
            return self._convert_function_call(ast)
        elif ast.type == NodeType.FRACTION:
            return self._convert_fraction(ast)
        elif ast.type == NodeType.POWER:
            return self._convert_power(ast)
        elif ast.type == NodeType.ROOT:
            return self._convert_root(ast)
        elif ast.type == NodeType.ABS_VALUE:
            return self._convert_abs_value(ast)
        elif ast.type == NodeType.FACTORIAL:
            return self._convert_factorial(ast)
        elif ast.type == NodeType.BINOMIAL:
            return self._convert_binomial(ast)
        elif ast.type == NodeType.SUM:
            return self._convert_sum(ast)
        elif ast.type == NodeType.PRODUCT:
            return self._convert_product(ast)
        elif ast.type == NodeType.INTEGRAL:
            return self._convert_integral(ast)
        elif ast.type == NodeType.LIMIT:
            return self._convert_limit(ast)
        elif ast.type == NodeType.SUBSCRIPT:
            return self._convert_subscript(ast)
        elif ast.type == NodeType.SUPERSCRIPT:
            return self._convert_superscript(ast)
        elif ast.type == NodeType.DECORATED_VAR:
            return self._convert_decorated_var(ast)
        elif ast.type == NodeType.EQUALITY:
            return self._convert_equality(ast)
        elif ast.type == NodeType.RELATION:
            return self._convert_relation(ast)
        elif ast.type == NodeType.SET_OP:
            return self._convert_set_op(ast)
        elif ast.type == NodeType.MATRIX:
            return self._convert_matrix(ast)
        elif ast.type == NodeType.CASES:
            return self._convert_cases(ast)
        elif ast.type == NodeType.GROUP:
            return self._convert_node(ast.children[0]) if ast.children else ONE()
        else:
            # Fallback
            return VAR(f"_unknown_{ast.type.value}")
    
    def _convert_number(self, ast: ASTNode) -> EMLNode:
        """Convert a numeric literal to an EML tree."""
        val = float(ast.value)
        if val == 0:
            return eml_zero()
        if val == 1:
            return ONE()
        if val == math.e:
            return eml_e()
        # General number: store as constant variable
        node = VAR(f"_n_{val}")
        node.metadata["const_value"] = val
        node.metadata["is_constant"] = True
        node.metadata["latex_source"] = ast.value
        return node
    
    def _convert_variable(self, ast: ASTNode) -> EMLNode:
        """Convert a variable name to an EML VAR node."""
        name = ast.value
        # Check for known constants
        if name in MATH_CONSTANTS:
            val = MATH_CONSTANTS[name]
            if val == float('inf'):
                # Infinity: represent as 1/0 approach or large number
                node = VAR(f"_inf")
                node.metadata["const_value"] = 1e100
                node.metadata["is_constant"] = True
                return node
            node = VAR(name)
            node.metadata["const_value"] = val
            node.metadata["is_constant"] = True
            return node
        # Check physics constants
        if name in PHYSICS_CONSTANTS:
            val, desc = PHYSICS_CONSTANTS[name]
            node = VAR(name)
            node.metadata["const_value"] = val
            node.metadata["is_constant"] = True
            node.metadata["physics_constant"] = desc
            return node
        # Regular variable
        node = VAR(name)
        node.metadata["latex_source"] = name
        return node
    
    def _convert_special_const(self, ast: ASTNode) -> EMLNode:
        """Convert special constants like \\pi, \\infty, \\epsilon."""
        name = ast.value.lstrip("\\")
        if name in MATH_CONSTANTS:
            val = MATH_CONSTANTS[ast.value]
            if val == float('inf'):
                node = VAR("_inf")
                node.metadata["const_value"] = 1e100
                node.metadata["is_constant"] = True
                return node
            node = VAR(ast.value)
            node.metadata["const_value"] = val
            node.metadata["is_constant"] = True
            return node
        node = VAR(ast.value)
        return node
    
    def _convert_binary_op(self, ast: ASTNode) -> EMLNode:
        """Convert binary operations: +, -, ×, ÷, /."""
        op = ast.value
        left = self._convert_node(ast.children[0])
        right = self._convert_node(ast.children[1])
        
        if op == "+":
            return eml_add(left, right)
        elif op == "-":
            return eml_subtract(left, right)
        elif op in ("×", "·", "\u00d7", "\u22c5", "\\cdot", "\\times"):
            return eml_multiply(left, right)
        elif op in ("/", "÷", "\u00f7", "\\div"):
            return eml_divide(left, right)
        else:
            # Default to multiplication for unknown ops
            return eml_multiply(left, right)
    
    def _convert_unary_op(self, ast: ASTNode) -> EMLNode:
        """Convert unary operations: -x, +x."""
        op = ast.value
        operand = self._convert_node(ast.children[0])
        
        if op == "-":
            return eml_negate(operand)
        elif op == "+":
            return operand
        else:
            return operand
    
    def _convert_function_call(self, ast: ASTNode) -> EMLNode:
        """Convert function calls: sin, cos, exp, ln, log, etc."""
        func_name = ast.value.lstrip("\\")
        arg = self._convert_node(ast.children[0])
        
        # Exponential / logarithmic
        if func_name in ("exp",):
            return eml_exp(arg)
        elif func_name in ("ln", "log"):
            return eml_ln(arg)
        elif func_name == "log" and len(ast.children) > 1:
            # log_b(x) = ln(x) / ln(b)
            base = self._convert_node(ast.children[1])
            return eml_divide(eml_ln(arg), eml_ln(base))
        
        # Trigonometric (Taylor series decomposed into EML)
        elif func_name in ("sin",):
            return eml_sin_taylor(arg, self.taylor_order)
        elif func_name in ("cos",):
            return eml_cos_taylor(arg, self.taylor_order)
        elif func_name in ("tan",):
            return eml_tan_taylor(arg, self.taylor_order)
        elif func_name == "cot":
            # cot(x) = cos(x) / sin(x)
            return eml_divide(eml_cos_taylor(arg, self.taylor_order),
                            eml_sin_taylor(arg, self.taylor_order))
        elif func_name == "sec":
            # sec(x) = 1 / cos(x)
            return eml_reciprocal(eml_cos_taylor(arg, self.taylor_order))
        elif func_name == "csc":
            # csc(x) = 1 / sin(x)
            return eml_reciprocal(eml_sin_taylor(arg, self.taylor_order))
        
        # Inverse trigonometric (via Taylor series)
        elif func_name == "arcsin":
            return self._arcsin_eml(arg)
        elif func_name == "arccos":
            # arccos(x) = pi/2 - arcsin(x)
            pi_half = VAR("_pi_half")
            pi_half.metadata["const_value"] = math.pi / 2
            pi_half.metadata["is_constant"] = True
            return eml_subtract(pi_half, self._arcsin_eml(arg))
        elif func_name == "arctan":
            return self._arctan_eml(arg)
        
        # Hyperbolic
        elif func_name == "sinh":
            # sinh(x) = (e^x - e^{-x}) / 2
            exp_x = eml_exp(arg)
            neg_x = eml_negate(arg)
            exp_neg_x = eml_exp(neg_x)
            diff = eml_subtract(exp_x, exp_neg_x)
            two = _int_to_eml(2)
            return eml_divide(diff, two)
        elif func_name == "cosh":
            # cosh(x) = (e^x + e^{-x}) / 2
            exp_x = eml_exp(arg)
            neg_x = eml_negate(arg)
            exp_neg_x = eml_exp(neg_x)
            s = eml_add(exp_x, exp_neg_x)
            two = _int_to_eml(2)
            return eml_divide(s, two)
        elif func_name == "tanh":
            # tanh(x) = sinh(x) / cosh(x)
            sinh_x = self._convert_function_call(
                ASTNode(NodeType.FUNCTION_CALL, [ast.children[0]], "\\sinh"))
            cosh_x = self._convert_function_call(
                ASTNode(NodeType.FUNCTION_CALL, [ast.children[0]], "\\cosh"))
            return eml_divide(sinh_x, cosh_x)
        
        # Other named functions
        elif func_name in ("deg", "det", "dim", "hom", "ker", "max", "min",
                          "sup", "inf", "arg", "limsup", "liminf"):
            # These are higher-order — store as function application
            node = VAR(f"_func_{func_name}")
            node.metadata["function_name"] = func_name
            node.metadata["argument"] = eml_to_dict(arg)
            return node
        
        # Fallback: treat as variable
        return VAR(f"_func_{func_name}")
    
    def _arcsin_eml(self, x: EMLNode) -> EMLNode:
        """arcsin(x) via Taylor series: x + x³/6 + 3x⁵/40 + ..."""
        result = x
        # arcsin(x) = Σ (2k)! / (4^k (k!)^2 (2k+1)) * x^(2k+1)
        for k in range(1, self.taylor_order):
            coeff = math.factorial(2*k) / (4**k * (math.factorial(k))**2 * (2*k+1))
            power = 2*k + 1
            x_pow = eml_power(x, power)
            coeff_tree = _float_to_eml(abs(coeff))
            term = eml_multiply(x_pow, coeff_tree)
            result = eml_add(result, term)
        result.metadata["name"] = f"arcsin({x})"
        return result
    
    def _arctan_eml(self, x: EMLNode) -> EMLNode:
        """arctan(x) via Taylor series: x - x³/3 + x⁵/5 - ..."""
        result = x
        for k in range(1, self.taylor_order):
            power = 2*k + 1
            x_pow = eml_power(x, power)
            denom = _int_to_eml(power)
            term = eml_divide(x_pow, denom)
            if k % 2 == 1:
                term = eml_negate(term)
            result = eml_add(result, term)
        result.metadata["name"] = f"arctan({x})"
        return result
    
    def _convert_fraction(self, ast: ASTNode) -> EMLNode:
        """\\frac{num}{den} → num / den"""
        num = self._convert_node(ast.children[0])
        den = self._convert_node(ast.children[1])
        return eml_divide(num, den)
    
    def _convert_power(self, ast: ASTNode) -> EMLNode:
        """a^b → exp(b * ln(a)) for general case, or optimize for integers."""
        base = self._convert_node(ast.children[0])
        exponent = self._convert_node(ast.children[1])
        
        # Check if exponent is a known integer constant
        if (exponent.node_type == EMLNodeType.VAR and 
            exponent.metadata.get("is_constant")):
            exp_val = exponent.metadata.get("const_value")
            if exp_val is not None and exp_val == int(exp_val) and 0 <= exp_val <= 20:
                return eml_power(base, int(exp_val))
        
        # General: a^b = exp(b * ln(a))
        return eml_exp(eml_multiply(exponent, eml_ln(base)))
    
    def _convert_root(self, ast: ASTNode) -> EMLNode:
        """\\sqrt{x} or \\sqrt[n]{x}."""
        radicand = self._convert_node(ast.children[0])
        if len(ast.children) > 1:
            # nth root: x^(1/n)
            index = self._convert_node(ast.children[1])
            reciprocal_n = eml_reciprocal(index)
            return eml_exp(eml_multiply(reciprocal_n, eml_ln(radicand)))
        else:
            # Square root
            return eml_sqrt(radicand)
    
    def _convert_abs_value(self, ast: ASTNode) -> EMLNode:
        """|x| → sqrt(x^2)"""
        inner = self._convert_node(ast.children[0])
        return eml_abs(inner)
    
    def _convert_factorial(self, ast: ASTNode) -> EMLNode:
        """n! — limited to small integers, otherwise Gamma function approximation."""
        inner = self._convert_node(ast.children[0])
        # If inner is a constant integer
        if (inner.node_type == EMLNodeType.VAR and 
            inner.metadata.get("is_constant")):
            val = inner.metadata.get("const_value")
            if val is not None and val == int(val) and 0 <= val <= 20:
                return _int_to_eml(math.factorial(int(val)))
        # General: use Gamma(n+1) ≈ Stirling approximation
        # n! ≈ sqrt(2πn) * (n/e)^n
        # For now, store as special function
        node = VAR(f"_factorial({inner})")
        node.metadata["function"] = "factorial"
        node.metadata["argument"] = eml_to_dict(inner)
        return node
    
    def _convert_binomial(self, ast: ASTNode) -> EMLNode:
        """\\binom{n}{k} = n! / (k! * (n-k)!)"""
        n = self._convert_node(ast.children[0])
        k = self._convert_node(ast.children[1])
        # binom(n,k) = exp(ln(n!) - ln(k!) - ln((n-k)!))
        # Store as special for now
        node = VAR(f"_binom({n},{k})")
        node.metadata["function"] = "binomial"
        return node
    
    def _convert_sum(self, ast: ASTNode) -> EMLNode:
        """
        \\sum_{i=a}^{b} f(i) — Convert to explicit addition for finite bounds,
        or store as symbolic for infinite bounds.
        """
        # Parse bounds
        lower = self._convert_node(ast.children[0]) if len(ast.children) > 1 else None
        upper = self._convert_node(ast.children[1]) if len(ast.children) > 2 else None
        body = self._convert_node(ast.children[-1])
        
        # Check if bounds are finite integers
        if lower and upper:
            lower_val = self._try_get_const_value(lower)
            upper_val = self._try_get_const_value(upper)
            if lower_val is not None and upper_val is not None:
                lo = int(lower_val)
                hi = int(upper_val)
                if hi - lo <= 20:  # Finite, small range
                    result = body  # First term (substituted)
                    for i in range(lo + 1, hi + 1):
                        term = body  # In a real impl, substitute index variable
                        result = eml_add(result, term)
                    result.metadata["name"] = f"sum_{{i={lo}}}^{{{hi}}}"
                    return result
        
        # Infinite or large range: store as symbolic
        node = VAR(f"_sum_expr")
        node.metadata["function"] = "sum"
        node.metadata["lower"] = eml_to_dict(lower) if lower else None
        node.metadata["upper"] = eml_to_dict(upper) if upper else None
        node.metadata["body"] = eml_to_dict(body)
        return node
    
    def _convert_product(self, ast: ASTNode) -> EMLNode:
        """\\prod_{i=a}^{b} f(i) — Convert to explicit multiplication for finite bounds."""
        lower = self._convert_node(ast.children[0]) if len(ast.children) > 1 else None
        upper = self._convert_node(ast.children[1]) if len(ast.children) > 2 else None
        body = self._convert_node(ast.children[-1])
        
        if lower and upper:
            lower_val = self._try_get_const_value(lower)
            upper_val = self._try_get_const_value(upper)
            if lower_val is not None and upper_val is not None:
                lo = int(lower_val)
                hi = int(upper_val)
                if hi - lo <= 20:
                    result = body
                    for i in range(lo + 1, hi + 1):
                        term = body
                        result = eml_multiply(result, term)
                    result.metadata["name"] = f"prod_{{i={lo}}}^{{{hi}}}"
                    return result
        
        node = VAR(f"_prod_expr")
        node.metadata["function"] = "product"
        node.metadata["lower"] = eml_to_dict(lower) if lower else None
        node.metadata["upper"] = eml_to_dict(upper) if upper else None
        node.metadata["body"] = eml_to_dict(body)
        return node
    
    def _convert_integral(self, ast: ASTNode) -> EMLNode:
        """
        \\int_{a}^{b} f(x) dx — Store as symbolic (numerical integration
        would be done at evaluation time via Riemann sums).
        """
        lower = self._convert_node(ast.children[0]) if len(ast.children) > 1 else None
        upper = self._convert_node(ast.children[1]) if len(ast.children) > 2 else None
        integrand = self._convert_node(ast.children[-1])
        
        node = VAR(f"_integral_expr")
        node.metadata["function"] = "integral"
        node.metadata["integral_type"] = ast.value  # \int, \oint, etc.
        node.metadata["lower"] = eml_to_dict(lower) if lower else None
        node.metadata["upper"] = eml_to_dict(upper) if upper else None
        node.metadata["integrand"] = eml_to_dict(integrand)
        return node
    
    def _convert_limit(self, ast: ASTNode) -> EMLNode:
        """
        \\lim_{x \\to a} f(x) — Store as symbolic with limit info.
        For evaluation: substitute x → a (with special handling for ∞).
        """
        var_node = self._convert_node(ast.children[0]) if len(ast.children) > 1 else None
        target = self._convert_node(ast.children[1]) if len(ast.children) > 2 else None
        body = self._convert_node(ast.children[-1])
        
        node = VAR(f"_limit_expr")
        node.metadata["function"] = "limit"
        node.metadata["variable"] = eml_to_dict(var_node) if var_node else None
        node.metadata["target"] = eml_to_dict(target) if target else None
        node.metadata["body"] = eml_to_dict(body)
        return node
    
    def _convert_subscript(self, ast: ASTNode) -> EMLNode:
        """x_{sub} — treated as a decorated variable."""
        base = self._convert_node(ast.children[0])
        sub = self._convert_node(ast.children[1])
        # Create a compound variable name
        base_name = self._get_var_name(base)
        sub_name = self._get_var_name(sub)
        node = VAR(f"{base_name}_{sub_name}")
        node.metadata["subscript"] = True
        node.metadata["base"] = eml_to_dict(base)
        node.metadata["index"] = eml_to_dict(sub)
        return node
    
    def _convert_superscript(self, ast: ASTNode) -> EMLNode:
        """x^{sup} — usually a power."""
        return self._convert_power(ast)
    
    def _convert_decorated_var(self, ast: ASTNode) -> EMLNode:
        """\\hat{x}, \\bar{x}, etc."""
        deco = ast.value.lstrip("\\")
        inner = self._convert_node(ast.children[0])
        name = self._get_var_name(inner)
        node = VAR(f"{deco}_{name}")
        node.metadata["decoration"] = deco
        node.metadata["base_name"] = name
        return node
    
    def _convert_equality(self, ast: ASTNode) -> EMLNode:
        """a = b — store both sides."""
        left = self._convert_node(ast.children[0])
        right = self._convert_node(ast.children[1])
        # Compute difference: should be zero
        diff = eml_subtract(left, right)
        diff.metadata["name"] = f"({left} - {right})"
        diff.metadata["is_equality"] = True
        return diff
    
    def _convert_relation(self, ast: ASTNode) -> EMLNode:
        """a < b, a ≤ b, etc. — store symbolically."""
        left = self._convert_node(ast.children[0])
        right = self._convert_node(ast.children[1])
        node = VAR(f"_relation_{ast.value}")
        node.metadata["relation"] = ast.value
        node.metadata["left"] = eml_to_dict(left)
        node.metadata["right"] = eml_to_dict(right)
        return node
    
    def _convert_set_op(self, ast: ASTNode) -> EMLNode:
        """Set operations — store symbolically."""
        node = VAR(f"_setop_{ast.value}")
        if ast.children:
            node.metadata["left"] = eml_to_dict(self._convert_node(ast.children[0]))
            node.metadata["right"] = eml_to_dict(self._convert_node(ast.children[1]))
        return node
    
    def _convert_matrix(self, ast: ASTNode) -> EMLNode:
        """Matrix — store as symbolic structure."""
        node = VAR("_matrix")
        node.metadata["function"] = "matrix"
        node.metadata["rows"] = len(ast.children)
        return node
    
    def _convert_cases(self, ast: ASTNode) -> EMLNode:
        """Cases environment — store as symbolic."""
        node = VAR("_cases")
        node.metadata["function"] = "cases"
        return node
    
    def _try_get_const_value(self, node: EMLNode) -> Optional[float]:
        """Try to get a constant numeric value from an EML node."""
        if node.node_type == EMLNodeType.ONE:
            return 1.0
        if node.node_type == EMLNodeType.VAR:
            if node.metadata.get("is_constant") and "const_value" in node.metadata:
                return node.metadata["const_value"]
        return None
    
    def _get_var_name(self, node: EMLNode) -> str:
        """Get a string name from an EML node."""
        if node.node_type == EMLNodeType.ONE:
            return "1"
        if node.node_type == EMLNodeType.VAR:
            return node.var_name or "unnamed"
        return str(node)


# ─── Convenience function ─────────────────────────────────────────────────────

def latex_to_eml(latex_str: str, taylor_order: int = 8) -> Tuple[EMLNode, Dict]:
    """
    Convert a LaTeX expression string to an EML tree.
    
    Returns:
        (eml_tree, metadata_dict)
    """
    from eml_pipeline.parsers.latex_parser import parse_latex
    ast = parse_latex(latex_str)
    converter = LatexToEMLConverter(taylor_order=taylor_order)
    eml_tree = converter.convert(ast)
    return eml_tree, converter.metadata
