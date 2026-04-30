"""
latex_parser.py — Comprehensive LaTeX math parser.

Tokenises a LaTeX math string, builds an AST (Abstract Syntax Tree), and
exposes a single public function :func:`parse_latex` that returns the root
:class:`ASTNode`.

Supported constructs
---------------------
- Basic arithmetic, fractions, powers, roots
- Exponential / logarithmic / trigonometric / hyperbolic functions
- Limits, summations, products, integrals
- Special constants, Greek letters
- Parentheses / brackets / absolute values
- Factorials, binomial coefficients
- Set notation operators
- Custom operators (``\\operatorname``, ``\\mathrm``)
- Decorated variables (hat, bar, vec, tilde, dot)
- Matrices (pmatrix, bmatrix, vmatrix) and cases environments
- Both display (``$$…$$``) and inline (``$…$``) math delimiters
"""

from __future__ import annotations

import re
import enum
from dataclasses import dataclass, field
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# 1. Token types
# ---------------------------------------------------------------------------

class TokenType(enum.Enum):
    """Enumeration of every token category the lexer can emit."""

    # Literal values
    NUMBER = "NUMBER"          # 3, 3.14, .5
    VARIABLE = "VARIABLE"      # single letter like a, x, \alpha

    # Operators
    PLUS = "PLUS"              # +
    MINUS = "MINUS"            # -
    TIMES = "TIMES"            # \times
    CDOT = "CDOT"              # \cdot
    DIV = "DIV"                # \div
    SLASH = "SLASH"            # /

    # Comparison / relation (kept as tokens even if not fully parsed)
    EQUALS = "EQUALS"          # =
    LT = "LT"                  # <
    GT = "GT"                  # >
    LE = "LE"                  # \leq
    GE = "GE"                  # \geq
    NEQ = "NEQ"                # \neq
    APPROX = "APPROX"          # \approx

    # Grouping
    LBRACE = "LBRACE"          # {
    RBRACE = "RBRACE"          # }
    LPAREN = "LPAREN"          # (
    RPAREN = "RPAREN"          # )
    LBRACKET = "LBRACKET"      # [
    RBRACKET = "RBRACKET"      # ]
    PIPE = "PIPE"              # |
    LANGLE = "LANGLE"          # \langle
    RANGLE = "RANGLE"          # \rangle

    # Structural
    CARET = "CARET"            # ^
    UNDERSCORE = "UNDERSCORE"  # _
    AMPERSAND = "AMPERSAND"    # &  (matrix column separator)
    COMMA = "COMMA"            # ,
    BANG = "BANG"              # !
    DOT = "DOT"                # .

    # Commands (the value field carries the command name without the backslash)
    COMMAND = "COMMAND"        # \frac, \sqrt, \sum, …

    # Special
    EOF = "EOF"


# ---------------------------------------------------------------------------
# 2. AST node types
# ---------------------------------------------------------------------------

class NodeType(enum.Enum):
    """Types of nodes that can appear in the AST."""

    NUMBER = "NUMBER"
    VARIABLE = "VARIABLE"
    BINARY_OP = "BINARY_OP"
    UNARY_OP = "UNARY_OP"
    FUNCTION_CALL = "FUNCTION_CALL"
    FRACTION = "FRACTION"
    POWER = "POWER"
    ROOT = "ROOT"
    SUM = "SUM"
    PRODUCT = "PRODUCT"
    INTEGRAL = "INTEGRAL"
    LIMIT = "LIMIT"
    ABS_VALUE = "ABS_VALUE"
    FACTORIAL = "FACTORIAL"
    BINOMIAL = "BINOMIAL"
    SPECIAL_CONST = "SPECIAL_CONST"
    DECORATED_VAR = "DECORATED_VAR"
    MATRIX = "MATRIX"
    CASES = "CASES"
    SUBSCRIPT = "SUBSCRIPT"
    SUPERSCRIPT = "SUPERSCRIPT"
    EQUALITY = "EQUALITY"
    SET_OP = "SET_OP"
    RELATION = "RELATION"
    GROUP = "GROUP"


# ---------------------------------------------------------------------------
# 3. AST node
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """A single node in the abstract syntax tree.

    Attributes
    ----------
    type : NodeType
        The category of this node.
    children : list[ASTNode]
        Child nodes (empty for leaves like NUMBER / VARIABLE).
    value : str | None
        For leaves — the literal text (e.g. ``"3.14"``, ``"x"``, ``"\\pi"``).
        For operators — the operator symbol (e.g. ``"+"``, ``"\\times"``).
    """

    type: NodeType
    children: List[ASTNode] = field(default_factory=list)
    value: Optional[str] = None

    # Convenience helpers ---------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        if self.value is not None:
            return f"ASTNode({self.type.value}, {self.value!r}, {len(self.children)} children)"
        return f"ASTNode({self.type.value}, {len(self.children)} children)"

    def to_dict(self) -> dict:
        """Serialise the AST to a nested dict (useful for debugging / JSON)."""
        d: dict = {"type": self.type.value}
        if self.value is not None:
            d["value"] = self.value
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# ---------------------------------------------------------------------------
# 4. Token dataclass
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """A single lexeme produced by the tokenizer."""

    type: TokenType
    value: str
    pos: int = 0  # character offset in the original input (for errors)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Token({self.type.value}, {self.value!r})"


# ---------------------------------------------------------------------------
# 5. Known LaTeX commands (categorised for the parser)
# ---------------------------------------------------------------------------

BINARY_OPS: dict[str, str] = {
    "times": "\u00d7",
    "cdot": "\u22c5",
    "div": "\u00f7",
    "pm": "\u00b1",
    "mp": "\u2213",
}

RELATION_OPS: dict[str, str] = {
    "leq": "\u2264",
    "geq": "\u2265",
    "neq": "\u2260",
    "approx": "\u2248",
    "equiv": "\u2261",
    "sim": "\u223c",
    "propto": "\u221d",
    "ll": "\u226a",
    "gg": "\u226b",
}

SET_OPS: dict[str, str] = {
    "in": "\u2208",
    "notin": "\u2209",
    "subset": "\u2282",
    "supset": "\u2283",
    "subseteq": "\u2286",
    "supseteq": "\u2287",
    "cup": "\u222a",
    "cap": "\u2229",
    "setminus": "\u2216",
    "emptyset": "\u2205",
    "varnothing": "\u2205",
}

ARROW_OPS: dict[str, str] = {
    "to": "\u2192",
    "rightarrow": "\u2192",
    "leftarrow": "\u2190",
    "Rightarrow": "\u21d2",
    "Leftarrow": "\u21d0",
    "mapsto": "\u21a6",
}

SPECIAL_CONSTANTS: dict[str, str] = {
    "pi": "\u03c0",
    "infty": "\u221e",
    "epsilon": "\u03b5",
    "varepsilon": "\u03f5",
    "hbar": "\u210f",
}

# Lower-case Greek letters that behave as variables.
GREEK_LETTERS: dict[str, str] = {
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "epsilon": "\u03b5",
    "varepsilon": "\u03f5",
    "zeta": "\u03b6",
    "eta": "\u03b7",
    "theta": "\u03b8",
    "vartheta": "\u03d1",
    "iota": "\u03b9",
    "kappa": "\u03ba",
    "lambda": "\u03bb",
    "mu": "\u03bc",
    "nu": "\u03bd",
    "xi": "\u03be",
    "omicron": "\u03bf",
    "pi": "\u03c0",
    "rho": "\u03c1",
    "varrho": "\u03f1",
    "sigma": "\u03c3",
    "varsigma": "\u03c2",
    "tau": "\u03c4",
    "upsilon": "\u03c5",
    "phi": "\u03c6",
    "varphi": "\u03d5",
    "chi": "\u03c7",
    "psi": "\u03c8",
    "omega": "\u03c9",
    # Uppercase
    "Gamma": "\u0393",
    "Delta": "\u0394",
    "Theta": "\u0398",
    "Lambda": "\u039b",
    "Xi": "\u039e",
    "Pi": "\u03a0",
    "Sigma": "\u03a3",
    "Upsilon": "\u03a5",
    "Phi": "\u03a6",
    "Psi": "\u03a8",
    "Omega": "\u03a9",
}

# Functions that take a single argument (possibly in parentheses).
NAMED_FUNCTIONS: set[str] = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh",
    "exp", "ln", "log",
    "deg", "det", "dim", "hom", "ker", "max", "min",
    "sup", "inf", "arg", "limsup", "liminf",
}

# Decorations that sit above a variable.
DECORATIONS: dict[str, str] = {
    "hat": "\u0302",
    "bar": "\u0304",
    "vec": "\u20d7",
    "tilde": "\u0303",
    "dot": "\u0307",
    "ddot": "\u0308",
    "acute": "\u0301",
    "grave": "\u0300",
    "breve": "\u0306",
    "check": "\u030c",
}


# ---------------------------------------------------------------------------
# 6. Tokenizer (Lexer)
# ---------------------------------------------------------------------------

class LaTeXTokenizer:
    """Convert a raw LaTeX math string into a flat list of :class:`Token` objects."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []

    # -- public API --------------------------------------------------------

    def tokenize(self) -> List[Token]:
        """Return the full token list (including a trailing EOF)."""
        while self.pos < len(self.text):
            ch = self.text[self.pos]

            if ch in " \t\n\r":
                self.pos += 1
                continue

            if ch == "\\":
                self._read_command()
            elif ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                self._read_number()
            elif ch.isalpha():
                self._emit(TokenType.VARIABLE, ch)
            elif ch == "+":
                self._emit(TokenType.PLUS, ch)
            elif ch == "-":
                self._emit(TokenType.MINUS, ch)
            elif ch == "/":
                self._emit(TokenType.SLASH, ch)
            elif ch == "=":
                self._emit(TokenType.EQUALS, ch)
            elif ch == "<":
                self._emit(TokenType.LT, ch)
            elif ch == ">":
                self._emit(TokenType.GT, ch)
            elif ch == "^":
                self._emit(TokenType.CARET, ch)
            elif ch == "_":
                self._emit(TokenType.UNDERSCORE, ch)
            elif ch == "{":
                self._emit(TokenType.LBRACE, ch)
            elif ch == "}":
                self._emit(TokenType.RBRACE, ch)
            elif ch == "(":
                self._emit(TokenType.LPAREN, ch)
            elif ch == ")":
                self._emit(TokenType.RPAREN, ch)
            elif ch == "[":
                self._emit(TokenType.LBRACKET, ch)
            elif ch == "]":
                self._emit(TokenType.RBRACKET, ch)
            elif ch == "|":
                self._emit(TokenType.PIPE, ch)
            elif ch == "!":
                self._emit(TokenType.BANG, ch)
            elif ch == "&":
                self._emit(TokenType.AMPERSAND, ch)
            elif ch == ",":
                self._emit(TokenType.COMMA, ch)
            elif ch == ".":
                self._emit(TokenType.DOT, ch)
            elif ch == "*":
                # Some authors use * for multiplication
                self._emit(TokenType.CDOT, ch)
            else:
                # Skip unknown character silently
                self.pos += 1

        self.tokens.append(Token(TokenType.EOF, "", len(self.text)))
        return self.tokens

    # -- internals ---------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.text[idx] if idx < len(self.text) else ""

    def _emit(self, ttype: TokenType, value: str) -> None:
        self.tokens.append(Token(ttype, value, self.pos))
        self.pos += len(value)

    def _read_number(self) -> None:
        start = self.pos
        has_dot = False
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch.isdigit():
                self.pos += 1
            elif ch == "." and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break
        self.tokens.append(Token(TokenType.NUMBER, self.text[start:self.pos], start))

    def _read_command(self) -> None:
        """Read a backslash-command like ``\\frac`` or ``\\alpha``."""
        start = self.pos
        self.pos += 1  # skip '\'

        # Escaped single special character:  \{  \}  \|  \  etc.
        if self.pos < len(self.text) and not self.text[self.pos].isalpha():
            ch = self.text[self.pos]
            self.pos += 1
            cmd = self.text[start:self.pos]
            if ch == "{":
                self.tokens.append(Token(TokenType.LBRACE, cmd, start))
            elif ch == "}":
                self.tokens.append(Token(TokenType.RBRACE, cmd, start))
            elif ch == "|":
                self.tokens.append(Token(TokenType.PIPE, cmd, start))
            elif ch == " ":
                pass  # escaped space — ignore
            else:
                self.tokens.append(Token(TokenType.COMMAND, cmd, start))
            return

        # Alphabetic command name
        cmd_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalpha():
            self.pos += 1
        name = self.text[cmd_start:self.pos]

        # Consume one trailing space that some authors put after a command
        if self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1

        # --- dispatch by category ---
        if name in BINARY_OPS:
            if name == "times":
                tt = TokenType.TIMES
            elif name == "cdot":
                tt = TokenType.CDOT
            elif name == "div":
                tt = TokenType.DIV
            elif name in ("pm", "mp"):
                tt = TokenType.PLUS  # ± / ∓ are additive-level operators
            else:
                tt = TokenType.TIMES
            self.tokens.append(Token(tt, BINARY_OPS[name], start))
        elif name in ("left", "right", "big", "Big", "bigg", "Bigg"):
            pass  # sizing commands — skip, the next token is the delimiter itself
        elif name == "lvert" or name == "lVert":
            self.tokens.append(Token(TokenType.PIPE, "|" , start))
        elif name == "rvert" or name == "rVert":
            self.tokens.append(Token(TokenType.PIPE, "|", start))
        elif name == "langle":
            self.tokens.append(Token(TokenType.LANGLE, "\\langle", start))
        elif name == "rangle":
            self.tokens.append(Token(TokenType.RANGLE, "\\rangle", start))
        elif name == "operatorname" or name == "mathrm":
            # \operatorname{f} or \mathrm{f} — read the brace group and emit as VARIABLE
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == "{":
                self.pos += 1  # skip {
                inner_start = self.pos
                depth = 1
                while self.pos < len(self.text) and depth > 0:
                    if self.text[self.pos] == "{":
                        depth += 1
                    elif self.text[self.pos] == "}":
                        depth -= 1
                    if depth > 0:
                        self.pos += 1
                inner = self.text[inner_start:self.pos].strip()
                self.pos += 1  # skip closing }
                self.tokens.append(Token(TokenType.COMMAND, f"\\operatorname{{{inner}}}", start))
            else:
                self.tokens.append(Token(TokenType.COMMAND, f"\\{name}", start))
        elif name == "text" or name == "textrm" or name == "textit" or name == "textbf":
            # \text{...} — skip the brace group content, emit as COMMAND
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == "{":
                self.pos += 1
                depth = 1
                inner_start = self.pos
                while self.pos < len(self.text) and depth > 0:
                    if self.text[self.pos] == "{":
                        depth += 1
                    elif self.text[self.pos] == "}":
                        depth -= 1
                    if depth > 0:
                        self.pos += 1
                inner = self.text[inner_start:self.pos].strip()
                self.pos += 1
                self.tokens.append(Token(TokenType.COMMAND, f"\\{name}{{{inner}}}", start))
            else:
                self.tokens.append(Token(TokenType.COMMAND, f"\\{name}", start))
        else:
            self.tokens.append(Token(TokenType.COMMAND, f"\\{name}", start))

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1


# ---------------------------------------------------------------------------
# 7. Parser (recursive descent)
# ---------------------------------------------------------------------------

class LaTeXParser:
    """Recursive-descent parser that consumes a token list and produces an AST."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- helpers -----------------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, "")

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> Token:
        tok = self._advance()
        if tok.type != ttype:
            raise SyntaxError(
                f"Expected {ttype.value} but got {tok.type.value} ({tok.value!r}) at pos {tok.pos}"
            )
        return tok

    def _match(self, ttype: TokenType) -> Optional[Token]:
        if self._peek().type == ttype:
            return self._advance()
        return None

    def _match_command(self, name: str) -> Optional[Token]:
        tok = self._peek()
        if tok.type == TokenType.COMMAND and tok.value == name:
            return self._advance()
        return None

    def _at_command(self, *names: str) -> bool:
        tok = self._peek()
        return tok.type == TokenType.COMMAND and tok.value in names

    def _parse_brace_group(self) -> ASTNode:
        """Parse ``{ ... }`` and return the inner expression as a GROUP node."""
        self._expect(TokenType.LBRACE)
        inner = self._parse_expression()
        self._expect(TokenType.RBRACE)
        return inner

    # -- grammar entry point -----------------------------------------------

    def parse(self) -> ASTNode:
        node = self._parse_expression()
        return node

    # -- expression --------------------------------------------------------

    def _parse_expression(self) -> ASTNode:
        """expression = relation"""
        return self._parse_relation()

    def _parse_relation(self) -> ASTNode:
        """relation = additive ( ( '=' | '<' | '>' | '\\leq' | … ) additive )*"""
        left = self._parse_additive()
        while True:
            tok = self._peek()
            if tok.type == TokenType.EQUALS:
                self._advance()
                right = self._parse_additive()
                left = ASTNode(NodeType.EQUALITY, [left, right], "=")
            elif tok.type in (TokenType.LT, TokenType.GT):
                op = self._advance()
                right = self._parse_additive()
                left = ASTNode(NodeType.RELATION, [left, right], op.value)
            elif tok.type == TokenType.COMMAND and tok.value in (
                "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv",
                "\\sim", "\\propto",
            ):
                op = self._advance()
                right = self._parse_additive()
                left = ASTNode(NodeType.RELATION, [left, right], op.value)
            elif tok.type == TokenType.COMMAND and tok.value in (
                f"\\{k}" for k in SET_OPS
            ):
                op = self._advance()
                right = self._parse_additive()
                left = ASTNode(NodeType.SET_OP, [left, right], op.value)
            elif self._at_command("\\in", "\\notin", "\\subset", "\\supset",
                                  "\\subseteq", "\\supseteq", "\\cup", "\\cap",
                                  "\\setminus"):
                op = self._advance()
                right = self._parse_additive()
                left = ASTNode(NodeType.SET_OP, [left, right], op.value)
            else:
                break
        return left

    def _parse_additive(self) -> ASTNode:
        """additive = multiplicative ( ( '+' | '-' ) multiplicative )*"""
        left = self._parse_multiplicative()
        while True:
            tok = self._peek()
            if tok.type == TokenType.PLUS:
                self._advance()
                right = self._parse_multiplicative()
                left = ASTNode(NodeType.BINARY_OP, [left, right], "+")
            elif tok.type == TokenType.MINUS:
                self._advance()
                right = self._parse_multiplicative()
                left = ASTNode(NodeType.BINARY_OP, [left, right], "-")
            else:
                break
        return left

    def _parse_multiplicative(self) -> ASTNode:
        """multiplicative = unary ( ('\\times' | '\\cdot' | '/' | '\\div') unary )* 

        Also handles *implicit* multiplication when two atoms sit next to each
        other with no operator between them (e.g. ``2x``, ``ab``).
        """
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok.type in (TokenType.TIMES, TokenType.CDOT):
                op = self._advance()
                right = self._parse_unary()
                left = ASTNode(NodeType.BINARY_OP, [left, right], op.value)
            elif tok.type == TokenType.SLASH:
                self._advance()
                right = self._parse_unary()
                left = ASTNode(NodeType.BINARY_OP, [left, right], "/")
            elif tok.type == TokenType.DIV:
                self._advance()
                right = self._parse_unary()
                left = ASTNode(NodeType.BINARY_OP, [left, right], "\\div")
            elif self._can_implicit_multiply():
                right = self._parse_unary()
                left = ASTNode(NodeType.BINARY_OP, [left, right], "\\cdot")
            else:
                break
        return left

    # Commands that should NOT trigger implicit multiplication (structural / layout)
    _NO_IMPLICIT_MULT_COMMANDS: frozenset[str] = frozenset({
        "\\end", "\\\\", "&",
    })

    def _can_implicit_multiply(self) -> bool:
        """Return True if the next token could start a new factor, implying
        implicit multiplication with the preceding node."""
        tok = self._peek()
        if tok.type in (
            TokenType.NUMBER, TokenType.VARIABLE, TokenType.LPAREN,
            TokenType.LBRACKET, TokenType.LBRACE,
            TokenType.LANGLE,
        ):
            return True
        if tok.type == TokenType.COMMAND:
            # Functions / Greek letters / special constants all start atoms,
            # but structural commands like \end, \\ must NOT.
            if tok.value in self._NO_IMPLICIT_MULT_COMMANDS:
                return False
            return True
        # PIPE is deliberately excluded — it is ambiguous (could close |...|)
        # AMPERSAND is also excluded — it is a column separator in matrices
        return False

    def _parse_unary(self) -> ASTNode:
        """unary = '-' unary | '+' unary | postfix

        A lone ``+`` or ``-`` with nothing following (as in ``0^{+}``) is
        treated as a sign atom (SPECIAL_CONST).
        """
        tok = self._peek()
        if tok.type == TokenType.MINUS:
            self._advance()
            # If the next token cannot start an atom, treat '-' as a standalone sign
            if not self._can_start_atom():
                return ASTNode(NodeType.SPECIAL_CONST, value="-")
            operand = self._parse_unary()
            return ASTNode(NodeType.UNARY_OP, [operand], "-")
        if tok.type == TokenType.PLUS:
            self._advance()
            if not self._can_start_atom():
                return ASTNode(NodeType.SPECIAL_CONST, value="+")
            return self._parse_unary()
        return self._parse_postfix()

    def _can_start_atom(self) -> bool:
        """Return True if the current token can begin an atomic expression."""
        tok = self._peek()
        if tok.type in (
            TokenType.NUMBER, TokenType.VARIABLE, TokenType.LPAREN,
            TokenType.LBRACKET, TokenType.LBRACE, TokenType.PIPE,
            TokenType.LANGLE,
        ):
            return True
        if tok.type == TokenType.COMMAND:
            return True
        return False

    def _parse_postfix(self) -> ASTNode:
        """postfix = power ( '!' )*"""
        node = self._parse_power()
        while self._peek().type == TokenType.BANG:
            self._advance()
            node = ASTNode(NodeType.FACTORIAL, [node], "!")
        return node

    def _parse_power(self) -> ASTNode:
        """power = atom ( '^' atom )?

        Handles ``a^{bc}`` as a superscript group as well as ``a^b``.
        """
        base = self._parse_atom()
        if self._peek().type == TokenType.CARET:
            self._advance()
            if self._peek().type == TokenType.LBRACE:
                exponent = self._parse_brace_group()
            else:
                exponent = self._parse_atom()
            return ASTNode(NodeType.POWER, [base, exponent], "^")
        return base

    # -- atom (the big switch) ---------------------------------------------

    def _parse_atom(self) -> ASTNode:
        """Parse a single atomic expression element."""
        tok = self._peek()

        # --- NUMBER ---
        if tok.type == TokenType.NUMBER:
            self._advance()
            return ASTNode(NodeType.NUMBER, value=tok.value)

        # --- VARIABLE ---
        if tok.type == TokenType.VARIABLE:
            self._advance()
            var_node = ASTNode(NodeType.VARIABLE, value=tok.value)
            # Check for subscript: x_{...}
            if self._peek().type == TokenType.UNDERSCORE:
                self._advance()
                sub = self._parse_subscript_arg()
                var_node = ASTNode(NodeType.SUBSCRIPT, [var_node, sub], "_")
            # Heuristic: if a variable is immediately followed by (, treat as function call
            if self._peek().type == TokenType.LPAREN and var_node.type == NodeType.VARIABLE:
                self._advance()  # (
                arg = self._parse_expression()
                self._match(TokenType.RPAREN)
                return ASTNode(NodeType.FUNCTION_CALL, [arg], tok.value)
            return var_node

        # --- PARENTHESISED GROUP ---
        if tok.type == TokenType.LPAREN:
            return self._parse_paren_group()

        # --- BRACKET GROUP ---
        if tok.type == TokenType.LBRACKET:
            return self._parse_bracket_group()

        # --- ABSOLUTE VALUE | ... | ---
        if tok.type == TokenType.PIPE:
            return self._parse_abs_value()

        # --- ANGLE BRACKET ---
        if tok.type == TokenType.LANGLE:
            return self._parse_angle_group()

        # --- BRACE GROUP (standalone) ---
        if tok.type == TokenType.LBRACE:
            inner = self._parse_brace_group()
            return inner

        # --- BACKSLASH COMMANDS ---
        if tok.type == TokenType.COMMAND:
            return self._parse_command_atom()

        raise SyntaxError(
            f"Unexpected token {tok.type.value} ({tok.value!r}) at pos {tok.pos}"
        )

    def _parse_subscript_arg(self) -> ASTNode:
        """Parse the argument after ``_``, which may be a brace group or a single token."""
        if self._peek().type == TokenType.LBRACE:
            return self._parse_brace_group()
        return self._parse_atom()

    # -- parenthesised / bracket groups ------------------------------------

    def _parse_paren_group(self) -> ASTNode:
        self._expect(TokenType.LPAREN)
        inner = self._parse_expression()
        self._match(TokenType.RPAREN)  # tolerate missing close
        return inner

    def _parse_bracket_group(self) -> ASTNode:
        self._expect(TokenType.LBRACKET)
        inner = self._parse_expression()
        self._match(TokenType.RBRACKET)
        return inner

    def _parse_angle_group(self) -> ASTNode:
        self._expect(TokenType.LANGLE)
        inner = self._parse_expression()
        self._match(TokenType.RANGLE)
        return inner

    def _parse_abs_value(self) -> ASTNode:
        self._expect(TokenType.PIPE)
        inner = self._parse_expression()
        self._match(TokenType.PIPE)
        return ASTNode(NodeType.ABS_VALUE, [inner], "|")

    # -- command dispatch --------------------------------------------------

    def _parse_command_atom(self) -> ASTNode:
        """Dispatch on the current COMMAND token value."""
        tok = self._peek()
        cmd = tok.value  # e.g. "\\frac", "\\alpha", "\\sin"

        # Fractions
        if cmd == "\\frac":
            return self._parse_frac()

        # Square / n-th root
        if cmd == "\\sqrt":
            return self._parse_sqrt()

        # Summation
        if cmd == "\\sum":
            return self._parse_big_op(NodeType.SUM)

        # Product
        if cmd == "\\prod":
            return self._parse_big_op(NodeType.PRODUCT)

        # Integrals
        if cmd in ("\\int", "\\iint", "\\iiint", "\\oint"):
            return self._parse_integral(cmd)

        # Limits
        if cmd == "\\lim":
            return self._parse_limit()

        # Binomial
        if cmd == "\\binom":
            return self._parse_binom()

        # Named functions
        bare = cmd.lstrip("\\")
        if bare in NAMED_FUNCTIONS:
            return self._parse_named_function(cmd)

        # Decorations
        if bare in DECORATIONS:
            return self._parse_decoration(bare)

        # \operatorname{...} / \mathrm{...}
        if cmd.startswith("\\operatorname") or cmd.startswith("\\mathrm"):
            return self._parse_operatorname(cmd)

        # \text{...}
        if cmd.startswith("\\text"):
            self._advance()
            return ASTNode(NodeType.VARIABLE, value=cmd)

        # Special constants (\pi, \infty, \hbar, …)
        if bare in SPECIAL_CONSTANTS:
            self._advance()
            return ASTNode(NodeType.SPECIAL_CONST, value=cmd)

        # Greek letters
        if bare in GREEK_LETTERS:
            self._advance()
            # Check for subscript
            if self._peek().type == TokenType.UNDERSCORE:
                self._advance()
                sub = self._parse_subscript_arg()
                var = ASTNode(NodeType.VARIABLE, value=cmd)
                return ASTNode(NodeType.SUBSCRIPT, [var, sub], "_")
            return ASTNode(NodeType.VARIABLE, value=cmd)

        # Set operators
        if bare in SET_OPS:
            self._advance()
            return ASTNode(NodeType.SET_OP, value=cmd)

        # Arrows
        if bare in ARROW_OPS:
            self._advance()
            return ASTNode(NodeType.VARIABLE, value=cmd)

        # \begin{...} environments
        if cmd == "\\begin":
            return self._parse_begin_env()

        # \left / \right already stripped by the tokenizer; fall through

        # Fallback — treat as a variable
        self._advance()
        return ASTNode(NodeType.VARIABLE, value=cmd)

    # -- \frac -------------------------------------------------------------

    def _parse_frac(self) -> ASTNode:
        self._expect(TokenType.COMMAND)  # consume \frac
        numerator = self._parse_brace_group()
        denominator = self._parse_brace_group()
        return ASTNode(NodeType.FRACTION, [numerator, denominator], "\\frac")

    # -- \sqrt -------------------------------------------------------------

    def _parse_sqrt(self) -> ASTNode:
        self._expect(TokenType.COMMAND)  # consume \sqrt
        # Optional [n] for nth root
        if self._peek().type == TokenType.LBRACKET:
            self._advance()  # [
            index = self._parse_expression()
            self._expect(TokenType.RBRACKET)
            radicand = self._parse_brace_group()
            return ASTNode(NodeType.ROOT, [radicand, index], "\\sqrt")
        radicand = self._parse_brace_group()
        return ASTNode(NodeType.ROOT, [radicand], "\\sqrt")

    # -- \sum / \prod (big operators) --------------------------------------

    def _parse_big_op(self, ntype: NodeType) -> ASTNode:
        """Parse ``\\sum_{lower}^{upper} body`` or ``\\prod_{…}^{…} …``."""
        self._expect(TokenType.COMMAND)
        lower: Optional[ASTNode] = None
        upper: Optional[ASTNode] = None

        if self._peek().type == TokenType.UNDERSCORE:
            self._advance()
            lower = self._parse_subscript_arg()

        if self._peek().type == TokenType.CARET:
            self._advance()
            if self._peek().type == TokenType.LBRACE:
                upper = self._parse_brace_group()
            else:
                upper = self._parse_atom()

        body = self._parse_multiplicative()
        children: list[ASTNode] = []
        if lower is not None:
            children.append(lower)
        if upper is not None:
            children.append(upper)
        children.append(body)
        return ASTNode(ntype, children, "\\sum" if ntype == NodeType.SUM else "\\prod")

    # -- integrals ---------------------------------------------------------

    def _parse_integral(self, cmd: str) -> ASTNode:
        """Parse ``\\int_{a}^{b} f(x) dx`` etc."""
        self._expect(TokenType.COMMAND)
        lower: Optional[ASTNode] = None
        upper: Optional[ASTNode] = None

        if self._peek().type == TokenType.UNDERSCORE:
            self._advance()
            lower = self._parse_subscript_arg()

        if self._peek().type == TokenType.CARET:
            self._advance()
            if self._peek().type == TokenType.LBRACE:
                upper = self._parse_brace_group()
            else:
                upper = self._parse_atom()

        integrand = self._parse_expression()
        children: list[ASTNode] = []
        if lower is not None:
            children.append(lower)
        if upper is not None:
            children.append(upper)
        children.append(integrand)
        return ASTNode(NodeType.INTEGRAL, children, cmd)

    # -- \lim --------------------------------------------------------------

    def _parse_limit(self) -> ASTNode:
        """Parse ``\\lim_{x \\to a} f(x)``."""
        self._expect(TokenType.COMMAND)  # consume \lim
        var: Optional[ASTNode] = None
        target: Optional[ASTNode] = None

        if self._peek().type == TokenType.UNDERSCORE:
            self._advance()
            # The subscript is typically {x \to a}
            if self._peek().type == TokenType.LBRACE:
                self._advance()  # {
                # Parse just the variable (a single atom), then look for \to
                var = self._parse_unary()
                # Expect \to / \rightarrow or subscript superscript notation
                if self._at_command("\\to", "\\rightarrow"):
                    self._advance()
                    target = self._parse_unary()
                self._expect(TokenType.RBRACE)
            else:
                var = self._parse_atom()

        body = self._parse_multiplicative()
        children: list[ASTNode] = []
        if var is not None:
            children.append(var)
        if target is not None:
            children.append(target)
        children.append(body)
        return ASTNode(NodeType.LIMIT, children, "\\lim")

    # -- \binom ------------------------------------------------------------

    def _parse_binom(self) -> ASTNode:
        self._expect(TokenType.COMMAND)  # consume \binom
        n = self._parse_brace_group()
        k = self._parse_brace_group()
        return ASTNode(NodeType.BINOMIAL, [n, k], "\\binom")

    # -- named functions ---------------------------------------------------

    def _parse_named_function(self, cmd: str) -> ASTNode:
        """Parse ``\\sin(x)`` or ``\\sin x`` or ``\\log_{b}(x)``."""
        self._expect(TokenType.COMMAND)
        # Optional subscript (e.g. \log_{b})
        subscript: Optional[ASTNode] = None
        if self._peek().type == TokenType.UNDERSCORE:
            self._advance()
            subscript = self._parse_subscript_arg()
        # Argument may be in parentheses or bare
        if self._peek().type == TokenType.LPAREN:
            self._advance()  # (
            arg = self._parse_expression()
            self._match(TokenType.RPAREN)
        elif self._peek().type == TokenType.LBRACE:
            arg = self._parse_brace_group()
        else:
            arg = self._parse_atom()
        children: list[ASTNode] = [arg]
        if subscript is not None:
            children.append(subscript)
        return ASTNode(NodeType.FUNCTION_CALL, children, cmd)

    # -- decorations -------------------------------------------------------

    def _parse_decoration(self, deco: str) -> ASTNode:
        """Parse ``\\hat{x}``, ``\\vec{x}``, etc."""
        self._advance()  # consume the decoration command
        if self._peek().type == TokenType.LBRACE:
            var = self._parse_brace_group()
        else:
            var = self._parse_atom()
        return ASTNode(NodeType.DECORATED_VAR, [var], f"\\{deco}")

    # -- \operatorname / \mathrm -------------------------------------------

    def _parse_operatorname(self, cmd: str) -> ASTNode:
        """Parse ``\\operatorname{f}(x)`` or ``\\mathrm{f}(x)``."""
        self._advance()
        # cmd is already the full \operatorname{f}
        # Check for parentheses after
        if self._peek().type == TokenType.LPAREN:
            self._advance()
            arg = self._parse_expression()
            self._match(TokenType.RPAREN)
            return ASTNode(NodeType.FUNCTION_CALL, [arg], cmd)
        return ASTNode(NodeType.VARIABLE, value=cmd)

    # -- \begin{...} environments ------------------------------------------

    def _parse_begin_env(self) -> ASTNode:
        """Parse ``\\begin{pmatrix} ... \\end{pmatrix}`` and similar."""
        self._expect(TokenType.COMMAND)  # consume \begin
        self._expect(TokenType.LBRACE)
        # Collect all tokens until RBRACE to form the environment name.
        # The name may be a single VARIABLE token or multiple (e.g. "p", "m", "a", …)
        # because the tokenizer splits consecutive letters into individual tokens.
        parts: list[str] = []
        while self._peek().type not in (TokenType.RBRACE, TokenType.EOF):
            parts.append(self._advance().value)
        env_name = "".join(parts)
        self._expect(TokenType.RBRACE)

        if env_name in ("pmatrix", "bmatrix", "vmatrix", "matrix", "Vmatrix"):
            node = self._parse_matrix_env(env_name)
        elif env_name == "cases":
            node = self._parse_cases_env()
        elif env_name in ("aligned", "align", "array", "split", "gather", "multline"):
            node = self._parse_generic_env(env_name)
        else:
            node = self._parse_generic_env(env_name)

        # Consume \end{env_name}
        self._consume_end(env_name)
        return node

    def _parse_matrix_env(self, env_name: str) -> ASTNode:
        """Parse rows separated by ``\\\\`` and columns separated by ``&``."""
        rows: list[list[ASTNode]] = []
        current_row: list[ASTNode] = []
        while not self._at_end(env_name):
            if self._at_command("\\\\"):
                self._advance()
                rows.append(current_row)
                current_row = []
            elif self._peek().type == TokenType.AMPERSAND:
                self._advance()
                # column separator — continue
            else:
                cell = self._parse_expression()
                current_row.append(cell)
        if current_row:
            rows.append(current_row)

        row_nodes = [
            ASTNode(NodeType.GROUP, cells, f"row_{i}")
            for i, cells in enumerate(rows)
        ]
        return ASTNode(NodeType.MATRIX, row_nodes, env_name)

    def _parse_cases_env(self) -> ASTNode:
        """Parse a cases environment: each line is an expression + condition."""
        cases: list[ASTNode] = []
        while not self._at_end("cases"):
            if self._at_command("\\\\"):
                self._advance()
                continue
            expr = self._parse_expression()
            cond: Optional[ASTNode] = None
            if self._peek().type == TokenType.AMPERSAND:
                self._advance()
                cond = self._parse_expression()
            if cond is not None:
                cases.append(ASTNode(NodeType.GROUP, [expr, cond], "case"))
            else:
                cases.append(expr)
        return ASTNode(NodeType.CASES, cases, "cases")

    def _parse_generic_env(self, env_name: str) -> ASTNode:
        """Fallback parser for unknown environments — just gather content."""
        children: list[ASTNode] = []
        while not self._at_end(env_name):
            if self._at_command("\\\\"):
                self._advance()
                continue
            children.append(self._parse_expression())
        return ASTNode(NodeType.GROUP, children, env_name)

    def _at_end(self, env_name: str) -> bool:
        """Return True if the next tokens are ``\\end{env_name}``."""
        saved = self.pos
        try:
            if self._peek().type != TokenType.COMMAND:
                return False
            tok = self._peek()
            if tok.value != "\\end":
                return False
            # Look ahead
            self._advance()  # skip \end
            if self._peek().type == TokenType.LBRACE:
                self._advance()
                # Collect all tokens until RBRACE (env name may be split)
                parts: list[str] = []
                while self._peek().type not in (TokenType.RBRACE, TokenType.EOF):
                    parts.append(self._advance().value)
                full_name = "".join(parts)
                if full_name == env_name:
                    return True
            return False
        finally:
            self.pos = saved

    def _consume_end(self, env_name: str) -> None:
        """Consume ``\\end{env_name}``."""
        if self._at_command("\\end"):
            self._advance()
            if self._peek().type == TokenType.LBRACE:
                self._advance()
                # Collect all tokens until RBRACE (env name may be split)
                parts: list[str] = []
                while self._peek().type not in (TokenType.RBRACE, TokenType.EOF):
                    parts.append(self._advance().value)
                self._match(TokenType.RBRACE)


# ---------------------------------------------------------------------------
# 8. Public API
# ---------------------------------------------------------------------------

_MATH_DELIMITER_RE = re.compile(r"^\s*\${1,2}|\\\[|\\\(|\\begin\{math\}|\\begin\{equation\}", re.DOTALL)
_MATH_END_RE = re.compile(r"\${1,2}\s*$|\\\]|\\\)|\\end\{math\}|\\end\{equation\}", re.DOTALL)


def _strip_math_delimiters(text: str) -> str:
    """Remove surrounding math-mode delimiters from *text*.

    Handles ``$…$``, ``$$…$$``, ``\\[…\\]``, ``\\(…\\)``,
    ``\\begin{math}…\\end{math}``, and ``\\begin{equation}…\\end{equation}``.
    """
    text = text.strip()
    # $$ … $$
    if text.startswith("$$") and text.endswith("$$") and len(text) > 4:
        return text[2:-2].strip()
    # $ … $
    if text.startswith("$") and text.endswith("$") and len(text) > 2:
        return text[1:-1].strip()
    # \[ … \]
    if text.startswith("\\[") and text.endswith("\\]"):
        return text[2:-2].strip()
    # \( … \)
    if text.startswith("\\(") and text.endswith("\\)"):
        return text[2:-2].strip()
    # \begin{math} … \end{math}
    m = re.match(r"^\\begin\{math\}(.*)\\end\{math\}$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # \begin{equation} … \end{equation}
    m = re.match(r"^\\begin\{equation\}(.*)\\end\{equation\}$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # \begin{equation*} … \end{equation*}
    m = re.match(r"^\\begin\{equation\*\}(.*)\\end\{equation\*\}$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_latex(latex_str: str) -> ASTNode:
    """Parse a LaTeX math expression string into an :class:`ASTNode`.

    Parameters
    ----------
    latex_str : str
        A LaTeX math expression, optionally wrapped in ``$…$``,
        ``$$…$$``, ``\\[…\\]``, etc.

    Returns
    -------
    ASTNode
        The root node of the parsed abstract syntax tree.

    Examples
    --------
    >>> node = parse_latex(r"\\frac{a}{b}")
    >>> node.type
    <NodeType.FRACTION: 'FRACTION'>

    >>> node = parse_latex("$x^2 + 1$")
    >>> node.type
    <NodeType.BINARY_OP: 'BINARY_OP'>
    """
    cleaned = _strip_math_delimiters(latex_str)
    tokenizer = LaTeXTokenizer(cleaned)
    tokens = tokenizer.tokenize()
    parser = LaTeXParser(tokens)
    return parser.parse()


# ---------------------------------------------------------------------------
# 9. Convenience pretty-printer (optional, for debugging)
# ---------------------------------------------------------------------------

def ast_to_string(node: ASTNode, indent: int = 0) -> str:
    """Recursively format an AST as an indented string for debugging."""
    prefix = "  " * indent
    if node.value is not None:
        header = f"{prefix}{node.type.value}[{node.value!r}]"
    else:
        header = f"{prefix}{node.type.value}"
    if not node.children:
        return header
    lines = [header]
    for child in node.children:
        lines.append(ast_to_string(child, indent + 1))
    return "\n".join(lines)
