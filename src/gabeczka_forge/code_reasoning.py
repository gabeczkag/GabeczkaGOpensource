import ast
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    line: int


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[Symbol] = []
        self.class_depth = 0

    def _add_target(self, target: ast.AST, kind: str) -> None:
        if isinstance(target, ast.Name):
            self.symbols.append(Symbol(target.id, kind, target.lineno))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._add_target(item, kind)
        elif isinstance(target, ast.Attribute):
            self.symbols.append(Symbol(target.attr, "attribute", target.lineno))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._add_target(target, "variable")
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_target(node.target, "variable")
        self.generic_visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:
        self.symbols.append(Symbol(node.arg, "parameter", node.lineno))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbols.append(Symbol(node.name, "method" if self.class_depth else "function", node.lineno))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.symbols.append(Symbol(node.name, "function", node.lineno))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(Symbol(node.name, "class", node.lineno))
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.symbols.append(Symbol(alias.asname or alias.name.split(".")[0], "import", node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.symbols.append(Symbol(alias.asname or alias.name, "import", node.lineno))

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.symbols.append(Symbol(node.func.id, "function_call", node.lineno))
        elif isinstance(node.func, ast.Attribute):
            self.symbols.append(Symbol(node.func.attr, "method_call", node.lineno))
        self.generic_visit(node)

def _regex_symbols(code: str, language: str) -> list[Symbol]:
    patterns = {
        "rust": [
            (r"\b(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", "function"),
            (r"\b(?:pub\s+)?struct\s+(\w+)", "struct"),
            (r"\b(?:let|const|static)\s+(?:mut\s+)?(\w+)", "variable"),
            (r"\b(?:use|mod)\s+([\w:]+)", "import"),
        ],
        "cpp": [
            (r"\b(?:class|struct)\s+(\w+)", "class"),
            (r"\b(?:const\s+)?(?:auto|bool|char|double|float|int|long|string|void)\s+(\w+)\s*(?:[=;])", "variable"),
            (r"\b(?:[\w:<>]+\s+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", "function"),
            (r"#include\s*[<\"]([^>\"]+)", "import"),
        ],
    }
    symbols: list[Symbol] = []
    for pattern, kind in patterns.get(language.lower(), []):
        for match in re.finditer(pattern, code):
            symbols.append(Symbol(match.group(1), kind, code.count("\n", 0, match.start()) + 1))
    return symbols


def analyze_code(code: str, language: str) -> list[dict[str, object]]:
    if language.lower() == "python":
        try:
            visitor = _PythonSymbolVisitor()
            visitor.visit(ast.parse(code))
            symbols = visitor.symbols
        except SyntaxError:
            symbols = []
    else:
        symbols = _regex_symbols(code, language)
    unique = {(symbol.name, symbol.kind, symbol.line): symbol for symbol in symbols}
    return [asdict(symbol) for symbol in sorted(unique.values(), key=lambda item: (item.line, item.name, item.kind))]
