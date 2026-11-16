"""The deployable core must not depend on the surrounding verification suite."""

import ast
from pathlib import Path
import unittest


class CoreBoundaryTests(unittest.TestCase):
    def test_core_has_no_reference_or_test_dependencies(self):
        root = Path(__file__).resolve().parents[1] / "mini_ep"
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [x.name for x in node.names]
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    self.assertNotIn(
                        name.split(".")[0],
                        {"reference", "validation", "checks", "test"},
                        (path, name),
                    )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertNotIn("reference", node.name, path)


if __name__ == "__main__":
    unittest.main()
