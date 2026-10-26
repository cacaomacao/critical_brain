"""The historical demonstration must require an explicit launch."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LegacyImport(unittest.TestCase):
    def test_import_does_not_load_scientific_runtime_or_execute_demo(self):
        # A clean interpreter catches accidental work even when the main test
        # process has already imported NumPy or Matplotlib for other tests.
        code = """
import sys
sys.path.insert(0, sys.argv[1])
import Dynamics
assert callable(Dynamics.run_demo)
assert not {'numpy', 'matplotlib', 'scipy'}.intersection(sys.modules)
print('import complete')
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", code, str(ROOT)],
                                    cwd=directory, capture_output=True, text=True,
                                    timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "import complete")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
