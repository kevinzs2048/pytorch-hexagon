"""Portability and fail-closed asset checks, independent of torch/DSP availability."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qwen3.bundle import inside, sha, validate, write, SCHEMA


class BundleTests(unittest.TestCase):
    def test_containment_survives_relocation_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(inside(root, "a/b"), root / "a/b")
            for path in ("../old-checkout/file", "/home/someone/file"):
                with self.assertRaises(ValueError):
                    inside(root, path)
            (root / "link").symlink_to(root.parent)
            with self.assertRaises(ValueError):
                inside(root, "link/outside")

    def test_changed_asset_is_rejected_before_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model").write_bytes(b"accepted")
            write(root / "bundle.json", {"schema": SCHEMA, "files": {
                "model": {"sha256": sha(root / "model"), "bytes": 8}}})
            (root / "model").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed bundle file"):
                validate(root)

    def test_missing_asset_is_not_a_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "bundle.json", {"schema": SCHEMA, "files": {
                "missing": {"sha256": "0" * 64, "bytes": 8}}})
            result = subprocess.run([sys.executable, "-m", "qwen3", "doctor", "--bundle", str(root)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Missing or changed bundle file", result.stderr)

    def test_public_help_does_not_require_framework(self):
        result = subprocess.run([sys.executable, "-m", "qwen3", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        for name in ("export", "doctor", "generate", "compare"):
            self.assertIn(name, result.stdout)


if __name__ == "__main__":
    unittest.main()
