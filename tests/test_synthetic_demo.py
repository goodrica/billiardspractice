from __future__ import annotations

import unittest
from pathlib import Path

from billiardspractice.cli import synthetic_demo


class SyntheticDemoTests(unittest.TestCase):
    def test_synthetic_demo_writes_image_and_summary(self) -> None:
        output = Path("synthetic_demo_output.png")
        try:
            data = synthetic_demo(output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertIn("selected_pocket", data)
        finally:
            if output.exists():
                output.unlink()


if __name__ == "__main__":
    unittest.main()
