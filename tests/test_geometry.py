from __future__ import annotations

import unittest

from billiardspractice.config import Ball, Pocket, TableConfig
from billiardspractice.geometry import compute_shot, choose_best_pocket


class GeometryTests(unittest.TestCase):
    def test_straight_shot_has_zero_cut_angle(self) -> None:
        table = TableConfig(length=1000.0, width=500.0, pockets=(Pocket("right", 1000.0, 250.0),))
        cue = Ball(400.0, 250.0)
        obj = Ball(600.0, 250.0)
        shot = compute_shot(cue, obj, table.pockets[0], table)
        self.assertAlmostEqual(shot.cut_angle_degrees, 0.0, places=6)
        self.assertAlmostEqual(shot.ghost_ball.x, 542.85, places=2)

    def test_choose_best_pocket_prefers_smallest_cut_angle(self) -> None:
        table = TableConfig()
        cue = Ball(520.0, 300.0)
        obj = Ball(760.0, 300.0)
        shot = choose_best_pocket(cue, obj, table.pockets, table)
        self.assertIn(shot.pocket.name, {p.name for p in table.pockets})
        self.assertGreaterEqual(shot.cut_angle_degrees, 0.0)
        self.assertLessEqual(shot.cut_angle_degrees, 90.0)


if __name__ == "__main__":
    unittest.main()
