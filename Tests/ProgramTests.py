# -*- coding: utf-8 -*-
import unittest

from Program import Program


class ProgramTests(unittest.TestCase):
    def test_compute_minimum_area(self):
        from Drone import Drone
        drone_list = [Drone('test1', 30.0, 18.0, (3840, 2160), 12, 104),
                      Drone('test2', 30.0, 18.0, (3840, 2160), 12, 104),
                      Drone('test3', 30.0, 18.0, (3840, 2160), 6, 104)]
        obtained = Program().compute_minimum_area(drone_list)
        self.assertEqual(obtained, (15.359299586316945, 8.639606017303281))
        self.assertEqual(obtained[0] * obtained[1], 132.6982971275077)

