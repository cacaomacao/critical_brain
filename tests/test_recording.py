import importlib.util
import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dcm_izhik


class StubPopulation:
    N = 2

    def __init__(self):
        self.firings = []

    def step(self, external_input):
        self.firings.append(np.array([len(self.firings) % self.N]))
        return None, None, self.firings


class Recording(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_simulation_returns_independent_current_frames_not_cumulative_aliases(self):
        network = dcm_izhik.MultiAgentNetwork.__new__(dcm_izhik.MultiAgentNetwork)
        network.agents = [StubPopulation(), StubPopulation()]
        frames = network.simulate(3)
        self.assertEqual(len(frames), 3)
        self.assertEqual([frame[0].tolist() for frame in frames], [[0], [1], [0]])
        network.agents[0].firings[0][0] = 99
        self.assertEqual(frames[0][0].tolist(), [0])

    def test_degenerate_fit_is_reported_before_regression(self):
        with self.assertRaises(ValueError):
            dcm_izhik.fit_power_law([1, 1, 1], show=False)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy not installed in this runtime')
    def test_fit_has_finite_tail_instead_of_log_zero(self):
        result = dcm_izhik.fit_power_law([1, 1, 2, 4], show=False)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[0], -1.)


if __name__ == '__main__':
    unittest.main()
